"""
Stage 2 - train a BC policy on PickPlace4BinCan demonstrations.

Takes the demo.hdf5 that stage 1 wrote and runs the whole robomimic path in one command:
extract observations from the recorded simulator states, split train/validation, build a
config, train. The policy sees the target compartment through the last four entries of the
`object` observable, which are the target_bin_id one-hot.

Three presets, all behaviour cloning:

    rnn   (default)  LSTM + deterministic MSE head. The right default here: the demos come
                     from a single operator or a deterministic script, so the action
                     distribution is effectively unimodal, and the LSTM supplies the phase
                     information (approaching vs. already grasped) that a single observation
                     does not carry.
    mlp              Plain feed-forward + MSE. Fast, and a useful floor to compare against;
                     without history it plateaus early on this task.
    rnn_gmm          LSTM + Gaussian-mixture head. For genuinely multimodal demonstrations -
                     several operators solving the task different ways. Note `gmm.min_std` is
                     raised well above the robomimic default: the OSC_POSE rotation dimensions
                     are constant in scripted data, and a mixture fitting a zero-variance
                     dimension drives its std to the floor and blows the gradients up. Note
                     also that robomimic samples which mixture component to use at every eval
                     step, so an unconfident policy produces jittery actions.

Example:
    python stage2_train_bc.py --dataset ~/demos_4bin/demo.hdf5
    python stage2_train_bc.py --dataset ~/demos_4bin/demo.hdf5 --algo mlp --epochs 300
"""

import argparse
import json
import os

import h5py

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# At or below this many demos, a held-out validation set is not worth the demo it costs.
MIN_DEMOS_FOR_HELD_OUT_SPLIT = 10

# Index offset of the mirror links that let the training demos double as the validation set.
# robomimic sorts demos with int(name[5:]), so a mirror of demo_N has to be named demo_<int> too.
VAL_COPY_OFFSET = 100000


def has_observations(path):
    """Whether @path is already a robomimic-converted dataset (has per-demo obs groups)."""
    with h5py.File(path, "r") as f:
        first = next(iter(f["data"].keys()))
        return "obs" in f["data"][first]


def extract_observations(raw_path, out_path):
    """
    Replays the recorded simulator states to regenerate low-dim observations.

    Args:
        raw_path (str): stage 1's demo.hdf5
        out_path (str): where to write the converted dataset

    Returns:
        str: @out_path
    """
    from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

    args = argparse.Namespace(
        dataset=raw_path,
        output_name=os.path.basename(out_path),
        n=None,
        shaped=False,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        depth=False,
        done_mode=0,
        copy_rewards=False,
        copy_dones=False,
        exclude_next_obs=True,
        compress=False,
    )
    print(f"\n[stage 2] extracting observations: {raw_path} -> {out_path}")
    dataset_states_to_obs(args)
    return out_path


def ensure_env_args(raw_path):
    """
    robomimic needs data.attrs["env_args"]; stage 1 writes env_info. Adds it if missing.

    Args:
        raw_path (str): dataset to patch in place
    """
    with h5py.File(raw_path, "r+") as f:
        d = f["data"]
        if "env_args" in d.attrs:
            return
        env_info = json.loads(d.attrs["env_info"])
        d.attrs["env_args"] = json.dumps(
            {"env_name": env_info["env_name"], "type": 1, "env_kwargs": env_info}
        )
        print("[stage 2] added env_args attribute (derived from env_info)")


def demo_index(name):
    """The integer in a `demo_<N>` group name."""
    return int(name.split("_")[1])


def real_demos(hdf5_file):
    """The demo names in an open dataset, excluding any validation mirror links."""
    return sorted(
        (k for k in hdf5_file["data"] if demo_index(k) < VAL_COPY_OFFSET), key=demo_index
    )


def mirror_name(name):
    """Name of the validation mirror of @name."""
    return "demo_{}".format(demo_index(name) + VAL_COPY_OFFSET)


def remove_validation_mirrors(path):
    """
    Deletes the mirror links, if any. Returns whether any were there.

    Needed whenever a dataset that once fell below the threshold grows past it: a real train/valid
    split enumerates every demo in the file, mirrors included, and would put a trajectory's two
    names on opposite sides of the split.

    Args:
        path (str): converted dataset to clean

    Returns:
        bool: whether anything was removed
    """
    with h5py.File(path, "r+") as f:
        stale = [k for k in f["data"] if demo_index(k) >= VAL_COPY_OFFSET]
        for k in stale:
            del f["data"][k]
    return bool(stale)


def use_training_data_for_validation(path):
    """
    Makes the training demos double as the validation set.

    robomimic asserts that the train and valid filter keys name disjoint demos, so pointing both
    at the same names is rejected outright. Each demo is therefore mirrored under a second name
    via an HDF5 hard link - one trajectory, two names, no data copied - and `valid` is pointed at
    the mirrors. Only the converted dataset is touched, never stage 1's demo.hdf5, and
    `--overwrite` regenerates it from scratch.

    Args:
        path (str): converted dataset to write the links and masks into
    """
    from robomimic.utils.file_utils import create_hdf5_filter_key

    with h5py.File(path, "r+") as f:
        data = f["data"]
        demos = real_demos(f)
        for k in [k for k in data if demo_index(k) >= VAL_COPY_OFFSET]:
            del data[k]  # drops the link only; the trajectory it points at survives
        for k in demos:
            data[mirror_name(k)] = data[k]
        for name in ("train", "valid"):
            if "mask" in f and name in f["mask"]:
                del f["mask"][name]

    create_hdf5_filter_key(hdf5_path=path, demo_keys=demos, key_name="train")
    create_hdf5_filter_key(
        hdf5_path=path, demo_keys=[mirror_name(k) for k in demos], key_name="valid"
    )
    print(f"[stage 2] all {len(demos)} demos mirrored into the valid split; "
          f"validating on the training data")


def build_config(algo, dataset, output_dir, name, epochs, rollout_n, rollout_rate, seed):
    """
    Builds the robomimic config for one of the three presets.

    Args:
        algo (str): "rnn", "mlp" or "rnn_gmm"
        dataset (str): converted dataset path
        output_dir (str): where robomimic writes the run directory
        name (str): experiment name
        epochs (int): number of training epochs
        rollout_n (int): rollouts per evaluation
        rollout_rate (int): evaluate every this many epochs
        seed (int): training seed

    Returns:
        Config: a locked-down robomimic config
    """
    from robomimic.config import config_factory

    config = config_factory("bc")
    with config.values_unlocked():
        config.experiment.name = name
        config.experiment.validate = True
        config.experiment.logging.log_wandb = False
        config.experiment.save.every_n_epochs = 100
        config.experiment.save.on_best_rollout_success_rate = True
        config.experiment.rollout.enabled = True
        config.experiment.rollout.n = rollout_n
        config.experiment.rollout.horizon = 500
        config.experiment.rollout.rate = rollout_rate
        config.experiment.rollout.terminate_on_success = True
        config.experiment.render_video = True

        config.train.data = dataset
        config.train.output_dir = output_dir
        config.train.num_epochs = epochs
        config.train.batch_size = 100
        config.train.seed = seed
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.hdf5_cache_mode = "all"

        config.observation.modalities.obs.low_dim = list(OBS_KEYS)

        lr = config.algo.optim_params.policy.learning_rate
        lr.initial = 1e-4
        lr.decay_factor = 0.3
        lr.epoch_schedule = [int(epochs * 0.65), int(epochs * 0.9)]

        if algo == "mlp":
            config.train.seq_length = 1
            config.algo.rnn.enabled = False
            config.algo.gmm.enabled = False
            config.algo.actor_layer_dims = [1024, 1024]
        else:
            config.train.seq_length = 10
            config.algo.rnn.enabled = True
            config.algo.rnn.horizon = 10
            config.algo.rnn.hidden_dim = 400
            config.algo.rnn.num_layers = 2
            config.algo.actor_layer_dims = []
            config.algo.gmm.enabled = algo == "rnn_gmm"
            # Guards against zero-variance action dimensions collapsing the mixture; the
            # robomimic default of 1e-4 diverges on data whose rotation actions are constant.
            config.algo.gmm.min_std = 0.02
    return config


def main(args):
    dataset = os.path.abspath(args.dataset)

    if has_observations(dataset):
        converted = dataset
        print(f"[stage 2] {dataset} already has observations, using it as-is")
    else:
        ensure_env_args(dataset)
        converted = args.converted or os.path.join(
            os.path.dirname(dataset), os.path.basename(dataset).replace(".hdf5", "") + "_low_dim.hdf5"
        )
        if os.path.exists(converted) and not args.overwrite:
            print(f"[stage 2] reusing existing {converted} (pass --overwrite to regenerate)")
        else:
            extract_observations(dataset, converted)

    from robomimic.scripts.split_train_val import split_train_val_from_hdf5
    import robomimic.utils.torch_utils as TorchUtils

    with h5py.File(converted, "r") as f:
        n_demos = len(real_demos(f))
        has_split = "mask" in f and "train" in f["mask"] and "valid" in f["mask"]
        n_valid = len(f["mask"]["valid"]) if has_split else 0

    if n_demos <= MIN_DEMOS_FOR_HELD_OUT_SPLIT:
        # Below this, holding a demo out costs more than it measures: robomimic's splitter takes
        # int(val_ratio * n_demos), which rounds to zero and then fails deep in the dataloader
        # with "num_samples=0", and even when it rounds to one the validation loss is a sample of
        # size one. Validate on the training demos instead, and say so loudly.
        print(f"[stage 2] WARNING: {n_demos} demos is not enough to hold out a validation set "
              f"(need more than {MIN_DEMOS_FOR_HELD_OUT_SPLIT}).")
        print("[stage 2] WARNING: validating on the TRAINING demos - train and valid are the same "
              "data, so the validation loss measures fit, not generalisation. Ignore it.")
        use_training_data_for_validation(converted)
    else:
        dropped = remove_validation_mirrors(converted)
        if dropped:
            print("[stage 2] dataset has grown past the threshold; dropping the validation "
                  "mirrors and making a real split")
        # int(val_ratio * n_demos) can still round to zero for a low --val-ratio, and an empty
        # valid split fails deep in the dataloader with "num_samples=0"; hold out one demo instead
        val_ratio = args.val_ratio
        if int(val_ratio * n_demos) < 1:
            val_ratio = 1.0 / n_demos
            print(f"[stage 2] --val-ratio {args.val_ratio} rounds to 0 validation demos on "
                  f"{n_demos} demos; using {val_ratio:.2f} so one demo is held out")
        needs_split = not has_split or n_valid == 0 or dropped
        if needs_split or args.overwrite:
            print(f"[stage 2] splitting {n_demos} demos into train/valid ({val_ratio:.0%} validation)")
            split_train_val_from_hdf5(converted, val_ratio=val_ratio)

    output_dir = os.path.abspath(args.output_dir)
    name = args.name or f"bc_{args.algo}_4bin"
    config = build_config(
        args.algo, converted, output_dir, name, args.epochs, args.rollout_n, args.rollout_rate, args.seed
    )

    config_path = os.path.join(output_dir, f"{name}_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config.dump())
    print(f"[stage 2] config written to {config_path}")

    print(f"[stage 2] training '{name}' ({args.algo}) on {n_demos} demos for {args.epochs} epochs\n")
    config.lock()
    from robomimic.scripts.train import train

    train(config, device=TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda))

    run_dirs = sorted(
        os.path.join(output_dir, name, d) for d in os.listdir(os.path.join(output_dir, name))
    )
    models = os.path.join(run_dirs[-1], "models")
    best = sorted(f for f in os.listdir(models) if "success" in f)
    print(f"\n[stage 2] checkpoints in {models}")
    if best:
        print(f"[stage 2] best by rollout success: {best[-1]}")
        print(f"\nnext:  python stage3_evaluate.py --agent {os.path.join(models, best[-1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="demo.hdf5 from stage 1")
    parser.add_argument("--algo", type=str, default="rnn", choices=["rnn", "mlp", "rnn_gmm"])
    parser.add_argument("--output-dir", type=str, default="./trained_models")
    parser.add_argument("--name", type=str, default=None, help="experiment name (default: bc_<algo>_4bin)")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--rollout-n", type=int, default=25, help="rollouts per evaluation during training")
    parser.add_argument("--rollout-rate", type=int, default=50, help="evaluate every this many epochs")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--converted", type=str, default=None, help="path for the observation-extracted dataset")
    parser.add_argument("--overwrite", action="store_true", help="redo extraction and train/val split")
    main(parser.parse_args())
