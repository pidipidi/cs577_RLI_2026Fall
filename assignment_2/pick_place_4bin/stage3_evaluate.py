"""
Stage 3 - evaluate a policy on PickPlace4BinCan.

Runs 50 freshly sampled scenes (a new random Can position every time - nothing is replayed
from the training set) and reports success overall and per target bin.

Two modes:

    default      one rollout per scene with a randomly commanded target bin. 50 rollouts.
    --all-bins   every scene is replayed once per compartment from the SAME initial state,
                 so the only thing that differs across the four rollouts is the
                 target_bin_id one-hot in the observation. 200 rollouts, and it produces a
                 commanded-vs-achieved confusion matrix. This is what separates "the policy
                 solves the task" from "the policy always aims at one bin and gets credit a
                 quarter of the time" - a number the default mode cannot tell you.

The policy comes from either a robomimic checkpoint or your own code:

    --agent  path/to/model_epoch_450_....pth
    --policy path/to/my_policy.py

An external policy file must define `make_policy`:

    def make_policy(env=None, **kwargs):
        # env: the robosuite PickPlace4BinCan instance, if you need it
        def policy(obs):
            # obs: dict with "object" (14,), "robot0_eef_pos" (3,), "robot0_eef_quat" (4,),
            #      "robot0_gripper_qpos" (2,). obs["object"][-4:] is the target-bin one-hot.
            return action        # np.array, shape (7,): dx dy dz drx dry drz gripper
        return policy

    Anything callable works. If the returned object has a `start_episode()` method it is
    called at the start of each rollout, which is where a recurrent policy resets its state.

Example:
    python stage3_evaluate.py --agent .../model_epoch_450_....pth
    python stage3_evaluate.py --agent .../model.pth --all-bins --video
    python stage3_evaluate.py --policy my_policy.py --n-scenes 50
"""

import argparse
import importlib.util
import os
from collections import defaultdict

import numpy as np

from pipeline_common import (
    NUM_BINS,
    BIN_NAMES,
    achieved_bin,
    default_env_kwargs,
    format_confusion,
    format_per_bin,
)


def load_external_policy(path, env):
    """
    Imports a user-supplied policy file and calls its make_policy().

    Args:
        path (str): path to a .py file defining make_policy()
        env: the robosuite env, handed to make_policy()

    Returns:
        callable: obs dict -> action array
    """
    spec = importlib.util.spec_from_file_location("user_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "make_policy"), f"{path} does not define make_policy(env=None, **kwargs)"
    policy = module.make_policy(env=env)
    assert callable(policy), f"{path}: make_policy() must return something callable"
    return policy


def build_env_from_scratch(render_offscreen):
    """Creates the env directly, for evaluating a policy that carries no robomimic metadata."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"], "rgb": []}}
    )
    kwargs = default_env_kwargs()
    env_name = kwargs.pop("env_name")  # create_env_from_metadata passes this separately
    return EnvUtils.create_env_from_metadata(
        env_meta={"env_name": env_name, "type": 1, "env_kwargs": kwargs},
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=False,
    )


def rollout(env, policy, horizon, record_frames=False, camera="agentview"):
    """
    Runs one episode from the env's current state.

    Args:
        env (EnvRobosuite): the wrapped env, already reset to the desired state
        policy (callable): obs dict -> action
        horizon (int): max control steps
        record_frames (bool): collect rgb frames for a video
        camera (str): camera to record from

    Returns:
        3-tuple: (bool) task success, (int or None) compartment the can ended in, (list) frames
    """
    if hasattr(policy, "start_episode"):
        policy.start_episode()

    obs = env.get_observation()
    frames, success = [], False
    for t in range(horizon):
        obs, _, _, _ = env.step(np.asarray(policy(obs)))
        if record_frames and t % 3 == 0:
            frames.append(env.render(mode="rgb_array", height=480, width=480, camera_name=camera))
        if env.is_success()["task"]:
            success = True
            break
    return success, achieved_bin(env.base_env), frames


def label_frames(frames, text, ok):
    """Stamps a colored caption bar onto each frame."""
    from PIL import Image, ImageDraw

    out = []
    for f in frames:
        im = Image.fromarray(np.asarray(f))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 30], fill=(15, 90, 40) if ok else (110, 25, 25))
        d.text((8, 9), text, fill=(255, 255, 255))
        out.append(np.asarray(im))
    return out


def write_grid_video(clips, path, fps=20):
    """Tiles four labelled clips into one 2x2 mp4."""
    import imageio

    n = max(len(v) for v in clips.values())
    pad = lambda v: v + [v[-1]] * (n - len(v))
    grid = [
        np.concatenate(
            [np.concatenate([pad(clips[2 * r + c])[i] for c in range(2)], axis=1) for r in range(2)], axis=0
        )
        for i in range(n)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, [g[::2, ::2] for g in grid], fps=fps, quality=6, macro_block_size=8)
    print(f"\nwrote {path}")


def main(args):
    assert bool(args.agent) != bool(args.policy), "pass exactly one of --agent or --policy"

    import torch
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    if args.agent:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=False)
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt, render=False, render_offscreen=args.video, verbose=False
        )
        source = os.path.basename(args.agent)
    else:
        env = build_env_from_scratch(render_offscreen=args.video)
        policy = load_external_policy(args.policy, env.base_env)
        source = os.path.basename(args.policy)

    base = env.base_env
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    conf = np.zeros((NUM_BINS, NUM_BINS + 1), dtype=int)
    per_bin = defaultdict(lambda: [0, 0])
    clips = {}

    print(f"\nevaluating {source} on {args.n_scenes} scenes"
          f"{' x 4 bins' if args.all_bins else ''} (seed {args.seed})\n")

    for scene in range(args.n_scenes):
        env.reset()
        if not args.all_bins:
            bins = [int(base.target_bin_id)]  # whatever reset() sampled
            init_state = None
        else:
            bins = list(range(NUM_BINS))
            init_state = env.get_state()

        for b in bins:
            if init_state is not None:
                env.reset_to(init_state)
                base.set_target_bin_id(b)  # only difference between the four rollouts

            record = args.video and scene == args.video_scene
            success, got, frames = rollout(env, policy, args.horizon, record, args.camera)

            conf[b][NUM_BINS if got is None else got] += 1
            per_bin[b][0] += success
            per_bin[b][1] += 1
            if frames:
                tail = f"-> bin {got}" if got is not None else "-> no bin"
                clips[b] = label_frames(frames, f"commanded bin {b}  {tail}", got == b)
            print(f"[scene {scene:3d}] commanded=bin {b} ({BIN_NAMES[b]:>11s}) "
                  f"achieved={'bin ' + str(got) if got is not None else 'none':>6s} success={success}", flush=True)

    print()
    print(format_per_bin(per_bin))
    print()
    print(format_confusion(conf))

    if clips and len(clips) == NUM_BINS:
        write_grid_video(clips, os.path.join(args.video_dir, "stage3_rollouts.mp4"))
    elif clips:
        import imageio

        os.makedirs(args.video_dir, exist_ok=True)
        path = os.path.join(args.video_dir, "stage3_rollout.mp4")
        imageio.mimwrite(path, next(iter(clips.values())), fps=20, quality=6, macro_block_size=8)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("policy source (pick one)")
    src.add_argument("--agent", type=str, default=None, help="robomimic checkpoint (.pth)")
    src.add_argument("--policy", type=str, default=None, help="python file defining make_policy()")
    parser.add_argument("--n-scenes", type=int, default=50, help="random scenes to evaluate on")
    parser.add_argument("--all-bins", action="store_true", help="replay each scene once per compartment")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="record one scene to mp4")
    parser.add_argument("--video-scene", type=int, default=0)
    parser.add_argument("--video-dir", type=str, default=".")
    parser.add_argument("--camera", type=str, default="agentview")
    main(parser.parse_args())
