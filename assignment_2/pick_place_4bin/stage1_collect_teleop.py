"""
Stage 1 - collect teleoperated demonstrations for PickPlace4BinCan.

Each episode places the Can at a fresh random position in the source bin and commands one of
the four target compartments; you drive the arm with a keyboard or 3D mouse and the episode is
kept only if the Can ends up in the commanded compartment. Target bins are handed out so the
finished dataset is balanced: a bin is only crossed off the list when an episode for it
succeeds, so a botched attempt is retried rather than leaving that bin short.

The output demo.hdf5 matches collect_human_demonstrations.py's layout, plus a `target_bin_id`
attribute per demo, so it feeds straight into stage 2.

Keyboard controls are printed at startup. The two that matter most:
    space   toggle gripper open/close
    q       end the current episode (aborts it - use this to retry a bad attempt)

Example:
    python stage1_collect_teleop.py --directory ~/demos_4bin --episodes-per-bin 10
    python stage1_collect_teleop.py --directory ~/demos_4bin --device spacemouse
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from pipeline_common import (
    NUM_BINS,
    BIN_NAMES,
    TARGET_BIN_FILE,
    default_env_kwargs,
    target_bin_from_model_xml,
)


def collect_one_episode(env, device, bin_id, max_steps, arm="right", env_configuration=None):
    """
    Runs one teleoperated episode with @bin_id commanded as the target compartment.

    Args:
        env (DataCollectionWrapper): the wrapped env, recording to disk
        device (Device): keyboard or spacemouse driver
        bin_id (int): compartment the Can must be placed into
        max_steps (int): give up after this many control steps
        arm (str): which arm to control
        env_configuration (str or None): multi-arm configuration, unused for a single arm

    Returns:
        2-tuple: (bool) whether the task was completed, (int) steps taken
    """
    from robosuite.utils.input_utils import input2action

    # command the bin BEFORE reset: DataCollectionWrapper snapshots model.xml inside reset()
    env.set_next_target_bin_id(bin_id)
    env.reset()
    assert env.target_bin_id == bin_id, f"target bin {env.target_bin_id} != commanded {bin_id}"
    env.render()

    device.start_control()
    success, hold = False, -1
    t = 0

    # Pace the loop at the control frequency. Nothing else throttles it: stepping and rendering
    # this scene run at ~100 Hz, so unpaced the whole 1500-step budget is spent in ~15 seconds of
    # wall clock and the episode ends before you have had time to steer. Paced, a step of
    # simulated time costs a step of real time, and --max-steps reads as seconds * control_freq.
    period = 1.0 / env.unwrapped.control_freq
    next_step = time.perf_counter()

    for t in range(max_steps):
        action, _ = input2action(
            device=device, robot=env.robots[0], active_arm=arm, env_configuration=env_configuration
        )
        if action is None:  # the device asked for a reset -- abort this attempt
            break

        env.step(action)
        env.render()

        next_step += period
        lag = next_step - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        elif lag < -period:  # fell behind (a slow frame); resync instead of accruing debt
            next_step = time.perf_counter()

        if env._check_success():
            success = True
            if hold < 0:
                hold = 10  # keep recording briefly so the can settles
            elif hold == 0:
                break
            else:
                hold -= 1
        else:
            hold = -1

    # DataCollectionWrapper only creates ep_directory on the first step, so this has to happen
    # after the rollout, not after reset(); by now it points at this episode's directory.
    if env.ep_directory is not None:
        with open(os.path.join(env.ep_directory, TARGET_BIN_FILE), "w") as f:
            json.dump({"target_bin_id": int(bin_id)}, f)

    return success, t + 1


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, target_bin_placements):
    """
    Packs the per-episode npz files written by DataCollectionWrapper into a single hdf5.

    Unsuccessful episodes are dropped. Each surviving demo group carries `model_file`,
    `target_bin_id`, `states` and `actions`.

    Args:
        directory (str): directory holding the raw per-episode folders
        out_dir (str): where to write demo.hdf5
        env_info (str): JSON-encoded env configuration
        target_bin_placements (np.array): (NUM_BINS, 3) compartment centers, for episodes with
            no target_bin.json (e.g. collected by an older version of this script)

    Returns:
        2-tuple: (int) number of demos written, (str) path to the hdf5
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps, env_name = 0, None

    for ep_directory in sorted(os.listdir(directory)):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states, actions, success = [], [], False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0 or not success:
            continue

        # the wrapper records the state AFTER the action, leaving one extra state at the end
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        with open(os.path.join(directory, ep_directory, "model.xml"), "r") as xml_f:
            xml_str = xml_f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        bin_path = os.path.join(directory, ep_directory, TARGET_BIN_FILE)
        if os.path.exists(bin_path):
            with open(bin_path, "r") as bin_f:
                target_bin_id = int(json.load(bin_f)["target_bin_id"])
        else:
            target_bin_id = target_bin_from_model_xml(xml_str, target_bin_placements)
        ep_data_grp.attrs["target_bin_id"] = target_bin_id

        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()
    return num_eps, hdf5_path


def make_device(name, pos_sensitivity, rot_sensitivity):
    """
    Builds the requested teleop driver.

    Imports are deferred: robosuite.devices pulls in pynput, which opens an X connection at
    import time, so this fails on a headless machine. Teleoperation needs a display anyway -
    run stage 1 at a desk, not over a bare ssh session.
    """
    if name == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    if name == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    raise ValueError(f"invalid device {name!r}: choose 'keyboard' or 'spacemouse'")


def main(args):
    config = default_env_kwargs(robots=args.robots, controller=args.controller)
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    env_info = json.dumps(config)

    os.makedirs(args.directory, exist_ok=True)
    tmp_directory = os.path.join(args.directory, "raw_{}".format(str(time.time()).replace(".", "_")))
    env = DataCollectionWrapper(env, tmp_directory)

    target_bin_placements = np.array(env.unwrapped.target_bin_placements)
    device = make_device(args.device, args.pos_sensitivity, args.rot_sensitivity)

    # A bin leaves the queue only when an episode for it succeeds, so failed attempts are
    # retried instead of leaving that bin under-represented.
    remaining = {b: args.episodes_per_bin for b in range(NUM_BINS)}
    attempts, kept = 0, 0

    print(f"\ncollecting {args.episodes_per_bin} demos per bin ({args.episodes_per_bin * NUM_BINS} total)")
    print("press 'q' to abort the current episode and retry it\n")

    while sum(remaining.values()) > 0 and attempts < args.max_attempts:
        bin_id = max(remaining, key=lambda b: (remaining[b], -b))
        left = sum(remaining.values())
        print(f"[attempt {attempts + 1}] target = bin {bin_id} ({BIN_NAMES[bin_id]}) | {left} demos to go", flush=True)

        success, steps = collect_one_episode(env, device, bin_id, args.max_steps, args.arm)
        attempts += 1
        if success:
            remaining[bin_id] -= 1
            kept += 1
            print(f"    kept ({steps} steps)", flush=True)
        else:
            print(f"    discarded ({steps} steps) -- bin {bin_id} will be retried", flush=True)

    env.close()

    if sum(remaining.values()) > 0:
        print(f"\nstopped after {args.max_attempts} attempts with {remaining} still outstanding")

    num_eps, hdf5_path = gather_demonstrations_as_hdf5(
        tmp_directory, args.directory, env_info, target_bin_placements
    )
    shutil.rmtree(tmp_directory)

    with h5py.File(hdf5_path, "r") as f:
        counts = np.bincount(
            [int(f["data"][k].attrs["target_bin_id"]) for k in f["data"].keys()], minlength=NUM_BINS
        )
    print(f"\nkept {kept}/{attempts} attempts")
    print(f"per-bin demos: {counts.tolist()}")
    print(f"wrote {num_eps} demos -> {hdf5_path} ({os.path.getsize(hdf5_path) / 1e6:.1f} MB)")
    print(f"\nnext:  python stage2_train_bc.py --dataset {hdf5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=str, required=True, help="where to write demo.hdf5")
    parser.add_argument("--episodes-per-bin", type=int, default=10, help="successful demos to collect per bin")
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--robots", type=str, default="Panda")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera", type=str, default="agentview", help="viewpoint shown while teleoperating")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--max-steps", type=int, default=1500, help="give up on an episode after this many steps")
    parser.add_argument("--max-attempts", type=int, default=1000, help="safety stop for the retry loop")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    main(parser.parse_args())
