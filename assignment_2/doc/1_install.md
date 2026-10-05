# Install

The pipeline needs two editable checkouts — a **patched robosuite** that carries the
`PickPlace4BinCan` environment, and **robomimic** for training and policy loading. Both must be
installed with `pip install -e`, not from PyPI: the environment lives in the working tree, not
in a released package.

Verified on Ubuntu (Linux 6.8), Python 3.12, CUDA 13.0, RTX 4070 Ti SUPER.

## 0. Paths used in this document

Nothing here depends on a particular home directory. Two names stand in for the two locations
that differ per machine:

| Name | Meaning |
| --- | --- |
| `$REPO` | this repository's root — the directory holding `assignment_1/`, `assignment_2/` |
| `$EXT` | wherever you keep external checkouts; any directory you can write to |

Set them once per shell, and every command below can be pasted as written:

```bash
export REPO=$(git rev-parse --show-toplevel)   # run from anywhere inside this repo
export EXT=~/git/external                      # or any directory you prefer
mkdir -p "$EXT"
```

Paths written without a prefix (`assignment_2/pick_place_4bin/`) are relative to `$REPO`.

## 1. Environment

```bash
conda create -n cs577 python=3.12
conda activate cs577
```

## 2. robosuite (patched)

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git "$EXT/robosuite"
cd "$EXT/robosuite"
git checkout v1.4.1
pip install -e .
pip install h5py gymnasium
```

The assignment uses `OSC_POSE`, so the optional IK and rendering packages in
`requirements-extra.txt` are not needed. Installing that entire file can fail while building
`pybullet-svl` because its build code requires `pkg_resources`, which recent setuptools no
longer provides.

**The 4-bin environment is not part of upstream robosuite.** Two files make it exist, and
without them `suite.make(env_name="PickPlace4BinCan")` fails with an unknown-environment error:

| File | What it is |
| --- | --- |
| `robosuite/environments/manipulation/pick_place_4_bin.py` | `PickPlace4BinEnv` / `PickPlace4BinCan` |
| `robosuite/__init__.py` (line 8) | the import that registers them |

The registration line is:

```python
from robosuite.environments.manipulation.pick_place_4_bin import PickPlace4BinEnv, PickPlace4BinCan
```

robosuite registers environments by class definition, so importing the module is what makes the
name resolvable. If you are setting this up on a new machine, copy `pick_place_4_bin.py` across
and add that line.

## 3. robomimic

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git "$EXT/robomimic"
cd "$EXT/robomimic"
pip install -e .
```

The clone location does not matter — an editable install is importable from anywhere, so
`$EXT/robomimic` is only a suggestion. A checkout already sitting inside the repo works just as
well: on the machine this was written on, assignment 1's `assignment_1/robomimic/` is the
editable install both assignments share, and no second clone was needed.

## 4. PyTorch

Install the build that matches your CUDA. This machine runs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

CPU-only works for stages 1 and 3; stage 2 is slow without a GPU.

## Versions known to work

| Package | Version |
| --- | --- |
| python | 3.12.13 |
| robosuite | 1.4.1 (editable, patched) |
| robomimic | 0.4.0 (editable) |
| mujoco | 3.3.0 |
| torch | 2.13.0+cu130 |
| numpy | 1.26.4 |
| h5py | 3.16.0 |
| imageio / imageio-ffmpeg | 2.37.4 / 0.6.0 |
| pynput | 1.8.2 |
| PyOpenGL / egl_probe | 3.1.10 / 1.0.2 |

After all installs above, apply the following tested compatibility versions **before running
Verify**. The unpinned installs can select newer NumPy and MuJoCo versions that fail with
robosuite 1.4.1. SciPy, OpenCV and contourpy are pinned to remain compatible with NumPy 1.26.4.

```bash
pip install numpy==1.26.4 mujoco==3.3.0 scipy==1.15.3 \
  opencv-python==4.11.0.86 contourpy==1.3.2
pip check
```

## Verify

```bash
# environment is registered and constructible
MUJOCO_GL=egl python -c "
import robosuite as suite
from robosuite import load_controller_config
env = suite.make(env_name='PickPlace4BinCan', robots='Panda',
                 controller_configs=load_controller_config(default_controller='OSC_POSE'),
                 has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
env.reset(); print('ok, target bin =', env.target_bin_id)"

# the whole stage-3 path, using the bundled rule-based policy (expect 4/4)
cd "$REPO/assignment_2/pick_place_4bin"
MUJOCO_GL=egl python stage3_evaluate.py --policy example_policy.py --n-scenes 1 --all-bins
```

`MUJOCO_GL=egl` is required for headless offscreen rendering — stage 2's training rollouts and
stage 3's video recording both need it. Teleoperation (stage 1) is the opposite case: it needs a
real display, because the keyboard driver opens an X connection at import time.

## Where things live

This pipeline is **assignment 2**. It shares the environment and the robomimic install with
assignment 1 but keeps its own scripts, datasets and training runs:

```
$REPO/
├── assignment_1/
│   ├── robomimic/                          shared robomimic checkout (pip -e)
│   └── robomimic_data/pickplace4bin/       assignment 1's own dataset and BC run
└── assignment_2/
    ├── pick_place_4bin/                    the three stages
    └── robomimic_data/pickplace4bin/       datasets and runs produced by this pipeline
```
