# Assignment 2 — PickPlace4BinCan

Teleoperated demonstration collection, behavior cloning, and evaluation on the 4-bin
pick-and-place environment.

## Tutorials

Do these first: [0_tutorials.md](doc/0_tutorials.md) — two Colab notebooks, DMP and diffusion
policy, on the standard robomimic pick-and-place environment.

## Instructions

| Step | Document | What you do |
| --- | --- | --- |
| 1 | [Install](doc/1_install.md) | Set up robosuite, robomimic, PyTorch |
| 2 | [Teleoperation](doc/2_teleoperation.md) | Collect demonstrations → `demo.hdf5` |
| 3 | [Training a policy](doc/3_train_policy.md) | Train BC; the interface an external method must satisfy |
| 4 | [Evaluation](doc/4_evaluation.md) | Score a checkpoint, yours or ours |

## Pipeline

Scripts in [pick_place_4bin/](pick_place_4bin/):

| Stage | Script | Role |
| --- | --- | --- |
| 1 | [stage1_collect_teleop.py](pick_place_4bin/stage1_collect_teleop.py) | teleoperated collection → `demo.hdf5` |
| 2 | [stage2_train_bc.py](pick_place_4bin/stage2_train_bc.py) | observation extraction + split + BC training |
| 3 | [stage3_evaluate.py](pick_place_4bin/stage3_evaluate.py) | 50 random scenes; `--agent` checkpoint or `--policy` your file |

```bash
python stage1_collect_teleop.py --directory ~/demos_4bin --episodes-per-bin 10
MUJOCO_GL=egl python stage2_train_bc.py --dataset ~/demos_4bin/demo.hdf5
MUJOCO_GL=egl python stage3_evaluate.py --policy example_policy.py --all-bins
```
