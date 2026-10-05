# Preliminary tutorials — DMP and diffusion policy

Two Colab notebooks, to run before the assignment. They use the standard robomimic pick-and-place
dataset.

| Notebook | Approach | What it does with a demonstration |
| --- | --- | --- |
| [N03_pick_place_dmp_tutorial.ipynb](../../tutorials/N03_pick_place_dmp_tutorial.ipynb) | Dynamic Movement Primitives | **Fits** one trajectory and replays it under a new goal. No network |
| [N04_pick_place_dp.ipynb](../../tutorials/N04_pick_place_dp.ipynb) | Diffusion policy | **Trains** a network on many demos, rolled out closed-loop |

The contrast is the point: everything from step 3 of the assignment onwards is the second kind.

## Before you start

- **Set the runtime to GPU** (Runtime → Change runtime type). Tutorial 2 trains a network;
  tutorial 1 is fine on CPU.
- Run cells **in order**. The setup cell pins `numpy==1.26.4`; if Colab asks to restart, restart and
  continue rather than re-running the install.


## Tutorial 1 — DMP

Setup → download dataset → **implement the DMP** → rollout → visualize.


## Tutorial 2 — Diffusion policy

Setup → download dataset → **build the DP model** → **train** → rollout → visualize.



## What carries into the assignment

- The 4-bin environment adds **conditioning**: the target compartment is commanded per episode.
  See [3_train_policy.md](3_train_policy.md#the-policy-contract).
- Tutorial 2's rollout loop has the same shape as stage 3's — a thin adapter connects them.
- Tutorial 1's DMP plays the role [example_policy.py](../pick_place_4bin/example_policy.py) plays
  here: a non-learned baseline.
