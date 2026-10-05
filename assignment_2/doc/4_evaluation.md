# Evaluating a policy (Stage 3)

`stage3_evaluate.py` scores a policy on freshly sampled scenes and reports two things: how often
the can ends up in the commanded compartment, and — in `--all-bins` mode — a confusion matrix
saying *which* compartment it actually went to. The second is what tells you whether the policy
reads the task specification at all.

```bash
# a robomimic checkpoint from stage 2
MUJOCO_GL=egl python stage3_evaluate.py --agent trained_models/bc_rnn_4bin/<run>/models/<ckpt>.pth --all-bins

# your own policy, whatever it was trained with
MUJOCO_GL=egl python stage3_evaluate.py --policy my_policy.py --all-bins
```

Exactly one of `--agent` / `--policy` is required. `MUJOCO_GL=egl` is only needed for `--video`,
but it is harmless to always pass it.

Nothing is replayed from the training set: every scene is a fresh `env.reset()` with a new random
can position. `--seed` fixes the sequence, so two policies compared at the same seed see the same
scenes.

## The two modes

| Mode | Rollouts | What it answers |
| --- | --- | --- |
| default | `--n-scenes` (50) | Success rate against whatever bin `reset()` happened to sample |
| `--all-bins` | `--n-scenes` × 4 (200) | The same, **plus** a confusion matrix |

`--all-bins` replays each scene once per compartment **from the same initial state**, so the only
thing that differs across the four rollouts is the target one-hot in the observation
([stage3:195-198](../pick_place_4bin/stage3_evaluate.py#L195-L198)). That isolation is the point: a policy that ignores the command and
always drives to one bin scores ~25% in the default mode and looks merely mediocre. The confusion
matrix shows it immediately. **Always report `--all-bins`.**

## Reading the output

The bundled rule-based baseline, which solves the task:

```
[scene   0] commanded=bin 0 (  back-left) achieved= bin 0 success=True
[scene   0] commanded=bin 1 ( front-left) achieved= bin 1 success=True
[scene   0] commanded=bin 2 ( back-right) achieved= bin 2 success=True
[scene   0] commanded=bin 3 (front-right) achieved= bin 3 success=True

=== task success ===
  bin 0 (  back-left):   1/  1 = 1.000
  bin 1 ( front-left):   1/  1 = 1.000
  bin 2 ( back-right):   1/  1 = 1.000
  bin 3 (front-right):   1/  1 = 1.000
             overall:   4/  4 = 1.000

=== commanded vs achieved (rows = commanded) ===
          bin0  bin1  bin2  bin3  none
  bin 0      1     0     0     0     0
  bin 1      0     1     0     0     0
  bin 2      0     0     1     0     0
  bin 3      0     0     0     1     0
  commanded bin reached : 4/4 = 1.000
  correct | placed      : 4/4 = 1.000   (of the runs that landed the can in some bin)
```

A BC checkpoint trained on four demonstrations, which does not:

```
=== commanded vs achieved (rows = commanded) ===
          bin0  bin1  bin2  bin3  none
  bin 0      0     0     0     0     2
  bin 1      0     0     0     0     2
  bin 2      0     1     0     0     1
  bin 3      0     0     0     0     2
  commanded bin reached : 0/8 = 0.000
  correct | placed      : 0/1 = 0.000   (of the runs that landed the can in some bin)
```

Three columns of the matrix carry most of the diagnosis:

| Pattern | Reading |
| --- | --- |
| Mass on the **diagonal** | The policy solves the task *and* follows the command |
| Mass in **one column**, all rows | The policy solves pick-and-place but ignores the target one-hot — it always aims at the same bin. Check that `object[-4:]` reaches the network |
| Mass in the **`none`** column | The can never got into any compartment: the grasp fails, the policy stalls, or it runs out of horizon. A manipulation problem, not a conditioning one |

`correct | placed` separates those last two: it is the accuracy *given* the can was placed
somewhere, so a policy that grasps badly but aims correctly still scores high on it.

Two numbers are easy to confuse:

- `success=True` requires the environment's own success check — the can in the commanded
  compartment **and** the gripper backed away from it. Ending a rollout still holding the can over
  the right bin scores `achieved=bin N` but `success=False`.
- The success rate baked into a checkpoint filename (`..._success_0.04.pth`) is from *training-time*
  rollouts: 25 episodes, random bins, at that epoch. It is not comparable to stage 3's number.

## Evaluating a stage 2 checkpoint (`--agent`)

Checkpoints live in the run directory stage 2 prints:

```
trained_models/bc_rnn_4bin/<timestamp>/models/
├── model_epoch_100.pth                               every 100 epochs
├── model_epoch_150_PickPlace4BinCan_success_0.04.pth every time rollout success improves
└── model_epoch_200.pth
```

Pick the highest `success_` value, not the highest epoch. If training was interrupted the newest
file is whatever the last multiple of 100 was — there is no checkpoint at the point of interruption.

The checkpoint carries its own environment metadata and observation spec, so stage 3 rebuilds the
env from it ([stage3:166](../pick_place_4bin/stage3_evaluate.py#L166)) rather than from `default_env_kwargs()`. Nothing has to be kept in sync
by hand, and a checkpoint stays evaluable after you change the defaults.

```bash
MUJOCO_GL=egl python stage3_evaluate.py \
  --agent trained_models/bc_rnn_4bin/20260831192620/models/model_epoch_150_PickPlace4BinCan_success_0.04.pth \
  --all-bins --video
```

## Evaluating an external checkpoint (`--policy`)

Diffusion policy, ACT, or anything else you trained outside this pipeline is evaluated through the
same script. Stage 3 does not know or care what framework produced the weights — it imports your
`.py` file, calls `make_policy(env=...)`, and steps whatever callable comes back
([stage3:58-75](../pick_place_4bin/stage3_evaluate.py#L58-L75)).

So the adapter is a small file that loads your checkpoint and wraps it to satisfy the contract in
[3_train_policy.md](3_train_policy.md#the-policy-contract): a dict of observations in, a `(7,)`
action out, one step at a time.

```python
"""my_diffusion_policy.py -- adapter around a checkpoint trained elsewhere."""
import collections
import numpy as np
import torch

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
CKPT = "/path/to/your/checkpoint.ckpt"


class DiffusionPolicyAdapter:
    def __init__(self, model, normalizer, n_obs_steps=2, n_action_steps=8, device="cuda"):
        self.model, self.norm, self.device = model, normalizer, device
        self.n_obs, self.n_act = n_obs_steps, n_action_steps
        self.start_episode()

    def start_episode(self):
        """Stage 3 calls this before every rollout. Clear ALL per-episode state here."""
        self.history = collections.deque(maxlen=self.n_obs)
        self.queue = collections.deque()

    @torch.no_grad()
    def __call__(self, obs):
        flat = np.concatenate([np.asarray(obs[k]).ravel() for k in OBS_KEYS])   # (23,)
        self.history.append(flat)
        while len(self.history) < self.n_obs:          # pad at the start of an episode
            self.history.appendleft(flat)

        if not self.queue:
            x = np.stack(self.history)[None]                                    # (1, n_obs, 23)
            x = self.norm.normalize_obs(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            chunk = self.model.predict_action(x)["action"][0].cpu().numpy()     # (T, 7)
            chunk = self.norm.denormalize_action(chunk)     # <- back to env units
            self.queue.extend(chunk[: self.n_act])

        return np.clip(self.queue.popleft(), -1.0, 1.0)


def make_policy(env=None, **kwargs):
    ckpt = torch.load(CKPT, map_location="cuda")
    model = build_your_model(ckpt["config"])       # however your framework does it
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to("cuda")
    return DiffusionPolicyAdapter(model, ckpt["normalizer"])
```

Then:

```bash
MUJOCO_GL=egl python stage3_evaluate.py --policy my_diffusion_policy.py --all-bins --n-scenes 50
```

`make_policy` receives the live robosuite env, so anything you need from the scene —
`env.target_bin_placements`, `env.bin1_pos` — is available without hard-coding coordinates. See
[example_policy.py](../pick_place_4bin/example_policy.py) for a complete working file.

### Checklist before you trust a bad number

An external policy that scores 0 is usually an adapter bug, not a training failure. In order of
how often each one is the culprit:

1. **Denormalization.** If you normalized actions for training you must invert it here. Forget it
   and the arm creeps at a fraction of the intended speed, hits the horizon, and every rollout
   lands in the `none` column. Sanity check: print the first few actions — position entries should
   reach ±1, not ±0.1.
2. **`start_episode()`.** Without it the observation history and action queue carry over between
   rollouts. Scene 0 looks fine and everything after it degrades.
3. **Observation key order.** `OBS_KEYS` here must match training exactly. A permuted 23-vector
   fails silently — no exception, just a policy that behaves like it was never trained.
4. **The target one-hot.** If your training pipeline dropped or re-derived `object[-4:]`, the
   policy is unconditioned. The confusion matrix shows this as one populated column.
5. **`model.eval()` and `torch.no_grad()`.** Dropout or batchnorm left in training mode adds noise
   to every action; missing `no_grad` just wastes memory.
6. **Action ordering.** Index 6 is the gripper, and `< 0` opens while `> 0` closes. A flipped sign
   produces a policy that reaches correctly and never picks anything up.
7. **The horizon.** 500 steps at 20 Hz. Demonstrations run 550-800, so a policy that faithfully
   imitates the demonstrator's pace times out. Raise `--horizon` and say that you did.

## Video

`--video` records one scene to mp4. In `--all-bins` mode all four rollouts of that scene are tiled
into a 2×2 grid with a caption bar per panel — green for success, red for failure — which makes a
conditioning failure obvious at a glance (four panels, one destination).

```bash
MUJOCO_GL=egl python stage3_evaluate.py --agent <ckpt> --all-bins --video --video-scene 3 --video-dir ./videos
```

`MUJOCO_GL=egl` is required here: rendering is offscreen.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--agent` | — | robomimic checkpoint (`.pth`) |
| `--policy` | — | python file defining `make_policy()` |
| `--n-scenes` | 50 | scenes to sample (×4 rollouts with `--all-bins`) |
| `--all-bins` | off | replay each scene once per compartment; adds the confusion matrix |
| `--horizon` | 500 | max control steps per rollout (25 s at 20 Hz) |
| `--seed` | 0 | fixes the scene sequence |
| `--video` / `--video-scene` / `--video-dir` | off / 0 / `.` | record one scene |
| `--camera` | `agentview` | camera to record from |

## When it fails

| Symptom | Cause |
| --- | --- |
| `pass exactly one of --agent or --policy` | Both or neither given |
| `does not define make_policy(env=None, **kwargs)` | The policy file has no `make_policy`, or a typo in the name |
| Every rollout ends in the `none` column | The can never reaches a compartment — start with the denormalization and gripper-sign checks above |
| `EGLError: EGL_NOT_INITIALIZED` in a `__del__` traceback after the results printed | Renderer teardown noise; harmless. The results above it are valid |

## Reporting

For a comparison to mean anything, state the checkpoint, the mode, the number of scenes, the seed
and the horizon — and include the confusion matrix, not just the overall number.

```
model_epoch_450_..._success_0.72.pth, --all-bins --n-scenes 50 --seed 0 --horizon 500
overall 0.68 (136/200), commanded bin reached 0.71, correct|placed 0.94
```
