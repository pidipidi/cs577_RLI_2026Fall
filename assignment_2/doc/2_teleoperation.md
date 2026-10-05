# Teleoperation (Stage 1)

Collecting human demonstrations for `PickPlace4BinCan`: pick the can out of the source bin
and drop it into the compartment the environment tells you to.

```bash
python stage1_collect_teleop.py --directory ~/demos_4bin --episodes-per-bin 10
python stage1_collect_teleop.py --directory ~/demos_4bin --device spacemouse
```

## What one episode looks like

Every reset places the can at a fresh random position in the source bin and commands one of
the four compartments as the target. The commanded compartment is marked in the scene by a
translucent grey can — that is where the red can has to end up. The terminal prints the same
thing before each attempt.

An episode is **kept only if the can ends up in the commanded compartment**. A bin leaves the
work queue only when an attempt for it succeeds, so a botched attempt is retried rather than
leaving that bin short. The resulting dataset is always balanced across the four bins; the
per-bin counts are printed when collection finishes.

## Controls

| Key | Action |
| --- | --- |
| `w` `a` `s` `d` | move the arm in the x-y plane |
| `r` `f` | move the arm up / down |
| `space` | toggle the gripper open / closed |
| `z` `x` / `t` `g` / `c` `v` | rotate about x / y / z |
| `q` | **abort the current episode** — it is discarded and the same bin is retried |

Lower `--pos-sensitivity` if the arm moves faster than you can steer it.

## When recording starts and stops

Recording begins on the **first `env.step()`**, not on reset. `reset()` only snapshots the
scene XML and the initial simulator state in memory; the episode directory is created, the XML
written, and that initial state stored as `states[0]` when the first action arrives. Reset
without stepping and nothing is written to disk.

In practice: everything from the moment the window comes up is recorded. There is no
"start recording" key, so fumbling around at the beginning goes into the data.

The episode ends when the loop exits, which happens on any of:

- `q` — you aborted
- success, plus **10 more recorded steps** so the can settles
- `--max-steps` (default 1500)

The step budget is a budget of **time**, because the loop is paced at the control frequency
(20 Hz): one control step costs one 20th of a second of real time whether or not you touch a
key. The default 1500 steps is therefore **75 seconds** per attempt, and idling spends that
budget just as fast as steering does. Raise `--max-steps` if you need longer.

There is no way to trim part of an episode. The unit you discard is the whole episode, which
is what `q` does: unsuccessful episodes are skipped entirely when the hdf5 is assembled.

## What gets stored

During collection each episode folder accumulates `model.xml`, `state_*.npz` (flushed every
100 steps) and `target_bin.json`. At the end these are merged into `demo.hdf5` and **the raw
folders are deleted** — `demo.hdf5` is the only artifact.

```
data (attrs: date, time, repository_version, env, env_info)
└── demo_N
    ├── attrs: model_file      full MuJoCo scene XML (initial can pose, target marker)
    │          target_bin_id   commanded compartment, 0-3
    ├── states   (T, 71)  = time(1) + qpos(37) + qvel(33)
    └── actions  (T, 7)   = OSC_POSE delta dx dy dz drx dry drz + gripper
```

The 71-dim state is the whole simulator state: the Panda's 9 joints plus four free objects
(Milk, Bread, Cereal, Can — the first three are moved out of the scene but still occupy slots
in the model).

Two things worth knowing:

- **Observations are not stored.** Stage 2 regenerates them by replaying `states`. Changing
  the observation set later — or adding camera images — needs a re-extraction, not a
  re-collection.
- **`states[i]` is the state the action `actions[i]` was issued from.** The wrapper appends the
  state *after* each action, leaving `states` one longer; the trailing entry is dropped when
  the hdf5 is assembled, which is what lines the two up.

Rewards and done flags are not stored either. Success is evaluated during collection only to
decide what to keep, so every surviving demo is by definition a successful one.

## Requirements

Teleoperation needs a display: the keyboard driver uses `pynput`, which opens an X connection
at import time. Stage 1 cannot run over a bare ssh session — collect at a desk, then copy
`demo.hdf5` to wherever you train. (`--help` works headless; the device import is deferred.)
