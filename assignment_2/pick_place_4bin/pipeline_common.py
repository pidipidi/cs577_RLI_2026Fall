"""
Shared helpers for the three PickPlace4BinCan pipeline stages.

The one piece of domain knowledge that lives here is how the *target bin* is represented at
each layer: as an index 0-3 the env is commanded with, as a one-hot inside the `object`
observable the policy sees, and as the position of the VisualCan_main marker body baked into
each episode's saved model.xml (which is how playback recovers it).
"""

import os
import re

import numpy as np

NUM_BINS = 4
TARGET_BIN_FILE = "target_bin.json"

# Compartment centers of bin2, in the order PickPlace4BinEnv indexes them. See
# PickPlace4BinEnv._get_placement_initializer(): index 0 is (-x, -y), 1 is (+x, -y),
# 2 is (-x, +y), 3 is (+x, +y), relative to bin2_pos.
TARGET_BIN_PLACEMENTS = np.array(
    [
        [0.0025, 0.1575, 0.8],
        [0.1975, 0.1575, 0.8],
        [0.0025, 0.4025, 0.8],
        [0.1975, 0.4025, 0.8],
    ]
)

# Human-readable position of each compartment, as seen from the agentview camera.
BIN_NAMES = {0: "back-left", 1: "front-left", 2: "back-right", 3: "front-right"}

MARKER_RE = re.compile(r'<body name="VisualCan_main" pos="([^"]+)"')


def target_bin_from_model_xml(xml_str, placements=TARGET_BIN_PLACEMENTS):
    """
    Recovers the commanded bin from the marker body baked into a saved model.xml.

    Args:
        xml_str (str): contents of an episode's model.xml
        placements (np.array): (NUM_BINS, 3) compartment centers

    Returns:
        int: index of the compartment the marker sits in
    """
    match = MARKER_RE.search(xml_str)
    assert match is not None, "no VisualCan_main marker in model.xml"
    marker_pos = np.fromstring(match.group(1), sep=" ")
    dists = np.linalg.norm(np.asarray(placements) - marker_pos, axis=1)
    assert dists.min() < 1e-3, f"marker at {marker_pos} matches no compartment"
    return int(np.argmin(dists))


def achieved_bin(base_env):
    """
    Index of the compartment the Can is resting in, or None if it is outside all of them.

    Args:
        base_env (PickPlace4BinEnv): the unwrapped robosuite env

    Returns:
        int or None: compartment index
    """
    can_pos = base_env.sim.data.body_xpos[base_env.obj_body_id["Can"]]
    b = int(np.argmin(np.linalg.norm(base_env.target_bin_placements - can_pos, axis=1)))
    return None if base_env.not_in_bin(can_pos, b) else b


def format_confusion(conf, title="commanded vs achieved"):
    """
    Renders a commanded-vs-achieved matrix as a fixed-width table.

    Args:
        conf (np.array): (NUM_BINS, NUM_BINS + 1) counts; last column is "no bin"
        title (str): heading to print above the table

    Returns:
        str: printable table
    """
    lines = [f"=== {title} (rows = commanded) ===", "        " + "".join(f"  bin{j}" for j in range(NUM_BINS)) + "  none"]
    for i in range(NUM_BINS):
        lines.append(f"  bin {i} " + "".join(f"{conf[i][j]:6d}" for j in range(NUM_BINS + 1)))
    obeyed, total = sum(conf[i][i] for i in range(NUM_BINS)), conf.sum()
    placed = total - conf[:, NUM_BINS].sum()
    lines.append(f"\n  commanded bin reached : {obeyed}/{total} = {obeyed / max(total, 1):.3f}")
    if placed:
        lines.append(f"  correct | placed      : {obeyed}/{placed} = {obeyed / placed:.3f}"
                     "   (of the runs that landed the can in some bin)")
    return "\n".join(lines)


def format_per_bin(per_bin, label="task success"):
    """
    Renders per-bin [successes, attempts] counters.

    Args:
        per_bin (dict): bin index -> [successes, attempts]
        label (str): heading to print above the table

    Returns:
        str: printable table
    """
    lines = [f"=== {label} ==="]
    for b in sorted(per_bin):
        s, n = per_bin[b]
        if n:
            lines.append(f"  bin {b} ({BIN_NAMES[b]:>11s}): {s:3d}/{n:3d} = {s / n:.3f}")
    s = sum(v[0] for v in per_bin.values())
    n = sum(v[1] for v in per_bin.values())
    lines.append(f"  {'overall':>18s}: {s:3d}/{n:3d} = {s / max(n, 1):.3f}")
    return "\n".join(lines)


def default_env_kwargs(robots="Panda", controller="OSC_POSE"):
    """
    The env configuration all three stages share, as a suite.make() kwargs dict.

    Args:
        robots (str): robot model name
        controller (str): robosuite controller preset

    Returns:
        dict: kwargs for robosuite.make()
    """
    from robosuite import load_controller_config

    return dict(
        env_name="PickPlace4BinCan",
        robots=robots,
        controller_configs=load_controller_config(default_controller=controller),
    )


def resolve_out_path(path, default_name):
    """Treats @path as a directory if it exists or ends in a separator, else as a file path."""
    if path.endswith(os.sep) or os.path.isdir(path):
        return os.path.join(path, default_name)
    return path
