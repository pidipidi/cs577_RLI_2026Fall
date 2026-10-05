"""
A worked example of the stage 3 external-policy interface - and a strong baseline.

This is the rule-based state machine the scripted collector uses, rewritten as a stage 3
policy. It reads the commanded compartment out of the observation the same way a learned
policy has to: the last four entries of `object` are the target_bin_id one-hot.

    python stage3_evaluate.py --policy example_policy.py --n-scenes 50

Copy this file as a starting point for your own policy. The only contract is `make_policy`.
"""

import numpy as np

APPROACH_Z = 0.10  # hover height above the can before descending
LIFT_Z = 0.25  # height above the source bin to lift the can to
HOVER_Z = 0.18  # height above the target compartment
RELEASE_Z = 0.07  # height above the target compartment to open the gripper at
POS_TOL = 0.008
KP = 12.0


class ScriptedPickPlace:
    """
    APPROACH_CAN -> GRASP -> LIFT -> HOVER_TARGET -> PLACE -> RETREAT -> DONE

    RETREAT is not cosmetic: the environment's success check requires the gripper to back
    away from the can, so an episode that ends at PLACE is never scored successful even
    though the can is sitting in the right compartment.
    """

    def __init__(self, target_bin_placements, source_bin_z=0.8):
        self.placements = np.asarray(target_bin_placements)
        self.source_bin_z = source_bin_z
        self.start_episode()

    def start_episode(self):
        """Called by stage 3 at the start of every rollout."""
        self.stage = "APPROACH_CAN"
        self.counter = 0
        self.grip = -1.0  # open

    def _delta(self, eef, target):
        return np.clip(KP * (target - eef), -1.0, 1.0)

    def __call__(self, obs):
        obj = np.asarray(obs["object"])
        can = obj[:3]                                     # Can position
        tgt = self.placements[int(np.argmax(obj[-len(self.placements):]))]   # commanded compartment
        eef = np.asarray(obs["robot0_eef_pos"])

        a = np.zeros(7)
        self.counter += 1

        if self.stage == "APPROACH_CAN":
            goal = can + np.array([0, 0, APPROACH_Z])
            a[:3] = self._delta(eef, goal)
            self.grip = -1.0
            if np.linalg.norm(goal - eef) < POS_TOL or self.counter > 60:
                self.stage, self.counter = "GRASP", 0

        elif self.stage == "GRASP":
            goal = can + np.array([0, 0, 0.005])
            if self.counter < 40:
                a[:3] = self._delta(eef, goal)
                self.grip = -1.0
            else:
                self.grip = 1.0  # close
                if self.counter > 55:
                    self.stage, self.counter = "LIFT", 0

        elif self.stage == "LIFT":
            goal = np.array([eef[0], eef[1], self.source_bin_z + LIFT_Z])
            a[:3] = self._delta(eef, goal)
            self.grip = 1.0
            if eef[2] > self.source_bin_z + LIFT_Z - 0.02 or self.counter > 60:
                self.stage, self.counter = "HOVER_TARGET", 0

        elif self.stage == "HOVER_TARGET":
            goal = np.array([tgt[0], tgt[1], tgt[2] + HOVER_Z])
            a[:3] = self._delta(eef, goal)
            self.grip = 1.0
            if np.linalg.norm(goal - eef) < 0.015 or self.counter > 120:
                self.stage, self.counter = "PLACE", 0

        elif self.stage == "PLACE":
            goal = np.array([tgt[0], tgt[1], tgt[2] + RELEASE_Z])
            if self.counter < 45:
                a[:3] = self._delta(eef, goal)
                self.grip = 1.0
            else:
                self.grip = -1.0  # open
                if self.counter > 75:
                    self.stage, self.counter = "RETREAT", 0

        elif self.stage == "RETREAT":
            goal = np.array([eef[0], eef[1], tgt[2] + HOVER_Z + 0.05])
            a[:3] = self._delta(eef, goal)
            self.grip = -1.0
            if eef[2] > tgt[2] + HOVER_Z or self.counter > 60:
                self.stage, self.counter = "DONE", 0

        else:  # DONE -- hold still and let the can settle
            self.grip = -1.0

        a[6] = self.grip
        return a


def make_policy(env=None, **kwargs):
    """
    Stage 3 calls this to build the policy.

    Args:
        env (PickPlace4BinEnv or None): the robosuite env, if the policy needs it. This one
            reads the compartment centers off it rather than hard-coding them, so it keeps
            working if the bin layout is changed.

    Returns:
        callable: obs dict -> action array of shape (7,)
    """
    from pipeline_common import TARGET_BIN_PLACEMENTS

    placements = env.target_bin_placements if env is not None else TARGET_BIN_PLACEMENTS
    source_bin_z = env.bin1_pos[2] if env is not None else 0.8
    return ScriptedPickPlace(placements, source_bin_z)
