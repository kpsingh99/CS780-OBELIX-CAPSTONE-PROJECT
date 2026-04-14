"""Submission template.

Edit `policy()` to generate actions from an observation.
The evaluator will import this file and call `policy(obs, rng)`.

Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values are 0/1.

Used Epsilon-Greedy approach.
"""

from typing import Sequence

import numpy as np


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return one action for the current observation."""
    # Baseline: biased random walk that mostly goes forward.
    # Replace with your own logic.
    l = np.sum(obs[0:4])
    f = np.sum(obs[4:12])
    r = np.sum(obs[12:16])
    red_sensor = obs[16]
    stuck = obs[17]
    
    action = "FW"
    epsilon = 0.0 

    if stuck == 1:
        action, epsilon = "L45", 0.10

    elif red_sensor == 1:
        action, epsilon = "FW", 0.0

    elif f > 0:
        action, epsilon = "FW", 0.05

    elif l > r:
        action, epsilon = "L22", 0.05

    elif r > l:
        action, epsilon = "R22", 0.05

    else:
        action, epsilon = "FW", 0.40  

    if rng.random() < epsilon:
        # EXPLORATION
        probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)
        return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]
    else:
        # EXPLOITATION
        return action
