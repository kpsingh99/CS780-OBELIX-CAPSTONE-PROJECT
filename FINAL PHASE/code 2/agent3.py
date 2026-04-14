import os
import numpy as np
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  
_OBS_STACK = deque(maxlen=4)

def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth") 

    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical

    class ActorCritic(nn.Module):
        def __init__(self, in_dim=72, n_actions=5):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, n_actions)
            )
            self.critic = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        def get_action(self, x):
            logits = self.actor(x)
            return torch.argmax(logits, dim=-1)

    model = ActorCritic()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    
    if len(_OBS_STACK) == 0:
        for _ in range(4):
            _OBS_STACK.append(obs)
    else:
        _OBS_STACK.append(obs)

    stacked_obs = np.concatenate(_OBS_STACK)

    import torch
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        action_idx = _MODEL.get_action(x).item()

    return ACTIONS[action_idx]