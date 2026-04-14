"""
train_ppo_v2.py
===============
PPO with:
  - Much stronger wall penalty shaping
  - Training horizon matched to eval (2000 steps)
  - Larger network for better generalization
  - Return normalization to fix value explosion
  - Linear LR annealing
  - Clipped value loss
  - Mixed curriculum optimized for final phase

Usage
-----
    python train_ppo_v2.py --mode curriculum
    python train_ppo_v2.py --mode mixed
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt

from obelix import OBELIX

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
OBS_DIM   = 18
OBS_STACK = 4
IN_DIM    = OBS_DIM * OBS_STACK  # 72

# ══════════════════════════════════════════════════════════════════════════════
# 2.  NETWORK  — larger than v1 for better generalization
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_actions=N_ACTIONS, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),  # extra layer vs v1
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.Tanh(),
            nn.Linear(128, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, x):
        feat   = self.trunk(x)
        logits = self.actor_head(feat)
        value  = self.critic_head(feat).squeeze(-1)
        return logits, value

    def get_action(self, x):
        logits, value = self.forward(x)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, value, entropy

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value, entropy


# ══════════════════════════════════════════════════════════════════════════════
# 3.  OBSERVATION STACK
# ══════════════════════════════════════════════════════════════════════════════

class ObsStack:
    def __init__(self, size=OBS_STACK, obs_dim=OBS_DIM):
        self.size  = size
        self.stack = deque(maxlen=size)

    def reset(self, obs):
        self.stack.clear()
        for _ in range(self.size):
            self.stack.append(obs.astype(np.float32))
        return self._get()

    def step(self, obs):
        self.stack.append(obs.astype(np.float32))
        return self._get()

    def _get(self):
        return np.concatenate(list(self.stack), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = np.array(self.rewards, dtype=np.float64)
        values  = np.array(self.values  + [last_value], dtype=np.float64)
        dones   = np.array(self.dones,   dtype=np.float64)

        # ── reward normalization ──────────────────────────────────────────────
        # This is the KEY fix — normalizing rewards prevents value explosion
        # that caused the 6e8 loss spikes in v1
        r_mean = rewards.mean()
        r_std  = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        # ── GAE ───────────────────────────────────────────────────────────────
        advantages = np.zeros_like(rewards_norm)
        gae = 0.0
        for t in reversed(range(len(rewards_norm))):
            delta       = rewards_norm[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae         = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.astype(np.float32), advantages.astype(np.float32)

    def get_tensors(self, device):
        obs     = torch.tensor(np.array(self.obs),      dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.actions),  dtype=torch.long).to(device)
        lp      = torch.tensor(np.array(self.log_probs),dtype=torch.float32).to(device)
        vals    = torch.tensor(np.array(self.values),   dtype=torch.float32).to(device)
        return obs, actions, lp, vals

    def __len__(self):
        return len(self.rewards)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Rollout
ROLLOUT_STEPS  = 2048      # larger rollout = more stable PPO updates
N_EPOCHS       = 10        # more epochs per rollout
MINIBATCH_SIZE = 64

# PPO
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF_START = 0.05      # start higher for more exploration
ENT_COEF_END   = 0.005     # anneal down to stabilize policy
MAX_GRAD_NORM  = 0.5

# Discount
GAMMA          = 0.99
GAE_LAMBDA     = 0.95

# Optimiser
LR_START       = 3e-4
LR_END         = 1e-5      # linear annealing

# Training
TOTAL_STEPS    = 2_000_000
MAX_EP_STEPS   = 2000      # KEY FIX: match eval horizon exactly
SAVE_EVERY     = 200_000
LOG_EVERY      = 20

# Reward shaping multipliers
WALL_EXTRA_PENALTY  = 600   # on top of env -200 → total -800 per wall hit
BLIND_PENALTY       = 0.5   # per step with no sensors active
SENSOR_BONUS        = 0.1   # per step with any sensor active (approach reward)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REWARD SHAPING
# ══════════════════════════════════════════════════════════════════════════════

def shape_reward(raw_reward, obs, prev_obs):
    shaped = raw_reward

    # ── wall penalty — the most important shaping signal ─────────────────────
    # obs[17] = stuck_flag. We make wall hits extremely costly during training
    # so the agent strongly learns to rotate away from walls.
    if obs[17] == 1:
        shaped -= WALL_EXTRA_PENALTY

    # ── exploration bonus — reward moving toward box ──────────────────────────
    # Any sonar/IR bit active = agent is near the box = good
    if any(obs[:17]):
        shaped += SENSOR_BONUS
    else:
        # penalize wandering completely blind
        shaped -= BLIND_PENALTY

    # ── progress reward — reward if more sensors active than before ───────────
    # This helps with credit assignment for approaching the box
    prev_active = sum(prev_obs[:17])
    curr_active = sum(obs[:17])
    if curr_active > prev_active:
        shaped += 0.5 * (curr_active - prev_active)

    return shaped


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LEVEL SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def pick_level(total_steps_done, total_steps, mode, fixed_level):
    if mode == "fixed":
        return fixed_level
    frac = total_steps_done / total_steps
    if mode == "curriculum":
        if frac < 0.25:
            return 1
        elif frac < 0.55:
            return random.choice([1, 2])
        else:
            return random.choice([1, 2, 3])
    # mixed
    if frac < 0.10:
        return 1
    return random.choice([1, 2, 3])


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_env = None

def env_reset(args, level):
    global _env
    if _env is None:
        _env = OBELIX(
            scaling_factor=5, 
            difficulty=level, 
            wall_obstacles=args.wall, 
            max_steps=MAX_EP_STEPS
        )
    obs = _env.reset()
    return np.array(obs, dtype=np.float32)

def env_step(action_str):
    # obelix.py returns exactly 3 values: obs, reward, done
    obs, rew, done = _env.step(action_str, render=False)
    
    # Create a dummy info dict since obelix doesn't return one
    info = {"success": rew >= 1000}
    
    return np.array(obs, dtype=np.float32), float(rew), bool(done), info


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PPO UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(model, optimizer, buffer, returns, advantages, device, clip_eps):
    obs_t, actions_t, old_lp_t, old_vals_t = buffer.get_tensors(device)
    returns_t    = torch.tensor(returns,    dtype=torch.float32).to(device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)

    total_loss = 0.0
    n = len(buffer)

    for _ in range(N_EPOCHS):
        idxs = np.random.permutation(n)
        for start in range(0, n, MINIBATCH_SIZE):
            mb = idxs[start:start + MINIBATCH_SIZE]

            new_lp, new_val, entropy = model.evaluate(obs_t[mb], actions_t[mb])

            # policy loss
            ratio  = torch.exp(new_lp - old_lp_t[mb])
            surr1  = ratio * advantages_t[mb]
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            # clipped value loss — prevents value function from changing too fast
            val_clipped = old_vals_t[mb] + torch.clamp(
                new_val - old_vals_t[mb], -clip_eps, clip_eps
            )
            vf_loss = torch.max(
                (new_val    - returns_t[mb]).pow(2),
                (val_clipped - returns_t[mb]).pow(2)
            ).mean()

            entropy_loss = -entropy.mean()

            loss = policy_loss + VF_COEF * vf_loss + current_ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()

    return total_loss


# ══════════════════════════════════════════════════════════════════════════════
# 10.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

current_ent_coef = ENT_COEF_START  # global so ppo_update can access it

def train(args):
    global current_ent_coef

    device    = torch.device("cpu")
    model     = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_START, eps=1e-5)
    buffer    = RolloutBuffer()
    obs_stack = ObsStack()

    os.makedirs("checkpoints", exist_ok=True)

    rewards_history = []
    success_history = []
    loss_history    = []

    total_steps = 0
    ep_count    = 0
    ep_reward   = 0.0
    ep_steps    = 0

    level    = pick_level(0, TOTAL_STEPS, args.mode, args.level)
    obs      = env_reset(args, level)
    stacked  = obs_stack.reset(obs)
    prev_obs = obs.copy()

    print(f"[*] PPO v2 | mode={args.mode} | wall={args.wall} | total_steps={TOTAL_STEPS:,}")
    print(f"    Network: 3-layer trunk, hidden=256")
    print(f"    MAX_EP_STEPS={MAX_EP_STEPS} (matches eval)")
    print(f"    Wall penalty: env(-200) + shaping(-{WALL_EXTRA_PENALTY}) = -{200+WALL_EXTRA_PENALTY} total")

    while total_steps < TOTAL_STEPS:

        # ── anneal LR and entropy linearly ────────────────────────────────────
        progress = total_steps / TOTAL_STEPS
        lr_now   = LR_START + (LR_END - LR_START) * progress
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now
        current_ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * progress

        # ── collect rollout ───────────────────────────────────────────────────
        buffer.clear()

        for _ in range(ROLLOUT_STEPS):
            x = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value, _ = model.get_action(x)

            action_idx = action.item()
            action_str = ACTIONS[action_idx]

            next_obs, raw_reward, done, info = env_step(action_str)
            if raw_reward <= -100:
                done = True
            shaped = shape_reward(raw_reward, next_obs, prev_obs)

            next_stacked = obs_stack.step(next_obs)

            buffer.add(stacked, action_idx, log_prob.item(),
                       shaped, value.item(), float(done))

            ep_reward   += raw_reward
            ep_steps    += 1
            total_steps += 1
            prev_obs     = next_obs.copy()

            if done or ep_steps >= MAX_EP_STEPS:
                rewards_history.append(ep_reward)
                success_history.append(int(info.get("success", False)))
                ep_count += 1

                if ep_count % LOG_EVERY == 0:
                    recent_r = np.mean(rewards_history[-LOG_EVERY:])
                    recent_s = np.mean(success_history[-LOG_EVERY:]) * 100
                    print(f"Ep {ep_count:4d} | Steps {total_steps:8,} | L{level} | "
                          f"AvgR({LOG_EVERY}): {recent_r:8.1f} | "
                          f"Success: {recent_s:5.1f}% | "
                          f"LR: {lr_now:.2e} | Ent: {current_ent_coef:.4f}")

                ep_reward = 0.0
                ep_steps  = 0
                level     = pick_level(total_steps, TOTAL_STEPS, args.mode, args.level)
                obs       = env_reset(args, level)
                stacked   = obs_stack.reset(obs)
                prev_obs  = obs.copy()
            else:
                stacked = next_stacked

        # ── PPO update ────────────────────────────────────────────────────────
        x = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, last_val, _ = model.evaluate(x, torch.tensor([0]))
        last_val = last_val.item()

        returns, advantages = buffer.compute_returns_and_advantages(
            last_val, GAMMA, GAE_LAMBDA
        )
        loss_val = ppo_update(model, optimizer, buffer, returns, advantages,
                              device, CLIP_EPS)
        loss_history.append(loss_val)

        # ── checkpoint ────────────────────────────────────────────────────────
        if total_steps % SAVE_EVERY < ROLLOUT_STEPS:
            ckpt = f"checkpoints3/weights_ppo_v2_{total_steps//1000}k.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"[✓] Checkpoint → {ckpt}")

    torch.save(model.state_dict(), "weights_ppo_v2.pth")
    print("[✓] Final weights → weights_ppo_v2.pth")
    plot_training(rewards_history, success_history, loss_history)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_training(rewards, successes, losses):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def smooth(x, w=30):
        return np.convolve(x, np.ones(w)/w, mode="valid") if len(x) >= w else x

    axes[0].plot(smooth(rewards), color="steelblue")
    axes[0].set_title("Smoothed Reward"); axes[0].set_xlabel("Episode")

    axes[1].plot(smooth(successes), color="green")
    axes[1].set_title("Smoothed Success Rate"); axes[1].set_xlabel("Episode")

    axes[2].plot(losses, color="red", alpha=0.6)
    axes[2].set_title("PPO Loss per Update"); axes[2].set_xlabel("Update")

    plt.tight_layout()
    plt.savefig("training_curves_ppo_v2.png", dpi=150)
    plt.show()
    print("[✓] Curves → training_curves_ppo_v2.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="curriculum",
                        choices=["mixed", "curriculum", "fixed"])
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--wall",  action="store_true")
    args = parser.parse_args()

    train(args)