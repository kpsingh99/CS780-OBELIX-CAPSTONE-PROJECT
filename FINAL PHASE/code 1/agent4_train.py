
import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from obelix import OBELIX  


OBS_DIM       = 18
OBS_STACK     = 4
IN_DIM        = OBS_DIM * OBS_STACK
N_ACTIONS     = 5
HIDDEN        = 128
ACTIONS       = ("L45", "L22", "FW", "R22", "R45")


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_actions=N_ACTIONS, hidden=HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.trunk(x)
        v    = self.value_stream(feat)
        a    = self.adv_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class SumTree:
    """Binary SumTree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.write    = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PERBuffer:
    """
    Prioritized Experience Replay buffer.

    Parameters
    ----------
    alpha : float   controls how much prioritization is used (0 = uniform)
    beta  : float   importance-sampling correction start value (annealed to 1)
    """

    def __init__(self, capacity=50_000, alpha=0.6, beta=0.4, beta_inc=1e-4, epsilon=1e-5):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.beta     = beta
        self.beta_inc = beta_inc
        self.epsilon  = epsilon
        self.max_prio = 1.0

    def push(self, transition):
        """transition = (s, a, r, s_next, done)"""
        self.tree.add(self.max_prio ** self.alpha, transition)

    def sample(self, batch_size):
        batch, idxs, weights = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + self.beta_inc)

        min_prob = (np.min(self.tree.tree[-self.tree.capacity:][
            self.tree.tree[-self.tree.capacity:] > 0]) / self.tree.total)
        max_w = (min_prob * self.tree.n_entries) ** (-self.beta)

        for i in range(batch_size):
            s    = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            prob = prio / self.tree.total
            w    = ((prob * self.tree.n_entries) ** (-self.beta)) / max_w
            idxs.append(idx)
            weights.append(w)
            batch.append(data)

        return batch, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            prio = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, prio)
            self.max_prio = max(self.max_prio, prio)

    def __len__(self):
        return self.tree.n_entries


class ObsStack:
    def __init__(self, size=OBS_STACK, obs_dim=OBS_DIM):
        self.size    = size
        self.obs_dim = obs_dim
        self.stack   = deque(maxlen=size)

    def reset(self, obs):
        """Call at the start of every episode with the first observation."""
        for _ in range(self.size):
            self.stack.append(obs.astype(np.float32))
        return self._get()

    def step(self, obs):
        """Call after each env.step()."""
        self.stack.append(obs.astype(np.float32))
        return self._get()

    def _get(self):
        return np.concatenate(list(self.stack), axis=0)


class D3QN_PER_Agent:
    def __init__(self, device="cpu"):
        self.device   = torch.device(device)
        self.online   = DuelingDQN().to(self.device)
        self.target   = DuelingDQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.buffer    = PERBuffer(capacity=BUFFER_SIZE)
        self.obs_stack = ObsStack()

        self.epsilon   = EPS_START
        self.steps     = 0

    def select_action(self, stacked_obs):
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        x = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online(x).argmax(dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.push((s, a, r, s2, done))

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        batch, idxs, weights = self.buffer.sample(BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)

        S  = torch.tensor(np.array(s),  dtype=torch.float32).to(self.device)
        A  = torch.tensor(a,             dtype=torch.long).to(self.device)
        R  = torch.tensor(r,             dtype=torch.float32).to(self.device)
        S2 = torch.tensor(np.array(s2), dtype=torch.float32).to(self.device)
        D  = torch.tensor(d,             dtype=torch.float32).to(self.device)
        W  = torch.tensor(weights,       dtype=torch.float32).to(self.device)

        with torch.no_grad():
            best_a   = self.online(S2).argmax(dim=1, keepdim=True)
            q_target = R + GAMMA * (1 - D) * self.target(S2).gather(1, best_a).squeeze(1)

        q_pred = self.online(S).gather(1, A.unsqueeze(1)).squeeze(1)

        td_errors = (q_pred - q_target).detach().cpu().numpy()
        loss = (W * nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(idxs, td_errors)

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path="weights.pth"):
        torch.save(self.online.state_dict(), path)
        print(f"[✓] Weights saved → {path}")


GAMMA        = 0.99
LR           = 1e-4
BATCH_SIZE   = 32
BUFFER_SIZE  = 20_000
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY_EP = 0.992
EPS_DECAY    = 0.999
TARGET_UPDATE= 500
MAX_EPISODES = 800
MAX_STEPS    = 500
SAVE_EVERY   = 200
WARMUP_STEPS = 1000


def pick_level(episode, total_episodes, mode, fixed_level):
    """
    Returns an integer level (1, 2, or 3).
    - 'fixed'      → always fixed_level
    - 'mixed'      → uniform random across 1/2/3  (good for final phase)
    - 'curriculum' → start on L1, gradually introduce L2 then L3
    """
    if mode == "fixed":
        return fixed_level
    if mode == "mixed":
        return random.choice([1, 2, 3])
    frac = episode / total_episodes
    if frac < 0.33:
        return 1
    if frac < 0.66:
        return random.choice([1, 2])
    return random.choice([1, 2, 3])


def shape_reward(raw_reward, obs, prev_obs, done, success):
    """
    Optional extra shaping on top of environment reward.
    Keep it small so it doesn't swamp the terminal +2000.
    """
    shaped = raw_reward

    if any(obs[:16]):
        shaped += 0.1

    if not any(obs[:17]) and not any(prev_obs[:17]):
        shaped -= 0.5

    return shaped


def train(args):
    agent   = D3QN_PER_Agent(device="cpu")
    rewards_history = []
    success_history = []
    loss_history    = []

    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, MAX_EPISODES + 1):
        level = pick_level(ep, MAX_EPISODES, args.mode, args.level)

        env_kwargs = dict(
            level=level,
            wall_obstacle=args.wall,
            render=False,
        )
        obs        = env_reset(args, level)
        stacked    = agent.obs_stack.reset(obs)
        prev_obs   = obs.copy()

        ep_reward  = 0.0
        ep_loss    = []
        success    = False

        for step in range(MAX_STEPS):
            action_idx = agent.select_action(stacked)
            action_str = ACTIONS[action_idx]

            next_obs, reward, done, info = env_step(args.env, action_str)

            shaped = shape_reward(reward, next_obs, prev_obs, done,
                                  success=info.get("success", False))

            next_stacked = agent.obs_stack.step(next_obs)
            agent.push(stacked, action_idx, shaped, next_stacked, float(done))

            if len(agent.buffer) >= WARMUP_STEPS:
                loss = agent.train_step()
                if loss is not None:
                    ep_loss.append(loss)

            ep_reward += reward
            stacked    = next_stacked
            prev_obs   = next_obs.copy()

            if done:
                success = info.get("success", False)
                break

        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY_EP)

        rewards_history.append(ep_reward)
        success_history.append(int(success))
        if ep_loss:
            loss_history.append(np.mean(ep_loss))

        if ep % 20 == 0:
            recent_r = np.mean(rewards_history[-20:])
            recent_s = np.mean(success_history[-20:]) * 100
            print(f"Ep {ep:4d}/{MAX_EPISODES} | L{level} | "
                  f"AvgReward(20): {recent_r:8.1f} | "
                  f"SuccessRate: {recent_s:5.1f}% | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer)}")

        if ep % SAVE_EVERY == 0:
            ckpt = f"checkpoints/weights_ep{ep}.pth"
            agent.save(ckpt)

    agent.save("weights.pth")
    plot_training(rewards_history, success_history, loss_history)


_env = None

def env_reset(args, level):
    """Reset (or create) the OBELIX env and return initial observation."""
    global _env

    _env = OBELIX(
        scaling_factor=5,
        difficulty=level,
        wall_obstacles=args.wall,
        max_steps=MAX_STEPS
    )
    
    obs = _env.reset()
    args.env = _env
    return np.array(obs, dtype=np.float32)

def env_step(env, action_str):
    """
    Call env.step() with the action string.
    Returns (obs, reward, done, info).
    """
    obs, rew, done = env.step(action_str, render=False)

    info = {"success": rew >= 1000}

    return np.array(obs, dtype=np.float32), float(rew), bool(done), info


def plot_training(rewards, successes, losses):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def smooth(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    axes[0].plot(smooth(rewards), color="steelblue")
    axes[0].set_title("Smoothed Cumulative Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    axes[1].plot(smooth(successes), color="green")
    axes[1].set_title("Smoothed Success Rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rate")

    if losses:
        axes[2].plot(smooth(losses), color="red")
        axes[2].set_title("Smoothed TD Loss")
        axes[2].set_xlabel("Update Step")
        axes[2].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("[✓] Training curves saved → training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  type=str, default="mixed",
                        choices=["mixed", "curriculum", "fixed"],
                        help="Level sampling strategy during training")
    parser.add_argument("--level", type=int, default=3,
                        help="Fixed level (only used when --mode fixed)")
    parser.add_argument("--wall",  action="store_true",
                        help="Enable wall obstacle in training")
    args = parser.parse_args()
    args.env = None

    print(f"[*] Training D3QN+PER | mode={args.mode} | wall={args.wall}")
    train(args)