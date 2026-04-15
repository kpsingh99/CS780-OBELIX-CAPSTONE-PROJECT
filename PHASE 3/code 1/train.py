import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class StackEnv:
    def __init__(self, env, stack_size=4):
        self.env = env
        self.stack_size = stack_size
        self.stack = deque(maxlen=stack_size)
        
    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        for _ in range(self.stack_size):
            self.stack.append(obs)
        return np.concatenate(self.stack)
        
    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render=render)
        self.stack.append(obs)
        return np.concatenate(self.stack), reward, done

class DQN(nn.Module):
    def __init__(self, in_dim=72, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=2) # Defaulting to Phase 2
    ap.add_argument("--wall_obstacles", action="store_true")
    
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    q = DQN()
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps: return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    print("Starting Phase 2 DDQN Training (with Frame Stacking)...")

    for ep in range(args.episodes):
        base_env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            seed=args.seed + ep,
        )
        env = StackEnv(base_env, stack_size=4) 
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            
            is_stuck = (r <= -100)
            train_done = done or is_stuck
            scaled_r = float(r) / 100.0

            replay.add(Transition(s=s, a=a, r=scaled_r, s2=s2, done=bool(train_done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch) and steps % 4 == 0:
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t, ab_t, rb_t, s2b_t, db_t = map(torch.tensor, (sb, ab, rb, s2b, db))

                with torch.no_grad():
                    next_a = torch.argmax(q(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done or is_stuck:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f}")

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()