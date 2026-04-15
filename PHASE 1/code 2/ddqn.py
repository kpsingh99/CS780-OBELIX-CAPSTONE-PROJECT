from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

OBS_STACK_SIZE = 4          
OBS_DIM        = 18         
IN_DIM         = OBS_DIM * OBS_STACK_SIZE   


class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, n_actions: int = 5, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        v    = self.value_stream(feat)                      
        a    = self.adv_stream(feat)                        
        q = v + (a - a.mean(dim=1, keepdim=True))          
        return q


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
        idx   = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s  = np.stack([it.s  for it in items]).astype(np.float32)
        a  = np.array([it.a  for it in items], dtype=np.int64)
        r  = np.array([it.r  for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self): return len(self.buf)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


class ObsStack:
    def __init__(self, n: int = OBS_STACK_SIZE, obs_dim: int = OBS_DIM):
        self.n       = n
        self.obs_dim = obs_dim
        self.buf: deque = deque(maxlen=n)
        self.reset()

    def reset(self, first_obs: np.ndarray | None = None):
        self.buf.clear()
        fill = np.zeros(self.obs_dim, dtype=np.float32) if first_obs is None else first_obs
        for _ in range(self.n):
            self.buf.append(fill.copy())

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.buf.append(obs.astype(np.float32))
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.buf), axis=0)


SIDE_BITS    = list(range(0, 8))    
FORWARD_BITS = list(range(8, 16))   
IR_BIT       = 16
ATTACH_BIT   = 17

def shape_reward(
    env_reward: float,
    prev_obs:   np.ndarray,
    curr_obs:   np.ndarray,
) -> float:
    bonus = 0.0

    for bit in SIDE_BITS:
        if curr_obs[bit] > prev_obs[bit]:          
            bonus += 0.5

    for bit in FORWARD_BITS:
        if curr_obs[bit] > prev_obs[bit]:
            bonus += 1.0

    if curr_obs[IR_BIT] > prev_obs[IR_BIT]:
        bonus += 2.0

    if curr_obs[ATTACH_BIT] > prev_obs[ATTACH_BIT]:
        bonus += 5.0

    return env_reward + bonus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights.pth")
    ap.add_argument("--episodes",       type=int,   default=5000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--difficulty",     type=int,   default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)

    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--replay",         type=int,   default=100_000)
    ap.add_argument("--warmup",         type=int,   default=2000)
    ap.add_argument("--target_sync",    type=int,   default=2000)
    ap.add_argument("--eps_start",      type=float, default=1.0)
    ap.add_argument("--eps_end",        type=float, default=0.05)
    ap.add_argument("--eps_decay_steps",type=int,   default=200_000)
    ap.add_argument("--seed",           type=int,   default=0)

    ap.add_argument("--no_reward_shaping", action="store_true",
                    help="Disable reward shaping (use raw env reward only)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DuelingDQN(in_dim=IN_DIM)
    tgt = DuelingDQN(in_dim=IN_DIM)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps  = 0

    obs_stack = ObsStack(n=OBS_STACK_SIZE, obs_dim=OBS_DIM)

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        raw_obs = env.reset(seed=args.seed + ep)

        obs_stack.reset(first_obs=raw_obs)
        s = obs_stack.get()         

        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            raw_obs2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            stuck      = (r <= -100)
            train_done = done or stuck

            prev_raw_obs = obs_stack.buf[-1].copy()   
            if not args.no_reward_shaping:
                shaped_r = shape_reward(float(r), prev_raw_obs, raw_obs2)
            else:
                shaped_r = float(r)
            s2 = obs_stack.push(raw_obs2)            
            replay.add(Transition(s=s, a=a, r=shaped_r, s2=s2, done=bool(train_done)))

            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch) and steps % 4 == 0:
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)

                with torch.no_grad():
                    next_q     = q(s2b_t)
                    next_a     = torch.argmax(next_q, dim=1)     
                    next_q_tgt = tgt(s2b_t)
                    next_val   = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)  
                    y          = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done or stuck:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{args.episodes}  "
                f"return={ep_ret:.1f}  "
                f"eps={eps_by_step(steps):.3f}  "
                f"replay={len(replay)}"
            )

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()