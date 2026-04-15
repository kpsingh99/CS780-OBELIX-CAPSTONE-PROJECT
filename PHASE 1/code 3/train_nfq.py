import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class NFQNet(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return self.net(x)

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
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--k_epochs", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    q_net = NFQNet()
    optimizer = optim.RMSprop(q_net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    epsilon = 1.0
    eps_decay = 0.995
    eps_min = 0.05

    print("Starting NFQ Training...")

    for ep in range(args.episodes):
        # FIXED: Added scaling_factor=5 here
        env = OBELIX(scaling_factor=5, wall_obstacles=args.wall_obstacles, difficulty=0, max_steps=args.max_steps)
        s = env.reset()
        ep_ret = 0.0

        batch_s, batch_a, batch_r, batch_s2, batch_d = [], [], [], [], []

        for _ in range(args.max_steps):
            if random.random() < epsilon:
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    qs = q_net(torch.FloatTensor(s).unsqueeze(0))
                a = int(torch.argmax(qs).item())

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r

            batch_s.append(s)
            batch_a.append(a)
            batch_r.append(r)
            batch_s2.append(s2)
            batch_d.append(done)

            s = s2
            if done: break

        t_s = torch.FloatTensor(np.array(batch_s))
        t_a = torch.LongTensor(batch_a).unsqueeze(1)
        t_r = torch.FloatTensor(batch_r).unsqueeze(1)
        t_s2 = torch.FloatTensor(np.array(batch_s2))
        t_d = torch.FloatTensor(batch_d).unsqueeze(1)

        for _ in range(args.k_epochs):
            with torch.no_grad():
                max_next_q = q_net(t_s2).max(1)[0].unsqueeze(1)
                targets = t_r + args.gamma * max_next_q * (1.0 - t_d)

            current_q = q_net(t_s).gather(1, t_a)
            loss = loss_fn(current_q, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(eps_min, epsilon * eps_decay)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} | Return: {ep_ret:.1f} | Epsilon: {epsilon:.3f}")

    torch.save(q_net.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()