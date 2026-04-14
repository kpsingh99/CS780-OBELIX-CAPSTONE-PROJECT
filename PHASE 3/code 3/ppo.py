import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# --- Stacked Environment Wrapper ---
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

# --- PPO Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, in_dim=72, n_actions=5):
        super().__init__()
        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        # Critic Network (Value)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value(self, x):
        return self.critic(x)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="weights_ppo.pth")
    ap.add_argument("--difficulty", type=int, default=2) 
    ap.add_argument("--wall_obstacles", action="store_true")
    
    # PPO Hyperparameters
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01) # Entropy regularization
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=2048) # Steps per rollout
    ap.add_argument("--minibatch_size", type=int, default=64)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent = ActorCritic()
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    base_env = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=1000,
        wall_obstacles=args.wall_obstacles, difficulty=args.difficulty
    )
    env = StackEnv(base_env, stack_size=4)

    obs = torch.zeros((args.num_steps, 72))
    actions = torch.zeros((args.num_steps,))
    logprobs = torch.zeros((args.num_steps,))
    rewards = torch.zeros((args.num_steps,))
    dones = torch.zeros((args.num_steps,))
    values = torch.zeros((args.num_steps,))

    global_step = 0
    next_obs = torch.Tensor(env.reset())
    next_done = torch.zeros(1)
    
    num_updates = args.total_timesteps // args.num_steps
    
    print("Starting PPO Training...")

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase: Collect trajectory data
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step environment
            next_state, reward, done = env.step(ACTIONS[action.item()], render=False)
            
            # Custom scaling and termination for stuck scenarios
            is_stuck = (reward <= -100)
            train_done = done or is_stuck
            scaled_r = float(reward) / 100.0
            
            rewards[step] = torch.tensor(scaled_r).view(-1)
            next_obs = torch.Tensor(next_state)
            next_done = torch.Tensor([train_done])

            if train_done:
                next_obs = torch.Tensor(env.reset())

        # 2. Advantage Calculation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 3. Policy Optimization Phase
        b_obs = obs.reshape((-1, 72))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.num_steps)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Advantage Normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # Entropy Bonus (Encourages exploration)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if update % 5 == 0:
            print(f"Update {update}/{num_updates} | Avg Reward (unscaled): {rewards.mean().item() * 100:.2f} | Policy Loss: {pg_loss.item():.4f} | Entropy: {entropy_loss.item():.4f}")

    torch.save(agent.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()