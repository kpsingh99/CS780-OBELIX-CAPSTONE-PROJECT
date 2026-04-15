import argparse
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from obelix import OBELIX

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)


FRONT_NEAR  = [0, 1, 2, 3] 
SIDE_NEAR   = [4, 5, 6, 7]    
FRONT_FAR   = [8, 9, 10, 11] 
SIDE_FAR    = [12, 13, 14, 15]
IR_BIT      = 16              
STUCK_BIT   = 17              

def get_abstract_state(obs):
    obs = np.array(obs, dtype=np.float32)
    
    stuck = int(obs[STUCK_BIT])
    ir    = int(obs[IR_BIT])
    
    near_active  = any(obs[i] for i in FRONT_NEAR + SIDE_NEAR)
    far_active   = any(obs[i] for i in FRONT_FAR  + SIDE_FAR)
    front_near   = any(obs[i] for i in FRONT_NEAR)
    front_far    = any(obs[i] for i in FRONT_FAR)
    left_active  = int(obs[4] or obs[5] or obs[12] or obs[13])   
    right_active = int(obs[6] or obs[7] or obs[14] or obs[15])  

    if ir:
        phase = 3        
    elif front_near or any(obs[i] for i in SIDE_NEAR):
        phase = 2         
    elif front_far or any(obs[i] for i in SIDE_FAR):
        phase = 1          
    else:
        phase = 0         

    if ir or front_near or front_far:
        box_side = 0       
    elif left_active and not right_active:
        box_side = 1       
    elif right_active and not left_active:
        box_side = 2       
    else:
        box_side = 3       

    return (phase, box_side, stuck)


def state_to_idx(state):
    phase, box_side, stuck = state
    return phase * 8 + box_side * 2 + stuck

N_STATES = 4 * 4 * 2  



GAMMA         = 0.99
ALPHA_START   = 0.5     
ALPHA_END     = 0.05    
EPS_START     = 1.0     
EPS_END       = 0.05    
EPS_DECAY     = 0.995   

MAX_EP_STEPS  = 500    
SAVE_EVERY    = 500
LOG_EVERY     = 50


def shape_reward(raw_reward, obs, prev_obs):
    shaped = raw_reward

    obs      = np.array(obs,      dtype=np.float32)
    prev_obs = np.array(prev_obs, dtype=np.float32)

    if obs[STUCK_BIT] == 1:
        shaped -= 400

    prev_phase = get_abstract_state(prev_obs)[0]
    curr_phase = get_abstract_state(obs)[0]
    if curr_phase > prev_phase:
        shaped += 50 * (curr_phase - prev_phase)   
    elif curr_phase < prev_phase:
        shaped -= 20  

    if curr_phase == 0:
        shaped -= 0.5

    return shaped



def pick_level(ep, total_eps, mode, fixed_level):
    if mode == "fixed":
        return fixed_level
    frac = ep / total_eps
    if mode == "curriculum":
        if frac < 0.30:
            return 1
        elif frac < 0.60:
            return random.choice([1, 2])
        else:
            return random.choice([1, 2, 3])
    # mixed
    if frac < 0.10:
        return 1
    return random.choice([1, 2, 3])


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
    obs, rew, done = _env.step(action_str, render=False)
    info = {"success": rew >= 1000}
    return np.array(obs, dtype=np.float32), float(rew), bool(done), info



class TabularQAgent:
    def __init__(self):
        self.Q       = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.epsilon = EPS_START
        self.visits  = np.zeros((N_STATES, N_ACTIONS), dtype=np.int32)

    def select_action(self, state_idx):
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r, s2, done, alpha):
        target = r + (0.0 if done else GAMMA * np.max(self.Q[s2]))
        self.Q[s, a] += alpha * (target - self.Q[s, a])
        self.visits[s, a] += 1

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path="weights_tabular.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "epsilon": self.epsilon}, f)
        print(f"[✓] Q-table saved → {path}")

    def load(self, path="weights_tabular.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q       = data["Q"]
        self.epsilon = data["epsilon"]

    def print_policy(self):
        """Print the learned policy for each abstract state."""
        phase_names   = ["BLIND", "FAR", "NEAR", "IR"]
        side_names    = ["FRONT", "LEFT", "RIGHT", "UNKNOWN"]
        stuck_names   = ["FREE", "STUCK"]
        print("\n── Learned Policy ──────────────────────────────")
        for phase in range(4):
            for side in range(4):
                for stuck in range(2):
                    idx    = state_to_idx((phase, side, stuck))
                    action = ACTIONS[np.argmax(self.Q[idx])]
                    visits = self.visits[idx].sum()
                    if visits > 0:  # only print visited states
                        print(f"  {phase_names[phase]:5s} | {side_names[side]:7s} | "
                              f"{stuck_names[stuck]:5s} → {action:3s}  "
                              f"(Q={np.max(self.Q[idx]):7.1f}, visits={visits})")
        print("────────────────────────────────────────────────\n")



def train(args):
    agent = TabularQAgent()
    os.makedirs("checkpoints_tabular", exist_ok=True)

    rewards_history = []
    success_history = []

    for ep in range(1, args.episodes + 1):
        level   = pick_level(ep, args.episodes, args.mode, args.level)
        obs     = env_reset(args, level)
        state   = get_abstract_state(obs)
        s_idx   = state_to_idx(state)
        prev_obs = obs.copy()

        ep_reward = 0.0
        success   = False
        alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * (ep / args.episodes)

        for step in range(MAX_EP_STEPS):
            action_idx = agent.select_action(s_idx)
            action_str = ACTIONS[action_idx]

            next_obs, raw_reward, done, info = env_step(action_str)
            shaped = shape_reward(raw_reward, next_obs, prev_obs)

            next_state = get_abstract_state(next_obs)
            ns_idx     = state_to_idx(next_state)

            agent.update(s_idx, action_idx, shaped, ns_idx, done, alpha)

            ep_reward += raw_reward
            prev_obs   = next_obs.copy()
            s_idx      = ns_idx

            if done:
                success = info.get("success", False)
                break

        agent.decay_epsilon()
        rewards_history.append(ep_reward)
        success_history.append(int(success))

        if ep % LOG_EVERY == 0:
            recent_r = np.mean(rewards_history[-LOG_EVERY:])
            recent_s = np.mean(success_history[-LOG_EVERY:]) * 100
            print(f"Ep {ep:5d}/{args.episodes} | L{level} | "
                  f"AvgReward({LOG_EVERY}): {recent_r:8.1f} | "
                  f"SuccessRate: {recent_s:5.1f}% | "
                  f"Eps: {agent.epsilon:.3f} | Alpha: {alpha:.3f}")

        if ep % SAVE_EVERY == 0:
            ckpt = f"checkpoints_tabular/weights_tabular_ep{ep}.pkl"
            agent.save(ckpt)

    agent.save("weights_tabular.pkl")
    agent.print_policy()
    plot_training(rewards_history, success_history)
    return agent



def plot_training(rewards, successes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def smooth(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode="valid") if len(x) >= w else x

    axes[0].plot(smooth(rewards), color="steelblue")
    axes[0].set_title("Smoothed Cumulative Reward")
    axes[0].set_xlabel("Episode")

    axes[1].plot(smooth(successes), color="green")
    axes[1].set_title("Smoothed Success Rate")
    axes[1].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig("training_curves_tabular.png", dpi=150)
    plt.show()
    print("[✓] Curves → training_curves_tabular.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--mode",     default="curriculum",
                        choices=["mixed", "curriculum", "fixed"])
    parser.add_argument("--level",    type=int, default=3)
    parser.add_argument("--wall",     action="store_true")
    args = parser.parse_args()

    print(f"[*] Tabular Q-learning | states={N_STATES} | actions={N_ACTIONS}")
    print(f"    mode={args.mode} | episodes={args.episodes}")
    train(args)