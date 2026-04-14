
from __future__ import annotations
import argparse
import random
import numpy as np
from collections import defaultdict

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)  # 5

def obs_to_state(obs) -> int:
    """Convert 18-bit observation list/array to a unique integer state key."""
    state = 0
    for bit in obs:
        state = (state << 1) | int(bit)
    return state

SIDE_BITS    = list(range(0, 8))
FORWARD_BITS = list(range(8, 16))
IR_BIT       = 16
ATTACH_BIT   = 17

def shape_reward(env_reward: float, prev_obs, curr_obs) -> float:
    """Add small bonuses for newly activated sensor bits."""
    bonus = 0.0
    prev = list(prev_obs)
    curr = list(curr_obs)

    for bit in SIDE_BITS:
        if curr[bit] > prev[bit]:
            bonus += 0.5

    for bit in FORWARD_BITS:
        if curr[bit] > prev[bit]:
            bonus += 1.0

    if curr[IR_BIT] > prev[IR_BIT]:
        bonus += 2.0

    if curr[ATTACH_BIT] > prev[ATTACH_BIT]:
        bonus += 5.0

    return env_reward + bonus


class QTable:
    def __init__(self, n_actions: int = N_ACTIONS, init_value: float = 0.0):
        self.n_actions   = n_actions
        self.init_value  = init_value
        # defaultdict: first access for a new state creates a zero-array
        self._table: dict[int, np.ndarray] = defaultdict(
            lambda: np.full(n_actions, init_value, dtype=np.float64)
        )

    def get_q(self, state: int) -> np.ndarray:
        """Return Q-values for all actions in the given state."""
        return self._table[state]

    def update(self, state: int, action: int, target: float, alpha: float):
        """
        Tabular Q-learning update (in-place):
            Q(s,a) ← Q(s,a) + α * (target - Q(s,a))
        """
        q = self._table[state]
        q[action] += alpha * (target - q[action])

    def best_action(self, state: int) -> int:
        """Return the greedy action (argmax Q) for the given state."""
        return int(np.argmax(self._table[state]))

    def save(self, path: str):
        """
        Save as a numpy .npy file containing a dict:
          {"table": {state_int: q_array, ...}, "n_actions": int}
        We convert the defaultdict to a plain dict for serialisation.
        """
        np.save(path, {"table": dict(self._table), "n_actions": self.n_actions}, allow_pickle=True)
        print(f"Q-table saved to {path}  ({len(self._table)} states visited)")

    @classmethod
    def load(cls, path: str) -> "QTable":
        data = np.load(path, allow_pickle=True).item()
        qt = cls(n_actions=data["n_actions"])
        for state, q_arr in data["table"].items():
            qt._table[state] = q_arr.astype(np.float64)
        return qt


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",         type=str,   required=True)
    ap.add_argument("--out",               type=str,   default="qtable.npy")
    ap.add_argument("--episodes",          type=int,   default=8000)
    ap.add_argument("--max_steps",         type=int,   default=1000)
    ap.add_argument("--difficulty",        type=int,   default=0)
    ap.add_argument("--wall_obstacles",    action="store_true")
    ap.add_argument("--box_speed",         type=int,   default=2)
    ap.add_argument("--scaling_factor",    type=int,   default=5)
    ap.add_argument("--arena_size",        type=int,   default=500)
    ap.add_argument("--seed",              type=int,   default=0)

    # --- Q-learning hyperparameters ---
    ap.add_argument("--gamma",             type=float, default=0.99,
                    help="Discount factor")

    # Learning rate: tabular Q is much more sensitive to alpha than neural nets.
    # 0.3 is a good starting point — high enough to learn fast, low enough to
    # not oscillate. Decay it if training is unstable.
    ap.add_argument("--alpha",             type=float, default=0.3,
                    help="Learning rate (step size)")
    ap.add_argument("--alpha_min",         type=float, default=0.01,
                    help="Minimum learning rate after decay")
    ap.add_argument("--alpha_decay",       type=float, default=0.9995,
                    help="Multiply alpha by this factor each episode")

    # Epsilon-greedy exploration schedule
    ap.add_argument("--eps_start",         type=float, default=1.0)
    ap.add_argument("--eps_end",           type=float, default=0.05)
    ap.add_argument("--eps_decay",         type=float, default=0.9995,
                    help="Multiply epsilon by this factor each episode")

    # Reward shaping toggle
    ap.add_argument("--no_reward_shaping", action="store_true")

    # Optimistic initialisation: set Q-values to a positive constant instead of 0.
    # This encourages exploration of unvisited (s,a) pairs because they look
    # optimistically good. Particularly helpful in the sparse Find phase.
    ap.add_argument("--optimistic_init",   type=float, default=5.0,
                    help="Initial Q-value for all (state, action) pairs. "
                         "Set to 0 to disable optimistic initialisation.")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    qt  = QTable(n_actions=N_ACTIONS, init_value=args.optimistic_init)
    eps = args.eps_start
    alpha = args.alpha

    # Tracking
    ep_returns  = []
    success_count = 0

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
        state   = obs_to_state(raw_obs)
        ep_ret  = 0.0

        for step in range(args.max_steps):

            # ---------------------------------------------------------------
            # Epsilon-greedy action selection
            # ---------------------------------------------------------------
            if random.random() < eps:
                action = random.randrange(N_ACTIONS)
            else:
                action = qt.best_action(state)

            # ---------------------------------------------------------------
            # Step the environment
            # ---------------------------------------------------------------
            prev_raw_obs = raw_obs
            raw_obs, env_reward, done = env.step(ACTIONS[action], render=False)
            ep_ret += float(env_reward)

            stuck      = (env_reward <= -100)
            train_done = done or stuck

            # ---------------------------------------------------------------
            # Reward shaping (training only)
            # ---------------------------------------------------------------
            if not args.no_reward_shaping:
                r = shape_reward(float(env_reward), prev_raw_obs, raw_obs)
            else:
                r = float(env_reward)

            # ---------------------------------------------------------------
            # Tabular Q-learning update
            #
            # Standard Q-learning (off-policy, Watkins 1989):
            #
            #   target = r  +  γ * max_a' Q(s', a')    if not terminal
            #   target = r                              if terminal
            #
            # Then:
            #   Q(s, a) ← Q(s, a) + α * (target - Q(s, a))
            #
            # This is the same Bellman update as DDQN but exact:
            # no neural net approximation, no replay buffer lag.
            # ---------------------------------------------------------------
            next_state = obs_to_state(raw_obs)

            if train_done:
                target = r
            else:
                target = r + args.gamma * np.max(qt.get_q(next_state))

            qt.update(state, action, target, alpha)

            state = next_state

            if train_done:
                if done and not stuck:
                    success_count += 1
                break

        ep_returns.append(ep_ret)

        # ---------------------------------------------------------------
        # Decay epsilon and alpha after each episode
        # ---------------------------------------------------------------
        eps   = max(args.eps_end,   eps   * args.eps_decay)
        alpha = max(args.alpha_min, alpha * args.alpha_decay)

        if (ep + 1) % 200 == 0:
            recent     = ep_returns[-200:]
            mean_ret   = np.mean(recent)
            success_r  = success_count / (ep + 1) * 100
            states_seen = len(qt._table)
            print(
                f"Ep {ep+1:>6}/{args.episodes}  "
                f"mean_ret(last200)={mean_ret:>9.1f}  "
                f"eps={eps:.4f}  alpha={alpha:.4f}  "
                f"states_seen={states_seen:>7}  "
                f"success%={success_r:.1f}"
            )

    qt.save(args.out)
    print(f"\nTraining complete. Total successes: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()