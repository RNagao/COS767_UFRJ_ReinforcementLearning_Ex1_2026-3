"""Tabular Q-learning: model-free control from sampled transitions only."""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

from .env import CleaningEnv, State
from .policy import PolicyMixin
from .model import n_states, state_index


class QLearning(PolicyMixin):
    """Tabular Q-learning.

    Learns only from interaction with the environment: env.random_init, env.next_state_sample
    and env.reward. Do not use the model (P and R). state_index(env, state) from
    cleaning_robot.model gives a row of the Q table for each state.

    act() must return the greedy action of the learned Q.
    """

    def __init__(self, env: CleaningEnv, gamma: float, n_steps: int, seed=None, lr: float = 1e-2, epsilon = 0.3, q_init = 0.0):

        # ckpt_dir = 'q_learning_checkpoints'
        # os.makedirs(ckpt_dir, exist_ok=True)

        # tag = f"Ql_gamma{gamma}_n{n_steps}_seed{seed}_lr{lr}_ep{epsilon}_qinit{q_init}"
        # ckpt_path = os.path.join(ckpt_dir, tag + ".pkl")

        self.rng = np.random.default_rng(seed)
        # if os.path.exists(ckpt_path):                      
        #     with open(ckpt_path, 'rb') as f:
        #         ck = pickle.load(f)
        #     Q, t, s = ck['Q'], ck['t'], ck['s']
        #     self.rng.bit_generator.state = ck['rng']
        # else:
        Q = np.full((n_states(env), 5), q_init, dtype=float)
        t = 0
        s = env.random_init(int(self.rng.integers(2**31)))

        # def save():
            # tmp = ckpt_path + ".tmp"
            # with open(tmp, 'wb') as f:
                # pickle.dump({'Q': Q, 't': t, 's': s,
                            #  'rng': self.rng.bit_generator.state}, f, protocol=pickle.HIGHEST_PROTOCOL)
            # os.replace(tmp, ckpt_path)    

        while t < n_steps:
            a = self.select_action(Q[state_index(env, s)], epsilon)
            r = env.reward(s)
            s_next = env.next_state_sample(s, a, int(self.rng.integers(2**31)))
            Q[state_index(env, s), a] = Q[state_index(env, s), a] + lr * (r + gamma * np.max(Q[state_index(env, s_next)]) - Q[state_index(env, s), a])
            s = s_next
            t += 1
            # if t % 10_000 == 0:
            #     save()

        # save()

        self.env = env
        self.actions = np.argmax(Q, axis=1)
        self.Q = Q
        self.count = t

    def select_action(self, Q_s: np.ndarray, epsilon = 0.3):
        random_value = self.rng.random() 
        if random_value < epsilon:
            return self.rng.integers(5) 
        return np.argmax(Q_s)

    def act(self, state: State, seed=None) -> int:
        return self.select_action(self.Q[state_index(self.env, state)], 0.0)
