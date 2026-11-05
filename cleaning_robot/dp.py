"""Model-based methods (dynamic programming)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .env import State
from .model import TabularModel, all_states
from .policy import PolicyMixin


class TabularPolicy(PolicyMixin):
    """Deterministic policy stored as one action per state index."""

    def __init__(self, model: TabularModel, actions: np.ndarray):
        self.model = model
        self.actions = actions

    def act(self, state: State, seed=None) -> int:
        return int(self.actions[self.model.state_index(state)])


def tabulate(model: TabularModel, policy: PolicyMixin) -> np.ndarray:
    """Action of a deterministic policy in every state, in index order."""
    return np.fromiter((policy.act(s) for s in all_states(model.env)), np.int64, count=model.n)

def tabulate_probs(model, policy: PolicyMixin) -> np.ndarray:
    """Action probabilities of a stochastic policy in every state, in index order."""
    return np.array([policy.probs(s) for s in all_states(model.env)])

def policy_evaluation(model: TabularModel, policy: np.ndarray, gamma: float, method_direct: bool = True, tolerance = 1e-6) -> np.ndarray:
    """Value V^pi of a policy, computed with the model.

    method_direct if True calculates the policy evalution using direct method, else using iteration

    tolerance is used in iterative method as stop criteria

    `policy` is either one action per state index (deterministic, shape (n,)) or the
    probability of each action in each state (shape (n, n_actions)).
    Returns V with V[i] the value of the state of index i.

    The model is model.P (one sparse n x n matrix per action, model.P[a][i, j] = P(j | i, a))
    and model.R (reward of each state).
    """
    is_policy_det = (policy.ndim == 1)
    V = np.zeros(model.n)

    if is_policy_det:
        P_pi = sp.vstack([model.P[policy[s]][s] for s in range(model.n)], format="csr")       
    else:
        P_pi = sum(sp.diags(policy[:, a]) @ model.P[a] for a in range(5)) 

    if method_direct:
        I = sp.identity(model.n, format="csc") 
        V = spsolve((I - gamma * P_pi), model.R)
    else:
        stop = False
        while(not stop):
            V_next = model.R + gamma * P_pi @ V
            if np.max(np.abs(V_next - V)) < tolerance:
                stop = True
            else:
                V = V_next

    return V


class ValueIteration(TabularPolicy):
    """Value iteration.

    After construction, self.actions must hold the greedy action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float, tolerance=1e-8):
        V = np.zeros(model.n)
        V_aux = np.full(model.n, np.inf)
        count = 0
        while np.max(np.abs(V - V_aux)) > tolerance:
            V_aux = V.copy()
            V = np.max(model.R[:, None] + gamma * np.stack([model.P[a] @ V_aux for a in range(5)], axis=1), axis=1)
            count += 1
        pi = np.argmax(model.R[:, None] + gamma * np.stack([model.P[a] @ V for a in range(5)], axis=1), axis=1)
        
        self.model = model
        self.actions = pi
        self.V = V
        self.count = count
        



class PolicyIteration(TabularPolicy):
    """Policy iteration.

    After construction, self.actions must hold the action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        pi = np.zeros((model.n,), dtype=int)
        pi_aux = None
        count = 0
        while pi_aux is None or np.any(pi != pi_aux):
            pi_aux = pi.copy()
            V_pi = policy_evaluation(model, pi, gamma, method_direct=False, tolerance=1e-8)
            Q_pi = model.R[:, None] + gamma * np.stack([model.P[a] @ V_pi for a in range(5)], axis=1)
            pi = np.argmax(np.round(Q_pi, 10), axis=1)
            count += 1

        self.model = model
        self.actions = pi
        self.V = V_pi
        self.count = count