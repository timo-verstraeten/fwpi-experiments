import numpy as np

import scipy
from scipy.stats import multivariate_normal

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class MountainCarContinuous(Continuous_MountainCarEnv):

    def __init__(self, power=0.0015):
        super().__init__()
        self.power = power

    def reset(self):
        s = super().reset().copy()
        s[1] /= self.max_speed
        return s

    def step(self, a):
        s_, r, done, _ = super().step(a)
        return np.array([s_[0], s_[1] / self.max_speed]), r, done, {}

    def random_batch(self, s_bounds, actions, N):
        S, A, S_ = [], [], []
        start_states = np.array([np.random.uniform(*b, size=N) for b in s_bounds]).T
        for s in start_states:
            self.state = np.array([s[0], s[1] * self.max_speed])
            a = actions[np.random.choice(actions.shape[0]), :]
            s_, _, _, _ = self.step(a)
            S.append(s); A.append(a); S_.append(s_)
        return np.array(S), np.array(A), np.array(S_)

    def get_state(self, s):
        return s
