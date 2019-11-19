from copy import deepcopy
import json
import numpy as np

import lib.floris as floris

class WindFarm:
    def __init__(self, config_file):
        self.max_angle = 45
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.reset()

    def reset(self):
        self.state = deepcopy(self.config)
        yaw1 = 0
        yaw2 = 0
        obs, _, _, _ = self.step(np.array([yaw1,yaw2]))
        return obs

    def _get_obs(self):
        yaws = [turbine["properties"]["yaw_angle"] for turbine in self.state["turbines"]]
        wind = [turbine.power/1e4 for _, turbine in self.floris.farm.turbine_map.items()]
        obs = np.array(yaws + [sum(wind)])
        obs[:2] = obs[:2] / self.max_angle
        obs[:2] = np.sign(obs[:2]) * np.minimum(np.abs(obs[:2]), [1,1])

        return obs

    def step(self, a):
        for dyaw, turbine in zip(a, self.state["turbines"]):
            yaw = turbine["properties"]["yaw_angle"]
            yaw += dyaw
            yaw = np.sign(yaw)*self.max_angle if np.abs(yaw) > self.max_angle else yaw
            turbine["properties"]["yaw_angle"] = yaw
        self.floris = floris.Floris(input_dict=self.state)

        costs = sum([turbine.power for _, turbine in self.floris.farm.turbine_map.items()]) / 1e6
        return self._get_obs(), costs, False, {}

    def random_batch(self, s_bounds, actions, N):
        S, A, S_ = [], [], []
        start_states = np.array([np.random.uniform(*b, size=N) for b in s_bounds]).T
        for s in start_states:
            self.state = self._set_state(s)
            s, _, _, _ = self.step([0,0])
            a = actions[np.random.choice(actions.shape[0]), :]
            s_, _, _, _ = self.step(a)
            s = self.get_state(s)
            S.append(s); A.append(a); S_.append(s_)
        return np.array(S), np.array(A), np.array(S_)

    def _set_state(self, s):
        yaws = self.max_angle*s[:2]
        for i, turbine in enumerate(self.state["turbines"]):
            turbine["properties"]["yaw_angle"] = yaws[i]
        return self.state

    def seed(self, seed):
        pass

    def render(self, mode='human'):
        powers = [turbine.power for _, turbine in self.floris.farm.turbine_map.items()]
        for turbine in self.state['turbines']:
            print("yaw: ", turbine["properties"]["yaw_angle"],)
        print("power: ", sum(powers) / 1e6)

    def get_state(self, s):
        if len(s.shape) == 1:
            s = np.hstack((s[:2], 0))
        else:
            s = np.hstack((s[:,:2], np.zeros((s.shape[0], 1))))
        return s
