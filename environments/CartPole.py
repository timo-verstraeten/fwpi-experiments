from gym.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from gym import spaces, logger
import math

class CartPole(CartPoleEnv):
    def __init__(self, masspole):
        super().__init__()
        #self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,))
        self.masspole = masspole

    def reset(self):
        s = super().reset()
        return s

    def step(self, a):
        a = self._transform_action(a[0])

        assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if a==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state).flatten(), reward, done, {}

    def random_batch(self, s_bounds, actions, N):
        S, A, S_ = [], [], []
        start_states = np.array([np.random.uniform(*b, size=N) for b in s_bounds]).T
        for s in start_states:
            self.reset()
            self.state = s
            a = actions[np.random.choice(actions.shape[0]), :]
            s_, _, _, _ = self.step(a)
            S.append(s); A.append(a); S_.append(s_)
        return np.array(S), np.array(A), np.array(S_)

    def _transform_action(self, a):
        return 1 if a > 0 else 0

    def get_state(self, s):
        return s