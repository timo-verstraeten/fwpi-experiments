import logging
import math
import numpy as np
import scipy as sp

from gp.gprl import GPDM, ValueGP

class GPDMAgent:

    def __init__(self, actions, gamma, reward_stats, config, env):

        self.env = env

        # State and actions
        self.support_states = None
        self.actions = actions
        self.Ns, self.Ds = None, None
        self.Na, self.Da = actions.shape

        # Rewards
        self.gamma = gamma
        self.reward_index = np.array(reward_stats[0])
        self.reward_goal = np.array(reward_stats[1])
        self.reward_scale = np.array(reward_stats[2])
        self.reward_func = lambda s: sp.stats.multivariate_normal(mean=self.reward_goal,
                                                                  cov=self.reward_scale**2).pdf(s[self.reward_index])

        # GPs
        self.gpdm = GPDM(config['gpdm']['noise_var'], config['gpdm']['sparse'])
        self.gp_val = ValueGP(config['gp_val']['noise_var'])

    def act(self, state, epsilon=1.0):
        if np.random.uniform() < epsilon:
            a = self.actions[np.random.choice(self.actions.shape[0]), :]
        else:
            a = self.policy_improvement(np.atleast_2d(state))[0][0]
        return a

    def policy_iteration(self, batch, support_states, max_iter, convergence_threshold):

        logging.info("Fit GPDM")
        self.gpdm.fit(*batch)
        self.gpdm.print_parameters()

        logging.info("Initialise support points")
        self.initialise_support_points(support_states)

        logging.info("Policy iteration")
        i = 0
        converged = False
        while not converged and i < max_iter:

            # Get statistics on best actions
            _, _, R, W = self.policy_improvement(self.support_states)

            if W.shape != (self.Ns, self.Ns):
                raise AssertionError("Weight matrix W should have equally sized dimensions.")

            # Policy evaluation
            mse_val = self.policy_evaluation(R, W)

            # Check for convergence
            if mse_val < convergence_threshold:
                converged = True
            logging.info("MSE on support values: %s" % mse_val)
            self.gp_val.print_parameters()

            i += 1

    def policy_evaluation(self, R, W):
        # Evaluate new policy
        # Solve [(I - gamma W K_inv) V = R] for V
        X = np.identity(self.Ns) - self.gamma * np.dot(W, self.gp_val.K_inv)
        new_values = sp.linalg.solve(X, R)

        # Compare old values with new ones
        mse_val = np.mean((self.gp_val.support_values - new_values)**2)

        # Update values
        self.gp_val.fit(self.support_states, new_values)

        return mse_val

    def policy_improvement(self, states):
        R = np.zeros((self.Ns, 1))
        W = np.zeros((self.Ns, self.Ns))
        maximising_actions = np.zeros((self.Ns, self.Da))

        for state_index in range(len(states)):
            state = np.atleast_2d(states[state_index])
            max_val_index, r, w = self.find_max_action(state)

            maximising_actions[state_index] = self.actions[max_val_index, :]
            R[state_index][0] = r
            W[state_index] = w

        return maximising_actions, [], R, W

    def find_max_action(self, state):
        # Kernel parameters
        s2_val, ls_val = self.gp_val.var, self.gp_val.ls

        # Predict next states for state-action pairs
        S, A = np.repeat(state, repeats=self.Na, axis=0), self.actions
        S = self.env.get_state(S)
        MU_star, S2_star = self.gpdm.predict(S, A)

        # Weight matrix W
        d_states = (MU_star[:, :, None] - self.support_states.T) ** 2  # (Ns, Ds, 1) - (Ds, Ns x Na) = (Ns, Ds, Ns x Na)
        d_states = d_states.T
        Sigma = S2_star + ls_val ** 2
        W = np.exp(-0.5 * np.sum(d_states / Sigma.T, axis=1))  # (Ns, Ds, Ns x Na) / (Ds, Ns x Na) = (Ns, Ds, Ns x Na)
        W = np.prod(ls_val) * s2_val * W / np.sqrt(np.prod(Sigma, axis=1))  # (Ns, Ns x Na)
        W = W.T

        # Reward R given at the next states
        d_goal = (MU_star[:,self.reward_index] - self.reward_goal) ** 2
        scale = S2_star[:,self.reward_index] + self.reward_scale ** 2
        R = np.exp(-0.5 * np.sum(d_goal / scale, axis=1))
        R /= np.sqrt(np.prod(2 * math.pi * scale, axis=1))

        # Predict values and best actions
        V_star = R + self.gamma * np.dot(W, self.gp_val.V_kern).flatten()  # (Ns x Na,) + (Ns,) . (Ns, Ns x Na) = (Ns x Na,)

        action_index = np.argmax(V_star.reshape((state.shape[0], self.Na)), axis=1)
        action_index = action_index[0]

        return action_index, R[action_index], W[action_index,:]

    def initialise_support_points(self, support_states):
        self.support_states = support_states
        self.Ns, self.Ds = support_states.shape

        values = np.array([[self.reward_func(state)] for state in self.support_states])
        self.gp_val.fit(self.support_states, values)