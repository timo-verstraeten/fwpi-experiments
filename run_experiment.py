import argparse
import json
import logging
import numpy as np
import pickle

from agents.GPDMAgent import GPDMAgent
from environments.MountainCar import MountainCarContinuous
from environments.CartPole import CartPole
from environments.WindFarm import WindFarm
from utils.matrix import generate_full_support_mesh, generate_support_mesh

def execute_policy(policy, env, max_iter, render=False):
    s_ = env.reset()
    t, done = 0, False
    S, A, S_, R = [], [], [], []
    while not done and t < max_iter:
        s = env.get_state(s_)
        a = policy(s)
        s_, r, done, _ = env.step(a)
        S.append(s); A.append(a); S_.append(s_); R.append([r])
        t += 1

        if done:
            logging.info("Episode finished after %i timesteps" % t)
            break

        if render:
            env.render()
    return np.array(S), np.array(A), np.array(S_), np.array(R)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='mountain_car', help='Select an environment')
    parser.add_argument('--agent_type', default='single', help='Select an agent type (single / joint / fleet)')
    parser.add_argument('--seed', default=1, type=int, help='Seed for randomizers')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set parameters
    logging.info("=== CONFIG PARAMETERS ===")
    with open('config/config_%s.json' % args.env_id, 'r') as f:
        config = json.load(f)
    logging.info(config)
    env_params = config["environment"]["params"]
    n_samples = config["environment"]["n_samples"]
    s_bounds, a_bounds = np.array(config["environment"]["s_bounds"]), np.array(config["environment"]["a_bounds"])
    gamma = np.array(config["environment"]["gamma"])
    reward_index = np.array(config["environment"]["reward_index"])
    reward_goal, reward_scale = np.array(config["environment"]["reward_goal"]), np.array(config["environment"]["reward_scale"])
    n_support = config["policy_iteration"]["n_support"]
    max_iter_exp = config["experiment"]["max_iter"]
    max_iter_pi, convergence_threshold = config["policy_iteration"]["max_iter"], config["policy_iteration"]["convergence_threshold"]

    # Pick environment
    if args.env_id == 'mountain_car':
        env = MountainCarContinuous
    elif args.env_id == 'cart_pole':
        env = CartPole
    elif args.env_id == 'wind_farm':
        env = WindFarm
    else:
        raise ValueError('Environment does not exist')
    envs = [env(**param) for param in env_params]

    logging.info("SEED - %i" % args.seed)
    np.random.seed(args.seed)
    for env in envs:
        env.seed(np.random.randint(0,100000))

    # Create GPDM agent
    actions = generate_full_support_mesh(a_bounds)
    reward_stats = (reward_index, reward_goal, reward_scale)
    agent = GPDMAgent(actions, gamma, reward_stats, config['policy_iteration'], envs[0])

    # Create random batches
    support_states = generate_support_mesh(s_bounds[:,:-1], n_support)

    batches = [env.random_batch(s_bounds[:, :-1], actions, n) for m, (n, env) in enumerate(zip(n_samples, envs))]
    if args.agent_type == 'joint':
        batches = [[np.vstack(X) for X in zip(*batches)]]
    elif args.agent_type == 'single':
        batches = [batches[0]]
    elif args.agent_type == 'fleet':
        pass
    else:
        raise ValueError("Agent type unknown")
    batches = list(zip(*batches))

    # Learn policy
    logging.info("Learning policy")
    agent.policy_iteration(batches, support_states, max_iter_pi, convergence_threshold)

    # Demonstrate policy on environment
    S, A, S_, R = execute_policy(lambda s: agent.act(s, epsilon=0.0), envs[0], max_iter_exp, render=True)

    # Write to file
    out = {"members": batches[1:]}
    out[0] = {'S': S, 'A': A, 'S_': S_, 'R': R, 'V_supp': agent.gp_val.support_values, 'S_supp': agent.gp_val.support_points}
    out[0]['cross'] = []
    for d_out, gp in enumerate(agent.gpdm.gps):
        W = gp.coreg.B.W
        M = np.zeros((W.shape[0], W.shape[0]))
        for r in range(W.shape[1]):
            w = W[:,[r]]
            M += np.dot(w, w.T)
        kappa = np.diag(gp.coreg.B.kappa)
        M_full = M + kappa
        M_diag = np.diag(1 / np.sqrt(np.diag(M_full)))
        M_corr = np.dot(M_diag, np.dot(M_full, M_diag))
        out[0]['cross'].append({"W": M, "kappa": kappa, "corr": M_corr})
    with open('out/%s/%s/%i.pkl' % (args.env_id, args.agent_type, args.seed), 'wb') as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    main()
