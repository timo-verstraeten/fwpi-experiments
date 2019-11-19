import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import scipy as sp
import scipy.stats

from gp.gprl import ValueGP

def distance_MC():
    seed_file = 'seeds.txt'
    agent_types = ['joint', 'single', 'fleet']
    setting = 'out/mountain_car'
    seeds = list(map(str, pd.read_csv(seed_file).values.flatten()))

    dists = pd.DataFrame(np.nan, index=seeds, columns=agent_types)
    for agent_type in agent_types:
        folder = os.path.join(setting, agent_type)
        for seed in seeds:
            file = seed + '.pkl'
            path = os.path.join(folder, file)
            with open(path, 'rb') as f:
                dat = pickle.load(f)
            dist = (dat[0]['S_'] - np.array([0.45, 0]))

            dist = np.linalg.norm(dist, axis=1)
            dist = np.hstack((dist, np.zeros(200 - len(dist))))**2
            dists[agent_type].loc[seed] = np.sum(dist)

    dists = dists.unstack().reset_index().rename(columns={'level_0': 'agent_type', 'level_1': 'seed', 0: 'dist'})
    dists['agent_type'] = dists['agent_type'].str.capitalize()

    sns.boxplot(x='agent_type', y='dist', data=dists, showfliers=False)
    plt.xlabel('')
    plt.ylabel('Sum of squared distances to goal')
    plt.savefig('mc_si.pdf')
    plt.clf()

def distance_CP():
    seed_file = 'seeds.txt'
    agent_types = ['joint', 'single', 'fleet']
    setting = 'out/cart_pole'
    seeds = list(map(str, pd.read_csv(seed_file).values.flatten()))

    dists = pd.DataFrame(np.nan, index=seeds, columns=agent_types)
    for agent_type in agent_types:
        folder = os.path.join(setting, agent_type)
        for seed in seeds:
            file = seed + '.pkl'
            path = os.path.join(folder, file)
            with open(path, 'rb') as f:
                dat = pickle.load(f)

            reward_goal = np.array([0,0])
            max_dist = np.linalg.norm(np.array([2.4, 12 * 2 * np.pi / 360]) - reward_goal)

            dist = np.linalg.norm(dat[0]['S_'][:,[0,2]] - reward_goal, axis=1)
            dist = np.hstack((dist, (200 - len(dist))*[max_dist]))**2
            dists[agent_type].loc[seed] = np.sum(dist)
    dists = dists.unstack().reset_index().rename(columns={'level_0': 'agent_type', 'level_1': 'seed', 0: 'dist'})
    dists['agent_type'] = dists['agent_type'].str.capitalize()

    sns.boxplot(x='agent_type', y='dist', data=dists, showfliers=False)
    plt.xlabel('')
    plt.ylabel('Sum of squared distances to goal')
    plt.savefig('cp_si.pdf')
    plt.clf()

def performance_WF():
    seed_file = 'seeds.txt'
    setting = 'out/wind_farm'
    with open(seed_file, 'r') as f:
        seeds = f.read().split('\n')

    dfs_power = []
    corrs = collections.defaultdict(list)
    for agent_type in ['joint', 'single', 'fleet']:
        i = 0
        folder = os.path.join(setting, agent_type)
        for trial, seed in enumerate(seeds):
            file = seed + '.pkl'
            path = os.path.join(folder, file)
            with open(path, 'rb') as f:
                dat = pickle.load(f)
            power = dat[0]['R'][-1,0]

            df_power = pd.DataFrame(np.nan, index=[0],
                                   columns=['trial', 'agent_type', 'power'])
            df_power['trial'] = trial
            df_power['agent_type'] = agent_type
            df_power['power'] = power
            dfs_power.append(df_power)

            i += 1
            if agent_type == 'fleet':
                for d, cross in enumerate(dat[0]['cross']):
                    corrs[d].append(cross['corr'])
    df_power = pd.concat(dfs_power, axis=0).reset_index()
    df_power['agent_type'] = df_power['agent_type'].str.capitalize()

    # Check floris simulator
    baseline = 1027371.17753
    optimum = 1047887.97818
    baseline /= 1e6
    optimum /= 1e6

    plt.plot(np.arange(-1,10), baseline*np.ones(11), 'k--')
    plt.plot(np.arange(-1,10), optimum*np.ones(11), 'k--')
    sns.boxplot(x='agent_type', y='power', data=df_power, showfliers=False)
    yticks = [1.06, 1.00, baseline, optimum]
    ylabels = ["1.06 MW", "1.00 MW", "current\npractice", "optimum"]
    plt.ylabel('Power production')
    plt.xlabel('')
    plt.yticks(yticks, ylabels)
    plt.tight_layout()
    plt.savefig('wind_si.pdf')
    plt.clf()

def value_function_MC():
    setting = 'out/mountain_car'

    seed_map = {'fleet': 8708, 'joint': 3467, 'single': 46}  # Best runs

    with open('config/config_mountain_car.json', 'r') as f:
        config = json.load(f)
        noise_var = config['policy_iteration']['gp_val']['noise_var']

    for agent_type in ['fleet', 'joint', 'single']:
        path = os.path.join(setting, agent_type, str(seed_map[agent_type]) + '.pkl')
        with open(path, 'rb') as f:
            dat = pickle.load(f)
        S, V = dat[0]['S_supp'], dat[0]['V_supp']
        gp = ValueGP(noise_var)
        gp.fit(S, V)

        # Plot MEAN
        x1 = np.linspace(-1.1, 0.55, 100)
        x2 = np.linspace(-1, 1, 100)
        x, x_dot = np.meshgrid(x1, x2)
        X = np.vstack((x.flatten(), x_dot.flatten())).T
        MU = gp.predict(X)[0].flatten()
        MU = MU.reshape((100, 100))

        plt.clf()
        contour = plt.contourf(x, x_dot, MU, cmap='RdGy_r')
        clb = plt.colorbar(contour, shrink=0.5)
        clb.ax.set_title('Value')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.savefig(f"values_{agent_type}.pdf", dpi=300)

        # Plot VARIANCE
        x1 = np.linspace(-1.1, 0.55, 100)
        x2 = np.linspace(-1, 1, 100)
        x, x_dot = np.meshgrid(x1, x2)
        X = np.vstack((x.flatten(), x_dot.flatten())).T
        STD = gp.predict(X)[1].flatten()
        print('Average STD on value surface: ', agent_type, np.average(STD))

def corr_tables():
    seed_file = 'seeds.txt'
    seeds = pd.read_csv(seed_file)['seed'].values

    corrs = collections.defaultdict(list)
    for trial, seed in enumerate(seeds):
        path = f'sensitivity_analysis/mountain_car/fleet/sens_0.001000_0.000100_{seed}.pkl'
        with open(path, 'rb') as f:
            dat = pickle.load(f)

        for d, cross in enumerate(dat[0]['cross']):
            corrs[d].append(cross['corr'])

    mc_pos = sum(corrs[0])
    mc_vel = sum(corrs[1])
    mc_pos = mc_pos / mc_pos[0,0]
    mc_vel = mc_vel / mc_vel[0,0]
    sns.set(font_scale=1.5)
    sns.heatmap(mc_pos, vmax=1, vmin=-1, cmap='RdYlGn', annot=True, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 20, "weight": "bold"})
    plt.xticks(np.array([0, 1, 2]) + 0.5, ['T', 'SA', 'SB'])
    plt.yticks(np.array([0, 1, 2]) + 0.5, ['T', 'SA', 'SB'])
    plt.savefig('mc_corrs_pos.pdf')
    plt.clf()
    sns.heatmap(mc_vel, vmax=1, vmin=-1, cmap='RdYlGn', annot=True, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 20, "weight": "bold"})
    plt.xticks(np.array([0, 1, 2]) + 0.5, ['T', 'SA', 'SB'])
    plt.yticks(np.array([0, 1, 2]) + 0.5, ['T', 'SA', 'SB'])
    plt.savefig('mc_corrs_vel.pdf')
    plt.clf()

def sensitivity_analysis():
    seed_file = 'seeds.txt'
    main_folder = 'sensitivity_analysis/mountain_car'
    dfs = []
    for agent_type in ['fleet', 'joint']:
        folder = os.path.join(main_folder, agent_type)
        for file in os.listdir(folder):
            _, p1, p2, seed = file[:-4].split('_')

            seed = int(seed)
            all_seeds = set(pd.read_csv(seed_file)['seed'].values)
            if seed not in all_seeds:
                continue
            p1, p2 = float(p1), float(p2)
            with open(os.path.join(folder, file), 'rb') as f:
                dat = pickle.load(f)

            # Distance
            dist = (dat[0]['S_'] - np.array([0.45, 0]))
            dist = np.linalg.norm(dist, axis=1)
            dist = np.hstack((dist, np.zeros(200 - len(dist)))) ** 2
            dist = np.sum(dist)

            # Correlation
            df = pd.DataFrame([[p1, dist]], index=[0],
                              columns=['p1', 'dist'])
            df['seed'] = seed
            df['agent_type'] = agent_type
            dfs.append(df)

    # READ SINGLE TARGET
    folder = 'out/mountain_car/single'
    for file in os.listdir(folder):
        seed = int(file.split('.')[0])
        all_seeds = set(pd.read_csv(seed_file)['seed'].values)
        if seed not in all_seeds:
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            dat = pickle.load(f)

        # Distance
        dist = (dat[0]['S_'] - np.array([0.45, 0]))
        dist = np.linalg.norm(dist, axis=1)
        dist = np.hstack((dist, np.zeros(200 - len(dist)))) ** 2
        dist = np.sum(dist)

        for p1 in (np.arange(15)+1)/1e4:
            df = pd.DataFrame([[p1, dist]], index=[0], columns=['p1', 'dist'])
            df['seed'] = seed
            df['agent_type'] = 'single'
            dfs.append(df)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = df.sort_values(by=['agent_type', 'p1', 'seed'])

    df['agent_type'] = df['agent_type'].str.capitalize()
    df.rename(columns={'agent_type': 'Agent type'}, inplace=True)
    df_sens = df.loc[df['p1'] > 0.0004]

    plt.figure(figsize=(15, 4))
    sns.boxplot(x='p1', y='dist', hue='Agent type', data=df_sens, showfliers=False)
    plt.xticks(np.arange(11), np.arange(5,16))
    plt.xlabel('Power of car (x $10^{-4}$)')
    plt.ylabel('Sum of squared distances to goal')
    plt.savefig('sensitivity_analysis.pdf')
    plt.clf()


distance_MC()
distance_CP()
performance_WF()

value_function_MC()
corr_tables()
sensitivity_analysis()