import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import bootstrap
import seaborn as sns
import pandas as pd

env = "LunarLander-v3"

labels = {
    "fold_specific_learning_rates": "Folds 10x lr",
    "run_0": "Folds",
    "no_fold_baseline": "No Folds"
}

# create the figure and axes for 2 x 2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

folders = os.listdir("./")
for folder in folders:
    if (folder in labels.keys()) and os.path.isdir(folder): 
        rewards = []
        grad_norms = []
        grad_vars = []
        param_norms = []
        for run in os.listdir(folder):
            if run == "final_info.json" :
                continue
            ea = event_accumulator.EventAccumulator(os.path.join(folder, run))
            ea.Reload()

            # get rewards
            print(folder+'/'+run)
            episodic_return = ea.Scalars('charts/episodic_return')
            reward_steps = [e.step for e in episodic_return]
            rewards_across_episodes = [e.value for e in episodic_return]
            df = pd.DataFrame({'step': reward_steps, 'reward': rewards_across_episodes})
            grouped_df = df.groupby('step')['reward'].mean().reset_index()
            reward_steps = grouped_df['step'].to_numpy()
            rewards_across_episodes = grouped_df['reward'].to_numpy()

            indices = np.searchsorted(reward_steps, np.arange(200, 475001, 50), side='right') - 1
            indices[indices < 0] = 0
            rewards.append(rewards_across_episodes[indices])

            # get grad norms
            grad_norms_single_run = ea.Scalars('gradients/norm_together')
            grad_norm = [e.value for e in grad_norms_single_run]
            grad_and_param_steps = [e.step for e in grad_norms_single_run]

            # get grad variances
            grad_vars_single_run = ea.Scalars('gradients/variance_together')
            grad_var = [e.value for e in grad_vars_single_run]

            # get param norms
            param_norms_single_run = ea.Scalars('parameters/norm')
            param_norm = [e.value for e in param_norms_single_run]

            df = pd.DataFrame({'step': grad_and_param_steps, 'grad_norm': grad_norm, 
                               'grad_var': grad_var, 'param_norm': param_norm})
            grouped_df = df.groupby('step')[['grad_norm', 'grad_var', 'param_norm']].mean().reset_index()
            grad_and_param_steps = grouped_df['step'].to_numpy()
            grad_norms.append(grouped_df['grad_norm'].to_numpy())
            grad_vars.append(grouped_df['grad_var'].to_numpy())
            param_norms.append(grouped_df['param_norm'].to_numpy())

        # rewards = [run[:min([len(a) for a in rewards])] for run in rewards]
        # reward_steps = reward_steps[:min([len(a) for a in rewards])]
        # grad_norms = [run[:min([len(a) for a in grad_norms])] for run in grad_norms]
        # grad_and_param_steps = grad_and_param_steps[:min([len(a) for a in grad_norms])]
        # grad_vars = [run[:min([len(a) for a in grad_vars])] for run in grad_vars]
        # param_norms = [run[:min([len(a) for a in param_norms])] for run in param_norms]
        
        res = bootstrap((np.array(rewards),), np.mean)
        axs[0,0].plot(np.arange(200, 475001, 50), np.mean(np.array(rewards), axis=0), label=labels[folder])
        axs[0,0].fill_between(np.arange(200, 475001, 50), res.confidence_interval.low, res.confidence_interval.high, alpha=0.25)

        res = bootstrap((np.array(grad_norms),), np.mean)
        axs[0,1].plot(grad_and_param_steps, np.mean(np.array(grad_norms), axis=0), label=labels[folder])
        axs[0,1].fill_between(grad_and_param_steps, res.confidence_interval.low, res.confidence_interval.high, alpha=0.25)

        res = bootstrap((np.array(grad_vars),), np.mean)
        axs[1,0].plot(grad_and_param_steps, np.mean(np.array(grad_vars), axis=0), label=labels[folder])
        axs[1,0].fill_between(grad_and_param_steps, res.confidence_interval.low, res.confidence_interval.high, alpha=0.25)

        res = bootstrap((np.array(param_norms),), np.mean)
        axs[1,1].plot(grad_and_param_steps, np.mean(np.array(param_norms), axis=0), label=labels[folder])
        axs[1,1].fill_between(grad_and_param_steps, res.confidence_interval.low, res.confidence_interval.high, alpha=0.25)

# Set the labels and title
axs[0,0].set_xlabel('Steps')
axs[0,0].set_ylabel('Reward')
axs[0,0].set_title('Reward')
axs[0,1].set_xlabel('Steps')
axs[0,1].set_ylabel('Gradient Norm')
axs[0,1].set_title('Gradient Norm')
axs[1,0].set_xlabel('Steps')
axs[1,0].set_ylabel('Gradient Variance')
axs[1,0].set_title('Gradient Variance')
axs[1,1].set_xlabel('Steps')
axs[1,1].set_ylabel('Parameter Norm')
axs[1,1].set_title('Parameter Norm')

# create the legend for the figure and save the plot
# axs[0,0].legend()
axs[0,1].legend()
plt.tight_layout()
plt.savefig(f"{env}.png")
plt.close()
