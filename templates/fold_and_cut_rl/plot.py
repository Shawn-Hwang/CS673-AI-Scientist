import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import bootstrap
import seaborn as sns


env = "LunarLander-v3"

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Folds Baseline",
    "run_1": "No Folds",
}

# create the figure and axes for 2 x 2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

folders = os.listdir("./")
for folder in folders:
    if folder.startswith("run") and os.path.isdir(folder): 
        rewards = []
        grad_norms = []
        grad_vars = []
        param_norms = []
        for run in os.listdir(folder):
            ea = event_accumulator.EventAccumulator(os.path.join(folder, run))
            ea.Reload()

            # get rewards
            # print(folder+'/'+run)
            # episodic_return = ea.Scalars('charts/episodic_return')
            # reward_steps = [e.step for e in episodic_return]
            # rewards.append([e.value for e in episodic_return])

            # get grad norms
            grad_norms_single_run = ea.Scalars('gradients/norm_together')
            grad_norms.append([e.value for e in grad_norms_single_run])
            grad_and_param_steps = [e.step for e in grad_norms_single_run]

            # get grad variances
            grad_vars_single_run = ea.Scalars('gradients/variance_together')
            grad_vars.append([e.value for e in grad_vars_single_run])

            # get param norms
            param_norms_single_run = ea.Scalars('parameters/norm')
            param_norms.append([e.value for e in param_norms_single_run])

        # res = bootstrap((np.array(rewards),), np.mean)
        # axs[0,0].plot(reward_steps, np.mean(np.array(rewards), axis=0), label=labels[folder])
        # axs[0,0].fill_between(reward_steps, res.confidence_interval.low, res.confidence_interval.high, alpha=0.25)

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
# axs[0,0].set_xlabel('Steps')
# axs[0,0].set_ylabel('Reward')
# axs[0,0].set_title('Reward')
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
plt.savefig(f"{env}_metrics.png")
plt.close()
