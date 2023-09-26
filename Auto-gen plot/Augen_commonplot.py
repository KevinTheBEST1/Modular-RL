import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_palette(["#1f77b4", "#FF7F00","#FF0000"])


# specify the values of w and length for each subplot
ws = [20, 150, 250, 50,]
lengths = [1800, 1500, 1450, 1800,]

# create the subplots
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
plt.rcParams['font.size'] = '10'
plt.rcParams['font.size'] = '13.5'
name=[r"pruned1_new.csv",
      r"Traditional1_new.csv",
      r"traditional_shaped1_abstract.csv",
      r"qdecomp1.csv",
      r"pruned2_new.csv",
      r"Traditional2_new.csv",
      r"traditional_shaped2_abstract.csv",
      r"qdecomp2.csv",
      r"pruned3_new.csv",
      r"Traditional3_new.csv",
      r"traditional_shaped3_abstract.csv",
      r"qdecomp3_new.csv",
      r"pruned1_4b.csv",
      r"Traditional1_4b.csv",
      r"traditional_shaped1_abstract_4b.csv",
      r"qdecomp1_4b.csv",
      ]

c=0
# iterate over the subplots and plot the data
for i, ax in enumerate(axs.flatten()):
    ax.set_title("Auto-gen " +str(i+1))
    print(c)
    # read data from CSV files
    pruned = pd.read_csv(name[c], index_col=None).drop("Unnamed: 0", axis=1)
    traditional = pd.read_csv(name[c+1], index_col=None).drop("Unnamed: 0", axis=1)
    shaped=pd.read_csv(name[c+2], index_col=None).drop("Unnamed: 0", axis=1)
    if(i<4):
        shaped = pd.read_csv(name[c+2], index_col=None).drop("Unnamed: 0", axis=1)
        qdecomp = pd.read_csv(name[c+3], index_col=None).drop("Unnamed: 0", axis=1)
    # compute the rolling average and standard deviation of the rewards
    r_list = np.mean(np.array(pruned), axis=0).flatten()[:-lengths[i]]
    std_r = np.std(np.array(pruned), axis=0).flatten()[:-lengths[i]]
    std_rewards = np.convolve(std_r, np.ones(ws[i]), 'valid') / ws[i]
    r_list2 = np.convolve(r_list, np.ones(ws[i]), 'valid') / ws[i]
    if(c==0):
        ax.plot(r_list2[:],label="Q-Manipulation")
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)

    else:
        ax.plot(r_list2[:])
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)

    r_list = np.mean(np.array(traditional), axis=0).flatten()[:-lengths[i]]
    std_r = np.std(np.array(traditional), axis=0).flatten()[:-lengths[i]]
    std_rewards = np.convolve(std_r, np.ones(ws[i]), 'valid') / ws[i]
    r_list2 = np.convolve(r_list, np.ones(ws[i]), 'valid') / ws[i]
    if(c==0):
        ax.plot(r_list2[:], label="Q-Learning")
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)

    else:
        ax.plot(r_list2[:])
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)

    if(i<4):
        r_list = np.mean(np.array(qdecomp), axis=0).flatten()[:len(r_list)]
        std_r = np.std(np.array(qdecomp), axis=0).flatten()[:len(r_list)]
        std_rewards = np.convolve(std_r, np.ones(ws[i]), 'valid') / ws[i]
        r_list2 = np.convolve(r_list, np.ones(ws[i]), 'valid') / ws[i]
        if(c==0):
            ax.plot(r_list2[:], label="Q-Decomposition")
            ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)
    
        else:
            ax.plot(r_list2[:])
            ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15)
        
    r_list = np.mean(np.array(shaped), axis=0).flatten()[:len(r_list)]
    std_r = np.std(np.array(shaped), axis=0).flatten()[:len(r_list)]
    std_rewards = np.convolve(std_r, np.ones(ws[i]), 'valid') / ws[i]
    r_list2 = np.convolve(r_list, np.ones(ws[i]), 'valid') / ws[i]
    if(c==0):
        ax.plot(r_list2[:], label="Reward Shaping", color="green")
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15, color="green")

    else:
        ax.plot(r_list2[:], color="green")
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.15, color="green")
    if(i<4):
        c=c+4
    else:
        c=c+3

# set common x and y labels for all subplots
fig.text(0.5, 0.009, 'Episode', ha='center')
fig.text(0.08, 0.5, 'Reward', va='center', rotation='vertical')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
plt.show()