# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:31:30 2023

@author: Kevin
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_palette(["#1f77b4", "#FF7F00","#FF0000"])


# specify the values of w and length for each subplot
ws = [15, 150, 250, 400, 5000, 210]
lengths = [170, 1300, 1600, 1390, 3000, 200]

# create the subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
plt.rcParams['font.size'] = '13.5'
name=[r"pruned1.csv",
      r"Traditional1.csv",
      r"qdecomp1.csv",
      r"pruned2.csv",
      r"Traditional2.csv",
      r"qdecomp2.csv",
      r"pruned3.csv",
      r"Traditional3.csv",
      r"qdecomp3.csv",
      r"pruned4.csv",
      r"Traditional4.csv",
      r"qdecomp4.csv",
      #r"pruned5",
      #r"Traditional5",
      r"npruned1.csv",
      r"nTraditional1.csv",
      r"npruned2.csv",
      r"nTraditional2.csv",
      ]

c=0
# iterate over the subplots and plot the data
for i, ax in enumerate(axs.flatten()):
    # read data from CSV files
    pruned = pd.read_csv(name[c], index_col=None).drop("Unnamed: 0", axis=1)
    traditional = pd.read_csv(name[c+1], index_col=None).drop("Unnamed: 0", axis=1)
    if(i<4):
        qdecomp= pd.read_csv(name[c+2], index_col=None).drop("Unnamed: 0", axis=1)
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
        ax.plot(r_list2[:], label="Traditional")
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
    if(i<4):
        c=c+3
    else:
        c=c+2

# set common x and y labels for all subplots
fig.text(0.5, 0.07, 'Episode', ha='center')
fig.text(0.087, 0.5, 'Reward', va='center', rotation='vertical')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3)
