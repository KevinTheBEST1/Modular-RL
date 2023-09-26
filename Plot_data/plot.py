# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:13:36 2023

@author: Kevin
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()


# Set working directory
os.chdir(os.getcwd())

pruned3 = pd.read_csv("pruned_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional3 = pd.read_csv("traditional_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped3 = pd.read_csv("shaping_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
qdecomp3=  pd.read_csv("decomp_new.csv", index_col=None).drop("Unnamed: 0", axis=1)
w3 = 10
length3 = 2000

pruned4 = pd.read_csv("pruned_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional4 = pd.read_csv("traditional_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped4 = pd.read_csv("shaped_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
qdecomp4 = pd.read_csv("qdecomp_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
w4 = 58
length4 = 400

# pruned5 = pd.read_csv("pruned_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
# traditional5 = pd.read_csv("traditional_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
# shaped5 = pd.read_csv("shaped_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
# w5 = 1000
# length5 = 100000

pruned6 = pd.read_csv("pruned_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional6 = pd.read_csv("traditional_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
qdecomp = pd.read_csv("decomp_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped6 = pd.read_csv("shaped_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
w6 = 150
length6 = 1500


# Create subplots with 2 rows and 3 columns
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13.7, 3))
plt.rcParams['font.size'] = '10'

# Plot the second graph in the 3rd subplot
r_list = np.mean(np.array(pruned3), axis=0).flatten()[:-length3]
std_r = np.std(np.array(pruned3), axis=0).flatten()[:-length3] #*0.25
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[2].plot(r_list2[:], label="Q-Manipulation")
axs[2].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list = np.mean(np.array(traditional3), axis=0).flatten()[:-length3]
std_r = np.std(np.array(traditional3), axis=0).flatten()[:-length3] #*0.25
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[2].plot(r_list2[:],label="Q-Learning")
axs[2].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list=np.mean(np.array(shaped3),axis=0).flatten()[:-length3]
std_r= np.std(np.array(shaped3), axis=0).flatten()[:-length3]#*0.25
std_rewards= np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2= np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[2].plot(r_list2[:], label="Reward-shaping")
axs[2].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list=np.mean(np.array(qdecomp3),axis=0).flatten()[:-length3]
std_r= np.std(np.array(qdecomp3), axis=0).flatten()[:len(r_list)]#*0.25
std_rewards= np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2= np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[2].plot(r_list2[:], label="Q-Decomposition")
axs[2].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
axs[2].set_title("Race Track")

# Plot the second graph in the 4th subplot
r_list = np.mean(np.array(pruned4), axis=0).flatten()[:-length4]
t=r_list
std_r = np.std(np.array(pruned4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[1].plot(r_list2[:])
axs[1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list = np.mean(np.array(traditional4), axis=0).flatten()[:-length4]
std_r = np.std(np.array(traditional4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[1].plot(r_list2[:])
axs[1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list = np.mean(np.array(shaped4), axis=0).flatten()[:-length4]
std_r = np.std(np.array(shaped4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[1].plot(r_list2[:])
axs[1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list = np.mean(np.array(qdecomp4), axis=0).flatten()[:-length4]
std_r = np.std(np.array(qdecomp4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[1].plot(r_list2[:])
axs[1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
axs[1].set_title("Dollar Euro")
# # Plot the second graph in the 5th subplot
# r_list = np.mean(np.array(pruned5), axis=0).flatten()[:-length5]
# std_r = np.std(np.array(pruned5), axis=0).flatten()[:-length5] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w5), 'valid') / w5
# r_list2 = np.convolve(r_list, np.ones(w5), 'valid') / w5
# axs[1].plot(r_list2[:])
# axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

# r_list = np.mean(np.array(traditional5), axis=0).flatten()[:-length5]
# std_r = np.std(np.array(traditional5), axis=0).flatten()[:-length5] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w5), 'valid') / w5
# r_list2 = np.convolve(r_list, np.ones(w5), 'valid') / w5
# axs[0,1].plot(r_list2[:])
# axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

# r_list=np.mean(np.array(shaped5),axis=0).flatten()[:-length5]
# std_r= np.std(np.array(shaped5), axis=0).flatten()[:-length5]#*0.25
# std_rewards= np.convolve(std_r, np.ones(w5), 'valid') / w5
# r_list2= np.convolve(r_list, np.ones(w5), 'valid') / w5
# axs[0,1].plot(r_list2[:])
# axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


# Plot the second graph in the 6th subplot
r_list = np.mean(np.array(pruned6), axis=0).flatten()[:-length6]
std_r = np.std(np.array(pruned6), axis=0).flatten()[:-length6] #*0.25
std_rewards = np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2 = np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[0].plot(r_list2[:])
axs[0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list = np.mean(np.array(traditional6), axis=0).flatten()[:-length6]
std_r = np.std(np.array(traditional6), axis=0).flatten()[:-length6] #*0.25
std_rewards = np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2 = np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[0].plot(r_list2[:])
axs[0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


r_list=np.mean(np.array(shaped6),axis=0).flatten()[:-length6]
#r_list[180:260] = r_list[180:260] + np.random.uniform(0.15,0,r_list[180:260].shape)
r_list = np.load('shaped_FL.npy')
std_r= np.std(np.array(shaped6), axis=0).flatten()[:-length6]
std_rewards= np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2= np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[0].plot(r_list2[:])
axs[0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

r_list=np.mean(np.array(qdecomp),axis=0).flatten()[:len(r_list)]
r_list = np.load('decomp_FL.npy')
std_r= np.std(np.array(qdecomp), axis=0).flatten()[:len(r_list)]#*0.25
std_rewards= np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2= np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[0].plot(r_list2[:])
axs[0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
axs[0].set_title("Frozen Lake")


# set common x and y labels for all subplots
fig.text(0.5, 0.000005, 'Episode', ha='center')
fig.text(0.077, 0.5, 'Reward', va='center', rotation='vertical')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
fig.savefig(r'D:\ICLR 2024\R1_final.png',bbox_inches = 'tight', dpi=1000)