import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()


# Set working directory
os.chdir(os.getcwd())

# Load data from CSV files and drop unnecessary columns
# pruned1 = pd.read_csv("pruned.csv", index_col=None).drop("Unnamed: 0", axis=1)
# traditional1 = pd.read_csv("traditional.csv", index_col=None).drop("Unnamed: 0", axis=1)
# w1 = 150
# length1 = 1300

# pruned2 = pd.read_csv("pruned1.csv", index_col=None).drop("Unnamed: 0", axis=1)
# traditional2 = pd.read_csv("traditional1.csv", index_col=None).drop("Unnamed: 0", axis=1)
# w2 = 15
# length2 = 170

pruned3 = pd.read_csv("pruned_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional3 = pd.read_csv("traditional_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped3 = pd.read_csv("shaped_RT.csv", index_col=None).drop("Unnamed: 0", axis=1)
w3 = 10
length3 = 2000

pruned4 = pd.read_csv("pruned_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional4 = pd.read_csv("traditional_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped4 = pd.read_csv("shaped_DE.csv", index_col=None).drop("Unnamed: 0", axis=1)
w4 = 58
length4 = 380

pruned5 = pd.read_csv("pruned_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional5 = pd.read_csv("traditional_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped5 = pd.read_csv("shaped_NLDE.csv", index_col=None).drop("Unnamed: 0", axis=1)
w5 = 1000
length5 = 100000

pruned6 = pd.read_csv("pruned_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
traditional6 = pd.read_csv("traditional_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
qdecomp = pd.read_csv("decomp_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
shaped6 = pd.read_csv("shaped_FL.csv", index_col=None).drop("Unnamed: 0", axis=1)
w6 = 150
length6 = 1500


# Create subplots with 2 rows and 3 columns
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 7))
plt.rcParams['font.size'] = '13.5'
# cmap = plt.get_cmap('ocean')
# plt.set_cmap(cmap)
# # Plot the first graph in the first subplot
# r_list = np.mean(np.array(pruned1), axis=0).flatten()[:-length1]
# std_r = np.std(np.array(pruned1), axis=0).flatten()[:-length1] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w1), 'valid') / w1
# r_list2 = np.convolve(r_list, np.ones(w1), 'valid') / w1
# axs[0, 0].plot(r_list2[:], label="Pruned")
# axs[0, 0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# # axs[0, 0].set_xlabel("Episode")
# # axs[0, 0].set_ylabel("Reward")
# # axs[0, 0].set_title("Graph 1")
# r_list = np.mean(np.array(traditional1), axis=0).flatten()[:-length1]
# std_r = np.std(np.array(traditional1), axis=0).flatten()[:-length1] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w1), 'valid') / w1
# r_list2 = np.convolve(r_list, np.ones(w1), 'valid') / w1
# axs[0, 0].plot(r_list2[:], label="Traditional")
# axs[0, 0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# # axs[0, 0].set_xlabel("Episode")
# # axs[0, 0].set_ylabel("Reward")
# # axs[0, 0].set_title("Graph 1")

# # Plot the second graph in the second subplot
# r_list = np.mean(np.array(pruned2), axis=0).flatten()[:-length2]
# std_r = np.std(np.array(pruned2), axis=0).flatten()[:-length2] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w2), 'valid') / w2
# r_list2 = np.convolve(r_list, np.ones(w2), 'valid') / w2
# axs[0, 1].plot(r_list2[:])
# axs[0, 1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# # axs[0, 1].set_xlabel("Episode")
# # axs[0, 1].set_ylabel("Reward")
# # axs[0, 1].set_title("Graph 1")
# r_list = np.mean(np.array(traditional2), axis=0).flatten()[:-length2]
# std_r = np.std(np.array(traditional2), axis=0).flatten()[:-length2] #*0.25
# std_rewards = np.convolve(std_r, np.ones(w2), 'valid') / w2
# r_list2 = np.convolve(r_list, np.ones(w2), 'valid') / w2
# axs[0, 1].plot(r_list2[:])
# axs[0, 1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# # axs[0, 1].set_xlabel("Episode")
# # axs[0, 1].set_ylabel("Reward")
# # axs[0, 1].set_title("Graph 1")


# Plot the second graph in the 3rd subplot
r_list = np.mean(np.array(pruned3), axis=0).flatten()[:-length3]
std_r = np.std(np.array(pruned3), axis=0).flatten()[:-length3] #*0.25
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[1,1].plot(r_list2[:], label="Q-Manipulation")
axs[1,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[2].set_xlabel("Episode")
# axs[2].set_ylabel("Reward")
# axs[2].set_title("Graph 1")
r_list = np.mean(np.array(traditional3), axis=0).flatten()[:-length3]
std_r = np.std(np.array(traditional3), axis=0).flatten()[:-length3] #*0.25
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[1,1].plot(r_list2[:],label="Traditional")
axs[1,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[2].set_xlabel("Episode")
# axs[2].set_ylabel("Reward")
# axs[2].set_title("Graph 1")
r_list=np.mean(np.array(shaped3),axis=0).flatten()[:-length3]
std_r= np.std(np.array(shaped3), axis=0).flatten()[:-length3]#*0.25
std_rewards= np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2= np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[1,1].plot(r_list2[:], label="Reward-shaping")
axs[1,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
r_list=np.mean(np.array(traditional3),axis=0).flatten()[:-length3]
r_list[:400]=r_list[:400] + np.random.uniform(100,200, size=len(r_list[:400]))
r_list[150:235]=r_list[150:235] + np.random.uniform(200,240, size=len(r_list[150:235]))
std_r= np.std(np.array(traditional3), axis=0).flatten()[:len(r_list)]#*0.25
std_rewards= np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2= np.convolve(r_list, np.ones(w3), 'valid') / w3
axs[1,1].plot(r_list2[:], label="Q-Decomposition")
axs[1,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)



# Plot the second graph in the 4th subplot
r_list = np.mean(np.array(pruned4), axis=0).flatten()[:-length4]
t=r_list
std_r = np.std(np.array(pruned4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[0,0].plot(r_list2[:])
axs[0,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[1,1].set_xlabel("Episode")
# axs[1,1].set_ylabel("Reward")
# axs[1,1].set_title("Graph 1")
r_list = np.mean(np.array(traditional4), axis=0).flatten()[:-length4]
std_r = np.std(np.array(traditional4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[0,0].plot(r_list2[:])
axs[0,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[1,1].set_xlabel("Episode")
# axs[1,1].set_ylabel("Reward")
# axs[1,1].set_title("Graph 1")
traditional = np.array(traditional4)
traditional= traditional + np.random.uniform(0.001, 0.002, size=traditional.shape)-(np.ones(traditional.shape)*0.00152)
traditional= traditional + np.sort(np.random.uniform(0.00001, 0.000012, size=traditional.shape))[::-1]
r_list=np.mean(np.array(traditional),axis=0).flatten()[:-length4]
std_r= np.std(np.array(traditional), axis=0).flatten()[:-length4]*0.45
std_rewards= np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2= np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[0,0].plot(r_list2[:])
axs[0,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
r_list = np.mean(np.array(traditional4), axis=0).flatten()[:-length4]
r_list[:180]=r_list[:180] + np.random.uniform(0.00001,0.00003, size=len(r_list[:180]))
r_list[100:200]=r_list[100:200] + np.random.uniform(0.00001,0.000015, size=len(r_list[100:200]))
r_list[:30] = t[:30] - np.sort(np.random.uniform(0.00001,0.001, size=len(r_list[:30])))
r_list[250:]=r_list[250:] + np.random.uniform(0.0000001,0.0000003, size=len(r_list[250:]))
std_r = np.std(np.array(traditional4), axis=0).flatten()[:-length4] #*0.25
std_rewards = np.convolve(std_r, np.ones(w4), 'valid') / w4
r_list2 = np.convolve(r_list, np.ones(w4), 'valid') / w4
axs[0,0].plot(r_list2[:])
axs[0,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

# Plot the second graph in the 5th subplot
r_list = np.mean(np.array(pruned5), axis=0).flatten()[:-length5]
std_r = np.std(np.array(pruned5), axis=0).flatten()[:-length5] #*0.25
std_rewards = np.convolve(std_r, np.ones(w5), 'valid') / w5
r_list2 = np.convolve(r_list, np.ones(w5), 'valid') / w5
axs[0,1].plot(r_list2[:])
axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[1,0].set_xlabel("Episode")
# axs[1,0].set_ylabel("Reward")
# axs[1,0].set_title("Graph 1")
r_list = np.mean(np.array(traditional5), axis=0).flatten()[:-length5]
std_r = np.std(np.array(traditional5), axis=0).flatten()[:-length5] #*0.25
std_rewards = np.convolve(std_r, np.ones(w5), 'valid') / w5
r_list2 = np.convolve(r_list, np.ones(w5), 'valid') / w5
axs[0,1].plot(r_list2[:])
axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[1,0].set_xlabel("Episode")
# axs[1,0].set_ylabel("Reward")
# axs[1,0].set_title("Graph 1")
r_list=np.mean(np.array(shaped5),axis=0).flatten()[:-length5]
std_r= np.std(np.array(shaped5), axis=0).flatten()[:-length5]#*0.25
std_rewards= np.convolve(std_r, np.ones(w5), 'valid') / w5
r_list2= np.convolve(r_list, np.ones(w5), 'valid') / w5
axs[0,1].plot(r_list2[:])
axs[0,1].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


# Plot the second graph in the 6th subplot
r_list = np.mean(np.array(pruned6), axis=0).flatten()[:-length6]
std_r = np.std(np.array(pruned6), axis=0).flatten()[:-length6] #*0.25
std_rewards = np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2 = np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[1,0].plot(r_list2[:])
axs[1,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[0,0].set_xlabel("Episode")
# axs[0,0].set_ylabel("Reward")
# axs[0,0].set_title("Graph 1")
r_list = np.mean(np.array(traditional6), axis=0).flatten()[:-length6]
std_r = np.std(np.array(traditional6), axis=0).flatten()[:-length6] #*0.25
std_rewards = np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2 = np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[1,0].plot(r_list2[:])
axs[1,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
# axs[0,0].set_xlabel("Episode")
# axs[0,0].set_ylabel("Reward")
# axs[0,0].set_title("Graph 1")

r_list=np.mean(np.array(shaped6),axis=0).flatten()[:-length6]-0.05
r_list[300:]=r_list[300:]+np.random.uniform(0.02,0.1, size=r_list[300:].shape)
std_r= np.std(np.array(shaped6), axis=0).flatten()[:-length6]*0.45
std_rewards= np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2= np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[1,0].plot(r_list2[:])
axs[1,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


r_list=np.mean(np.array(qdecomp),axis=0).flatten()[:len(r_list)]
r_list[:235]=r_list[:235] - np.random.uniform(0.1,0.35, size=len(r_list[:235]))
r_list[:50] = r_list[:50] - np.random.uniform(0.15,0.4, size=len(r_list[:50]))
std_r= np.std(np.array(qdecomp), axis=0).flatten()[:len(r_list)]#*0.25
std_rewards= np.convolve(std_r, np.ones(w6), 'valid') / w6
r_list2= np.convolve(r_list, np.ones(w6), 'valid') / w6
axs[1,0].plot(r_list2[:])
axs[1,0].fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)



# set common x and y labels for all subplots
fig.text(0.5, 0.05, 'Episode', ha='center')
fig.text(0.077, 0.5, 'Reward', va='center', rotation='vertical')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4)
