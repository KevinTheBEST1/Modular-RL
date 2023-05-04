# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:03:43 2023

@author: Kevin
"""
import os
os.chdir(os.getcwd())
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pruned=pd.read_csv("pruned.csv",index_col=None).drop("Unnamed: 0",axis=1)

traditional=pd.read_csv("traditional.csv", index_col=None).drop("Unnamed: 0",axis=1)

#indv=pd.read_csv("prunedIdvPhi.csv", index_col=None).drop("Unnamed: 0",axis=1)

w=150#15#150
length=1300#170#1300
r_list=np.mean(np.array(pruned), axis=0).flatten()[:-length]
std_r= np.std(np.array(pruned), axis=0).flatten()[:-length]#*0.25
std_rewards= np.convolve(std_r, np.ones(w), 'valid') / w
r_list2= np.convolve(r_list, np.ones(w), 'valid') / w
plt.plot(r_list2[:])
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


r_list=np.mean(np.array(traditional),axis=0).flatten()[:-length]
std_r= np.std(np.array(pruned), axis=0).flatten()[:-length]#*0.25
std_rewards= np.convolve(std_r, np.ones(w), 'valid') / w
r_list2= np.convolve(r_list, np.ones(w), 'valid') / w
plt.plot(r_list2[:])
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

'''
r_list=np.array(indv).flatten()#[:-200]
r_list2= np.convolve(r_list, np.ones(w), 'valid') / w
plt.plot(r_list2[:470+w])
'''
#plt.legend(['Pruned', 'Traditional'],loc="lower right")
plt.xlabel("Episode")
plt.ylabel("Reward")