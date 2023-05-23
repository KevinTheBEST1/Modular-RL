# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:58:28 2023

@author: Kevin
"""
import os
os.chdir(os.getcwd())
import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt

def read_mdp(mdp):

    """Function to read MDP file"""
    #mdp="mdp_new.txt"
    f = open(mdp)

    S = int(f.readline())
    A = int(f.readline())

    # Initialize Transition and Reward arrays
    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
     # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    # Update the Transition Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            #print((s,a))
            for sPrime in range(S):
                #print(line[sPrime], end=" ")
                T[s][a][sPrime] = line[sPrime]
                 
            #print()

    # Read the value of gamma
    gamma = float(f.readline().rstrip())

    f.close()

    return S, A, R, R2, T, gamma

S, A, reward, reward2, P, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
alpha = 0.1

# Define the exploration probability and the number of episodes
#epsilon = 0.14
egreedy = 0.7
egreedy_final = 0.05
egreedy_decay = 0.01
n_episodes = 800


# Run Q-learning
n_runs=5
rewards = np.zeros((n_runs,n_episodes))
for run in range(n_runs):
    egreedy = 0.7
    np.random.seed(run)
    print("############################################## "+str(run+1) )
    r_list=[]
    # Initialize the Q-function
    Q = np.zeros((S, A))
    for episode in range(n_episodes):
        print(episode)
        # Initialize the state
        state = 0
        step=0
        cum_R=[]
        done=0
        # Loop until the terminal state is reached
        while not done:
            step=step+1
            # Choose an action
            if np.random.rand() < egreedy:
                action = np.random.randint(A)
            else:
                action = np.argmax(Q[state, :])
            # Take the action and observe the next state and reward
            next_state = np.argmax(P[state, action, :])#np.random.choice(S, p=P[state, action, :])
            r=reward[state, action, next_state] + reward2[state, action, next_state]
            Q[state, action] += alpha * (r + gamma * np.max(Q[next_state, :]) - Q[state, action])
            cum_R.append(r)
            # Update the state
            state = next_state
            if(r>0):
                done=1
        #r_list.append(sum(cum_R))
        rewards[run,episode]=sum(cum_R)
        if egreedy > egreedy_final:
            egreedy -= egreedy*egreedy_decay
        # Check for convergence
        delta = np.max(np.abs(Q - np.max(Q)))
        if delta < 0.001:
            break
