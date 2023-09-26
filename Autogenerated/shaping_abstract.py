# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:58:28 2023

@author: Kevin
"""
import os
os.chdir(os.getcwd())
import numpy as np
np.random.seed(2)
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


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
    terminal_state=int((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma,terminal_state

S, A, reward, reward2, P, gamma,terminal_state = read_mdp("mdp_exp4.txt")
import datetime

# Start measuring the execution time
start_time = datetime.datetime.now()
alpha = 0.1
def transition_to_adjacency(transition_matrix):
    num_states = transition_matrix.shape[0]
    adjacency_matrix = np.zeros((num_states, num_states), dtype=int)
    
    for state in range(num_states):
        for action in range(transition_matrix.shape[1]):
            #next_states = transition_matrix[state, action, :]
            #for next_state in range(num_states):
            adjacency_matrix[state, :] = np.zeros(num_states)
            adjacency_matrix[state,np.argmax(transition_matrix[state,action])]=1
    
    return adjacency_matrix

# Example adjacency matrix
adjacency_matrix = transition_to_adjacency(P)

# Find the coordinates of non-zero entries (neighboring states)
row_coords, col_coords = np.where(adjacency_matrix == 1)
neighbor_coords = np.column_stack((row_coords, col_coords))

# Apply k-means clustering
num_clusters = int(S/1.4)  # You can adjust this parameter
if(num_clusters<A):
    num_clusters=A
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(neighbor_coords)

# Create a mapping of abstract state index to original states
abstract_state_to_states = {i: np.where(cluster_labels == i)[0] for i in range(num_clusters)}

# Define the exploration probability and the number of episodes
#epsilon = 0.14
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.004 #0.03#0.0058
n_episodes = 2000
phi=v
'''
[34.079376, 34.12974 , 33.743665, 33.848413, 33.743306, 33.718366,
       33.975734, 43.308205, 33.79027 , 34.109397, 33.832132, 33.815177,
       33.894032, 33.885995, 34.071447, 33.739555, 34.033255, 33.829241,
       33.945045, 33.99349 , 33.855458, 33.80022 , 33.699135, 33.793992,
       33.92199 , 34.045478, 33.879911, 33.90942 , 33.804935, 33.925747,
       33.786155, 33.726258, 33.804177, 33.804819, 33.950618, 33.807448,
       33.803719, 33.774969, 33.810364, 33.726589, 34.164422, 33.904269,
       33.659551, 34.100469, 33.684835, 33.796789, 33.730093, 34.111457,
       33.814295, 34.00371 , 33.866046, 33.831166, 33.912421, 33.859939,
       34.102169, 33.714141, 33.774805, 33.77184 , 34.121287, 33.751203,
       34.079634, 33.822978, 33.945687, 33.868967, 33.925135, 33.830602,
       33.779292, 33.827659, 34.150369, 33.861339, 34.187917, 33.748617,
       33.742318, 33.776953, 34.158748, 33.820366, 33.854793, 34.087065,
       33.822855, 33.82523 , 33.826972, 33.978221, 34.032554, 33.764579,
       34.028244, 33.786184, 33.946165, 34.0021  , 33.695833, 33.691113,
       34.039243, 33.8706  ]
'''
'''
MDP 3
[36.994496, 37.342163, 36.864774, 37.24759 , 37.090662, 37.213243,
       37.029375, 36.874093, 37.050476, 37.017878, 36.874608, 36.973372,
       37.025398, 36.927196, 37.117066, 37.107061, 36.941554, 36.905916,
       37.046453, 36.917466, 36.888712, 37.154573, 37.033471, 37.405491,
       37.329265, 37.183174, 43.893577, 37.094303, 37.022739, 37.089254,
       37.029578, 37.016515, 37.029762, 37.315985, 37.19561 , 37.015179,
       36.870197, 37.093175, 37.165737, 37.050221, 37.64554 , 37.162304,
       36.948675, 36.964177, 37.22699 , 37.046822, 36.874079, 37.056109,
       37.757664, 37.453409, 37.053091, 36.976884, 37.111537, 37.051091,
       37.04933 , 37.366625]
'''
'''
#MDP2
[52.97171 , 52.96707 , 52.752391, 53.174091, 52.804347, 52.966812,
       53.075976, 52.935776, 52.953699, 52.847036, 52.8725  , 53.043331,
       53.06301 , 52.716624, 52.654172, 52.826123, 52.844997, 52.752808,
       52.859716, 53.006157, 52.743587, 53.059651, 53.088083, 53.000876,
       53.721974, 52.97143 , 52.909617, 52.969278, 52.928705, 52.865125,
       53.098091, 52.901734, 52.79906 , 52.897174, 52.955651, 53.357751,
       52.773299, 52.82827 , 53.077371, 53.095999, 53.058916, 52.898271,
       52.835986, 53.035682, 52.752362, 52.936545, 53.213496, 63.623895,
       53.257041, 53.322301, 53.339831, 52.906991, 53.017053, 52.928951,
       53.061017, 53.175268, 53.054303]
'''
#MDP1: 
'''
[103.80167, 102.2237 , 102.13141, 102.13611, 102.04237, 102.89759,
   102.13505, 103.43993, 102.26611, 102.15186, 102.88438, 103.37449,
   102.7111 , 102.99771, 102.6149 , 102.37263, 102.25576, 102.05238,
   101.93277, 103.46774, 102.57395, 103.11949, 102.10422, 118.81163,
   101.72548, 102.75825]
'''
max_steps=S*A
# Run Q-learning
n_runs=10
rewards = np.zeros((n_runs,n_episodes))
for run in range(n_runs):
    # if(run==3 or run==7):
    #     continue
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
            F= gamma*phi[cluster_labels[next_state]] - phi[cluster_labels[state]]
            Q[state, action] += alpha * (r+F + gamma * np.max(Q[next_state, :]) - Q[state, action])
            cum_R.append(r)
            # Update the state
            state = next_state
            if(state==terminal_state or step==max_steps):
                done=1
        #r_list.append(sum(cum_R))
        rewards[run,episode]=sum(cum_R)
        if egreedy > egreedy_final:
            egreedy -= egreedy*egreedy_decay
        # Check for convergence
        delta = np.max(np.abs(Q - np.max(Q)))
        # if delta < 0.000001:
        #     break
end_time = datetime.datetime.now()
total_time = (end_time - start_time).total_seconds() * 1000
# Print the total time taken
print("Total time taken: {:.2f} milliseconds".format(total_time))

# Print the optimal policy
policy = np.argmax(Q, axis=1)
#print(policy)
import pandas as pd
pd.DataFrame(rewards).to_csv("traditional_shaped4_abstract.csv")
'''
w=10
plt.plot(r_list[:-w], alpha=0.3, color='orange')
r_list2= np.convolve(r_list, np.ones(w), 'valid') / w
plt.plot(r_list2,color='orange')
'''
state=0
rsum=[]
while state != terminal_state:
    action=np.argmax(Q[state, :])
    next_state = np.argmax(P[state, action, :])
    rsum.append(reward[state, action, next_state] + reward2[state, action, next_state])
    state=next_state
print(sum(rsum))
    
