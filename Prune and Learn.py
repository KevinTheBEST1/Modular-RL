# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:29:55 2023

@author: Kevin
"""

import os
import numpy as np
np.random.seed(2)
from pulp import *
from matplotlib import pyplot as plt

os.chdir(r"C:\Users\Kevin\Desktop\Linear program")

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

S, A, R, R2, T1, gamma = read_mdp("gridworld_mdp.txt")

def find_q(V, T, R, gamma):

    """Function to find action value function Q"""

    # Initialize arrays of zeros for Value function after and before update
    Q = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        # Find action value for each state action pair
        Q[s] = np.sum(T[s] * R[s] + gamma * T[s] * V, axis=1)

    return Q

def solve_lp(T, R, gamma):

    """Function to solve Linear Programming using PuLP"""

    # Setting up problem and decision variables
    prob = pulp.LpProblem('mdp_lp', LpMinimize)
    decision_variables = pulp.LpVariable.dicts('v', range(T.shape[0]))
    # Objective function
    #print(decision_variables.values())
    prob += sum(list(decision_variables.values())) #np.sum(decision_variables.values()) 
    #print(decision_variables)
    #print("----------------------------------")
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            # Add constraint to LP for each state and action
            formula = 0.0
            for sPrime in range(T.shape[2]):
                formula += (T[s, a, sPrime] * (R[s, a, sPrime] + gamma * decision_variables[sPrime]))
            prob += decision_variables[s] >= formula
    #print(prob.variables()[-4:])
    # Solve the LP Problem and get results in V
    prob.solve()  # solvers.PULP_CBC_CMD(fracGap=0.000000001)
    V = np.array([v.varValue for v in prob.variables()])
    #print(V.T)
    return V


def lp(T, R, gamma):

    """Implementation of LP"""

    # Initialise policy to all zeros
    policy = [0 for i in range(T.shape[0])]

    # Find V and Q
    V = solve_lp(T, R, gamma)
    Q = find_q(V, T, R, gamma)

    # For each state, if action_0 value is less than action_1 value,
    # change its action to action_1
    '''
    for s in range(T.shape[0]):
        if (Q[s][0] < Q[s][1]) and (policy[s] != 1):
            policy[s] = 1
    '''
    policy='P'
    return Q, policy

q_p1,pp1 = lp(T1, R, gamma)
q_m1,pm1 = lp(T1, -R, gamma)

q_p2,pp2 = lp(T1, R2, gamma)
q_m2,pm2 = lp(T1, -R2, gamma)

q_p = q_p1 + q_p2
q_m = -(q_m1 + q_m2)




################################################################################
################################################################################
################################################################################

# q_p,pp = lp(T1, R+R2, gamma)
# q_m,pm = lp(T1, -R-R2, gamma)



prob_main = pulp.LpProblem('Main', LpMinimize)
decision_variables_phi = pulp.LpVariable.dicts('phi', range(S))
formula = 0.0
for i in range(S):
    for a in range(A):
        formula += q_p[i,a]-q_m[i,a]-2*decision_variables_phi[i]
        prob_main+= q_p[i,a]-decision_variables_phi[i]>=q_m[i,a]+decision_variables_phi[i]
prob_main += formula
#prob += formula>=0
for i in range(S):
    prob_main += decision_variables_phi[i]<=1000
    prob_main += decision_variables_phi[i]>=-1000
prob_main.solve()
V = np.array([v.varValue for v in prob_main.variables()])
#print(prob_main)
#print()
for i in range(S):
    for j in range(A):
        q_p[i,j]= q_p[i,j] - decision_variables_phi[i].varValue
        q_m[i,j]= q_m[i,j] + decision_variables_phi[i].varValue
# print(q_p)
# print()
# print(q_m)

info=[]
final_actions=set(list(range(A)))
prune={}
state_action={}
c=0
def getQ(q, state, action):
    return q.get((state, action), 0.0)
for i in range(S):
    alist=[]
    for action_l in range(A):
        for action_u in range(A):
            if(action_l==action_u):
                continue
            if( q_m[i, action_l] > q_p[i,action_u] ):
                info.append((i,action_l, action_u))
                alist.append(action_u)
                c=c+1
    prune[i]= set(alist)
    state_action[i]= final_actions.difference(set(alist))


# Define the exploration probability and the number of episodes
alpha = 0.1
epsilon = 0.3
n_episodes = 800
reward=R
reward2=R2
P=T1


# Run Q-learning
n_runs=5
rewards = np.zeros((n_runs,n_episodes))
for run in range(n_runs):
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
        # Loop until the terminal state is reached
        while step <= S*A*2:
            step=step+1
            # Choose an action
            if np.random.rand() < epsilon:
                action = np.random.choice(list(state_action[state]))
            else:
                action = np.argmax(Q[state, :])
            
            # Take the action and observe the next state and reward
            next_state = np.random.choice(S, p=P[state, action, :])
            #r = reward[state, action, next_state] + reward2[state, action, next_state]
            r_total=[]
            # Update the Q-function
            for i in range(S):
                r = P[state, action, i] * (reward[state, action, i] + reward2[state, action, i])
                r_total.append(r)
                Q[state, action] += alpha * (r + P[state, action, i]*gamma * np.max(Q[i, :]) - Q[state, action])
            cum_R.append(sum(r_total))
            # Update the state
            state = next_state
        r_list.append(sum(cum_R))
        rewards[run,episode]=sum(cum_R)
        # Check for convergence
        delta = np.max(np.abs(Q - np.max(Q)))
        if delta < 0.00001:
            break

# Print the optimal policy
policy = np.argmax(Q, axis=1)
#print(policy)
import pandas as pd
pd.DataFrame(rewards).to_csv("pruned.csv")
'''
w=10
plt.plot(r_list[:-w], alpha=0.3, color='orange')
r_list2= np.convolve(r_list, np.ones(w), 'valid') / w
plt.plot(r_list2,color='orange')
'''
