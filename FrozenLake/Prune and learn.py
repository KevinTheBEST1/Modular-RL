import os
import numpy as np
from pulp import *
from matplotlib import pyplot as plt
import pickle
import random
import pandas as pd

os.chdir(os.getcwd())
# os.system("python LB1.py ")
# os.system("python LB2.py ")
# os.system("python b1.py ")
# os.system("python b2.py ")
# input()
# q_p,pp = lp(T1, R+R2, gamma)
# q_m,pm = lp(T1, -R-R2, gamma)
with open('q1.pickle', 'rb') as file:
	q1_p = pickle.load(file)
with open('q2.pickle', 'rb') as file:
	q2_p = pickle.load(file)

		
with open('q1m_fit.pickle', 'rb') as file:
	q1_m = pickle.load(file)
with open('q2m_fit.pickle', 'rb') as file:
	q2_m = pickle.load(file)


def getQ(q, state, action):
    return q.get((state, action), 0.0)	
q_p= {k: q1_p.get(k, 0) + q2_p.get(k, 0) for k in set(q1_p) | set(q2_p)}
q_m= {k: -(q1_m.get(k, 0) + q2_m.get(k, 0)) for k in set(q1_m) | set(q2_m)}

S=16
A=4
prob_main = pulp.LpProblem('Main', LpMinimize)
decision_variables_phi = pulp.LpVariable.dicts('phi', range(S))
formula = 0.0
for i in range(S):
    for a in range(A):
        formula += q_p[i,a]-q_m[i,a]-2*decision_variables_phi[i]
        prob_main+= q_p[i,a]-decision_variables_phi[i]>=q_m[i,a]+decision_variables_phi[i]
prob_main += formula
#prob += formula>=0
# for i in range(S):
#     prob_main += decision_variables_phi[i]<=1000
#     prob_main += decision_variables_phi[i]>=-1000
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


import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
random.seed(1)
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
# )

env = gym.make('FrozenLake-v1',is_slippery=False, render_mode="ansi", map_name="4x4")

# Instantiate the Environment.
# env = gym.make('FrozenLake-v0')

# To check all environments present in OpenAI
# print(envs.registry.all())

env.reset()
env.render()

# Total number of States and Actions
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
print( "States = ", number_of_states)
print( "Actions = ", number_of_actions)

num_episodes = 2000


# PARAMS 

# Discount on reward
gamma = 0.95

# Factor to balance the ratio of action taken based on past experience to current situtation
learning_rate = 0.9


# exploit vs explore to find action
# Start with 70% random actions to explore the environment
# And with time, using decay to shift to more optimal actions learned from experience

egreedy = 0.9
egreedy_final = 0.1
egreedy_decay = 0.08
n_runs=20
r_list=np.zeros((n_runs,num_episodes))
stepPerEp=[]
for run in range(n_runs):
    Q = torch.zeros([number_of_states, number_of_actions])
    torch.manual_seed(1+run) 
    #random.seed(run)
    steps_total = []
    
    egreedy_total = []
    for i_episode in range(num_episodes):
        rewards_total = []
        # resets the environment
        state = env.reset()[0]
        step = 0
    
        while True:
            
            step += 1
            
            random_for_egreedy = torch.rand(1)[0]
            
    
            if random_for_egreedy > egreedy:      
                random_values = Q[state] + torch.rand(1,number_of_actions) / 1000      
                action = torch.max(random_values,1)[1][0]  
                action = action.item()
            else:
                action = random.choice(list(state_action[state]))#env.action_space.sample()
                
            
            
            new_state, reward, done, i, _ = env.step(action)
            
            if(done):
                if(reward==0):
                    reward=-1
                else:
                    reward=reward
    
            # Filling the Q Table
            Q[state, action] = reward + gamma * torch.max(Q[new_state])
            
            # Setting new state for next action
            state = new_state
            
            # env.render()
            # time.sleep(0.4)
            
            if done:
                steps_total.append(step)
                rewards_total.append(reward)
                egreedy_total.append(egreedy)
                if i_episode % 10 == 0:
                    print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode,reward, step))
                break
        r_list[run,i_episode]=sum(rewards_total)
        if egreedy > egreedy_final:
            egreedy -= egreedy*egreedy_decay
        stepPerEp.append(step)
    