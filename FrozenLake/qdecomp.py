#4th run
import sarsa

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
max_steos=2000

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
egreedy_decay = 0.99
n_runs=20
r_list=np.zeros((n_runs,num_episodes))
stepPerEp=[]
for run in range(n_runs):
    agent = sarsa.Sarsa(actions=range(number_of_actions), alpha=0.1, gamma=0.9, epsilon=0.1)
    Q = torch.zeros([number_of_states, number_of_actions])
    torch.manual_seed(run) 
    steps_total = []
    egreedy_total = []
    for i_episode in range(num_episodes):
        rewards_total = []
        # resets the environment
        state = env.reset()[0]
        step = 0
    
        while True:
            
            step += 1
            
            #random_for_egreedy = torch.rand(1)[0]
            
    
            # if random_for_egreedy > egreedy:      
            #     random_values = Q[state] + torch.rand(1,number_of_actions) / 1000      
            #     action = torch.max(random_values,1)[1][0]  
            #     action = action.item()
            # else:
            #     action = env.action_space.sample()
            action = agent.chooseAction(state, egreedy)
            new_state, reward, done, info, _ = env.step(action)
            r1=0
            r2=0
            if(done):
                if(reward==0):
                    r1=-0
                    r2=-1
                    #reward=r1+r2
                else:
                    r1=0.5
                    r2=0.5
                    #reward=reward
            
            new_action = agent.chooseAction(new_state, egreedy)
            # Filling the Q Table
            #Q[state, action] = reward + gamma * torch.max(Q[new_state])
            agent.learn(state, action, r1, r2, new_state, new_action)
            # Setting new state for next action
            state = new_state
            
            # env.render()
            # time.sleep(0.4)
            if(max_steos< step):
                done=True
            if done:
                steps_total.append(step)
                rewards_total.append(reward)
                egreedy_total.append(egreedy)
                if i_episode % 10 == 0:
                    print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode,reward, step))
                break
        r_list[run,i_episode]=sum(rewards_total)
        if egreedy > egreedy_final:
            egreedy = egreedy*egreedy_decay
        stepPerEp.append(step)
    
