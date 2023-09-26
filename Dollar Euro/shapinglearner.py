import os
os.chdir(os.getcwd())
import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt

#phi=[120. , 118.8, 120. , 118.8]
V=np.load('values.npy')
phi=np.zeros(15)
c=0
for i in range(3):
    for j in range(5):
        phi[c]=V[i,j]
        c=c+1
def z(s):
    if(s in [0,1,9,10]):
        return 0
    elif(s in [2,3,11,12]):
        return 1
    elif(s in [4,13]):
        return 2
    elif(s in [5,6,14,15]):
        return 3
    elif(s in [7,8,16,17]):
        return 4
    elif(s in [18,19,27,28]):
        return 5
    elif(s in [20,21,29,30]):
        return 6
    elif(s in [22,31]):
        return 7
    elif(s in [23,24,32,33]):
        return 8
    elif(s in [25,26,34,35]):
        return 9
    elif(s in [36,37]):
        return 10
    elif(s in [38,39]):
        return 11
    elif(s in [40]):
        return 12
    elif(s in [41,42]):
        return 13
    elif(s in [43,44]):
        return 14
    else:
        return None#print("***********"+str(s))


# Gridworld dimensions
num_rows = 5
num_cols = 9

# Actions: 0 - up, 1 - right, 2 - down, 3 - left
num_actions = 4

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.972
epsilon_min = 0.1
num_episodes = 500

# Rewards
rewards = np.zeros((num_rows, num_cols))
rewards[0, 0] = 1.0
rewards[0, 8] = 1.0
rewards[4, 4] = 1.2

# Initialize Q-values
q_values = np.zeros((num_rows, num_cols, num_actions))

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(q_values[state[0], state[1]])

n_runs=20
R = np.zeros((n_runs,num_episodes))
goal=(4,4)
# Main Q-learning loop
for run in range(n_runs):
    np.random.seed(run)
    r_list=[]
    q_values = np.zeros((num_rows, num_cols, num_actions))
    print(f"Run {run + 1}")
    epsilon = epsilon_initial
    
    for episode in range(num_episodes):
        state = (0, 4)  # Start state
        total_reward = 0
        cum_R=[]
        while state != (0, 0) and state != (0, 8) and state != (4, 4):
            action = epsilon_greedy_policy(state, epsilon)
            
            row, col = state
            next_row, next_col = row, col
            
            if action == 0:  # down
                next_row = min(row + 1, num_rows - 1)
            elif action == 1:  # up
                next_row = max(row - 1, 0)
            elif action == 2:  # left
                next_col = max(col - 1, 0)
            elif action == 3:  # right
                next_col = min(col + 1, num_cols - 1)
            
            reward = rewards[next_row, next_col]
            total_reward += reward
            F= discount_factor *phi[z(num_cols*next_row+next_col)] - phi[z(num_cols*row+col)]#*(abs(next_row-goal[0])+abs(next_row-goal[1])) - (abs(row-goal[0])+abs(col-goal[1]))#*phi[z(num_cols*next_row+next_col)] + phi[z(num_cols*row+col)]
            # Q-learning update
            q_values[row, col, action] += learning_rate * (
                reward+F + discount_factor * np.max(q_values[next_row, next_col]) - q_values[row, col, action]
            )
            
            state = (next_row, next_col)
            cum_R.append(reward)
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        R[run,episode]=sum(cum_R)
 