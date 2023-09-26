import numpy as np
np.random.seed(1)
# Gridworld dimensions
num_rows = 5
num_cols = 9

# Actions: 0 - up, 1 - right, 2 - down, 3 - left
num_actions = 4

# SARSA parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.971
epsilon_min = 0.1
num_episodes = 500

# Rewards
rewards = np.zeros((num_rows, num_cols))
rewards[0, 0] = 0.5
rewards[0, 8] = 0.5
rewards[4, 4] = 0.6

rewards2 = np.zeros((num_rows, num_cols))
rewards2[0, 0] = 0.5
rewards2[0, 8] = 0.5
rewards2[4, 4] = 0.6

# Initialize Q-values
q_values1 = np.zeros((num_rows, num_cols, num_actions))
q_values2 = np.zeros((num_rows, num_cols, num_actions))

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    q_values=np.zeros(num_actions)
    for i in range(num_actions):
        q_values[i]= q_values1[state[0],state[1],i] + q_values2[state[0],state[1],i]
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(q_values)

n_runs = 20
R = np.zeros((n_runs, num_episodes))

# Main SARSA loop
for run in range(n_runs):
    np.random.seed(run)
    r_list = []
    q_values = np.zeros((num_rows, num_cols, num_actions))
    print(f"Run {run + 1}")
    epsilon = epsilon_initial
    
    for episode in range(num_episodes):
        state = (0, 4)  # Start state
        total_reward = 0
        cum_R = []
        action = epsilon_greedy_policy(state, epsilon)
        
        while state != (0, 0) and state != (0, 8) and state != (4, 4):
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
            reward2 = rewards2[next_row, next_col]
            total_reward += reward
            
            next_action = epsilon_greedy_policy((next_row, next_col), epsilon)
            
            # SARSA update
            q_values1[row, col, action] += learning_rate * (
                reward + discount_factor * q_values[next_row, next_col, next_action] - q_values[row, col, action]
            )
            q_values1[row, col, action] += learning_rate * (
                reward2 + discount_factor * q_values[next_row, next_col, next_action] - q_values[row, col, action]
            )
            
            state = (next_row, next_col)
            action = next_action
            cum_R.append(reward+reward2)
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        R[run, episode] = sum(cum_R)