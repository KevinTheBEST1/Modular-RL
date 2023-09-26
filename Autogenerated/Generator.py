import numpy as np
import random
import os
os.chdir(os.getcwd())


num_actions = random.randint(4,20)
num_states = random.randint(num_actions, 100)

# Generate Transition probabilities for each state-action pair
transitions = []
for i in range(num_states):
    row = []
    for j in range(num_actions):
        temp = list(np.random.dirichlet(np.ones(num_states), size=1)[0])#list(a / np.sum(a))#list(np.random.dirichlet(np.ones(num_states), size=1)[0])
        action_probs = []
        for k in range(num_states):
            action_probs.append(temp[k])
        row.append(action_probs)
    transitions.append(row)

terminal_state=np.random.randint(1,num_states)


# Initialize the maximum probability and transition variables
max_probability = 0.0
max_transition = None
# Iterate through each row of the transition array
transitions=np.array(transitions)

# Generate Rewards for each state-action pair
rewards = np.zeros((num_states,num_actions,num_states))#[]
x=np.where(transitions[:,:,terminal_state]==np.max(transitions[:,:,terminal_state]))[0][0]
y=np.where(transitions[:,:,terminal_state]==np.max(transitions[:,:,terminal_state]))[1][0]
rewards[x,y,terminal_state]=np.random.randint(0,100)

indices = np.random.choice(rewards.size, size=num_states-np.random.randint(0,num_states), replace=False)
rewards.flat[indices] = np.random.uniform(-20, 0, size=indices.size)

# Generate Rewards for each state-action pair
rewards2 = np.zeros((num_states,num_actions,num_states))#[]
rewards2[x,y,terminal_state]=np.random.randint(0,100)
indices2 = np.random.choice(rewards2.size, size=num_states-np.random.randint(0,num_states), replace=False)
rewards2.flat[indices2] = np.random.uniform(-20, 0, size=indices2.size)


transitions=transitions.tolist()
rewards=rewards.tolist()
rewards2=rewards2.tolist()
# Write MDP to file
with open('mdp_exp.txt', 'w') as f:
    f.write(f'{num_states}\n{num_actions}\n')
    for row in rewards:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    for row in rewards2:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    for row in transitions:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    f.write('0.99\n')
    f.write(f'{terminal_state}\n')
