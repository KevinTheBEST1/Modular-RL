import os
os.chdir(os.getcwd())
import numpy as np
np.random.seed(2)
from matplotlib import pyplot as plt
import sarsa
def read_mdp(mdp):
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

S, A, reward, reward2, P, gamma,terminal_state = read_mdp("mdp_exp3.txt")
alpha = 0.1
import datetime

# Start measuring the execution time
start_time = datetime.datetime.now()
# Define the exploration probability and the number of episodes
#epsilon = 0.14
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.004
n_episodes = 2000

max_steps=S*A
# Run Q-learning
n_runs=20
rewards = np.zeros((n_runs,n_episodes))
for run in range(n_runs):
    # if(run==3 or run==7):
    #     continue
    agent = sarsa.Sarsa(actions=range(A), alpha=0.1, gamma=0.9, epsilon=0.1)
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
            # Take the action and observe the next state and reward
            action=agent.chooseAction(state, egreedy)
            next_state = np.argmax(P[state, action, :])#np.random.choice(S, p=P[state, action, :])
            r=reward[state, action, next_state] + reward2[state, action, next_state]
            next_action = agent.chooseAction(next_state, egreedy)
            agent.learn(state, action, reward[state, action, next_state], reward2[state, action, next_state], next_state, next_action)
            cum_R.append(r)
            # Update the state
            state = next_state
            if(state==terminal_state or step==max_steps):
                 done=1
        #r_list.append(sum(cum_R))
        rewards[run,episode]=sum(cum_R)
        if egreedy > egreedy_final:
            egreedy -= egreedy*egreedy_decay

end_time = datetime.datetime.now()
total_time = (end_time - start_time).total_seconds() * 1000
# Print the total time taken
print("Total time taken: {:.2f} milliseconds".format(total_time))
