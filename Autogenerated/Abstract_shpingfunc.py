import numpy as np
from sklearn.cluster import KMeans
from pulp import *


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

S, A, r, r2, P, gamma,terminal_state = read_mdp("mdp_exp4.txt")

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

# Example rewards and transition probabilities matrices
rewards_matrix = r

rewards_matrix2 = r2


transition_probs_matrix = P

# Find the coordinates of non-zero entries (neighboring states)
row_coords, col_coords = np.where(adjacency_matrix == 1)
neighbor_coords = np.column_stack((row_coords, col_coords))

# Apply k-means clustering
num_clusters = int(S/1.4) #int(S/A)  # You can adjust this parameter
if(num_clusters<A):
    num_clusters=A
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(neighbor_coords)

# Create a mapping of abstract state index to original states
abstract_state_to_states = {i: np.where(cluster_labels == i)[0] for i in range(num_clusters)}

def generate_abstract_reward_transition(rewards_matrix, rewards_matrix2, transition_probs_matrix, abstract_state_to_states):
    num_abstract_states = len(abstract_state_to_states)
    abstract_rewards = np.zeros((num_abstract_states, rewards_matrix.shape[1], num_abstract_states))
    abstract_rewards2 = np.zeros((num_abstract_states, rewards_matrix.shape[1], num_abstract_states))
    abstract_transitions = np.zeros((num_abstract_states, transition_probs_matrix.shape[1], num_abstract_states))
    
    for abstract_state in range(num_abstract_states):
        original_states = abstract_state_to_states[abstract_state]
        for action in range(rewards_matrix.shape[1]):
            for next_abstract_state in range(num_abstract_states):
                next_original_states = abstract_state_to_states[next_abstract_state]
                total_reward = 0.0
                total_reward2 = 0.0
                total_transition = 0.0
                for original_state in original_states:
                    for next_original_state in next_original_states:
                        total_reward += rewards_matrix[original_state, action, next_original_state]
                        total_reward2 += rewards_matrix2[original_state, action, next_original_state]
                        total_transition += transition_probs_matrix[original_state, action, next_original_state]
                abstract_rewards[abstract_state, action, next_abstract_state] = total_reward / (len(original_states) * len(next_original_states))
                abstract_rewards2[abstract_state, action, next_abstract_state] = total_reward2 / (len(original_states) * len(next_original_states))
                abstract_transitions[abstract_state, action, next_abstract_state] = total_transition / (len(original_states) * len(next_original_states))
    
    return abstract_rewards, abstract_rewards2, abstract_transitions

# Generate abstract reward and transition matrices
abstract_rewards, abstract_rewards2, abstract_transitions = generate_abstract_reward_transition(rewards_matrix, rewards_matrix2, transition_probs_matrix, abstract_state_to_states)


def find_v(T, R, gamma, policy):

    """Function to find value function V"""

    # Initialize arrays of zeros for Value function after and before update
    V1 = np.zeros(T.shape[0])
    V0 = np.zeros(T.shape[0])

    while(1):
        # Until the V1 and V0 are close enough element wise
        for s in range(T.shape[0]):
            # Find the V1
            V1[s] = np.sum(T[s, policy[s], :] * R[s, policy[s], :] +
                           gamma * T[s, policy[s], :] * V0)
        # If V1 and V0 are close enough
        if np.allclose(V1, V0, rtol=1e-13, atol=1e-15):
            break
        else:
            # Update V0 with V1
            np.copyto(V0, V1)
    return V1


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
    prob += sum(decision_variables.values())

    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            # Add constraint to LP for each state and action
            formula = 0.0
            for sPrime in range(T.shape[2]):
                formula += (T[s, a, sPrime] * (R[s, a, sPrime] +
                            gamma * decision_variables[sPrime]))
            prob += decision_variables[s] >= formula

    # Solve the LP Problem and get results in V
    prob.solve()  # solvers.PULP_CBC_CMD(fracGap=0.000000001)
    V = np.array([v.varValue for v in prob.variables()])

    return V


def lp(T, R, gamma):

    """Implementation of LP"""

    # Initialise policy to all zeros
    policy = [0 for i in range(T.shape[0])]

    # Find V and Q
    V = solve_lp(T, R, gamma)
    Q = find_q(V, T, R, gamma)
    policy=np.argmax(np.array(Q),axis=1)

    return V, policy, Q

v, p, q=lp(abstract_transitions,abstract_rewards+abstract_rewards2,gamma)
