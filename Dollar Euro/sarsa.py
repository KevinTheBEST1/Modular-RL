#import random
import numpy as np

class Sarsa:
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9, seed=1):
        self.q1 = {}
        self.q2 = {}
        np.random.seed(seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q1.get((state, action), 0.0), self.q2.get((state, action), 0.0)

    def learnQ(self, state, action, r1, r2, value1, value2):
        oldv1 = self.q1.get((state, action), None)
        oldv2 = self.q2.get((state, action), None)
        if oldv1 is None or oldv2 is None:
            self.q1[(state, action)] = r1 #reward
            self.q2[(state, action)] = r2 #reward
        else:
            self.q1[(state, action)] = oldv1 + self.alpha * (value1 - oldv1)
            self.q2[(state, action)] = oldv2 + self.alpha * (value2 - oldv2)

    def chooseAction(self, state, egreedy):
        if np.random.rand() < egreedy:
            action = np.random.choice(self.actions)
        else:
            #print([self.getQ(state, a) for a in self.actions])
            q_mix = [self.getQ(state, a) for a in self.actions]
            q=[]
            for i in q_mix:
                q.append(i[0] +i[1])
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 0:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = np.random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, r1, r2, state2, action2):
        qnext1, qnext2 = self.getQ(state2, action2)
        self.learnQ(state1, action1, r1, r2, r1 + self.gamma * qnext1, r2 + self.gamma * qnext2)







