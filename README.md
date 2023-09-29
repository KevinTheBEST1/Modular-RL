# Modular-RL
We introduce reward adaptation (RA), the problem where the learning agent adapts to a target reward function based on one or multiple existing behaviors learned a priori based on their corresponding source reward functions, providing a new perspective of modular reinforcement learning. Reward adaptation has many applications, such as adapting an autonomous driving agent that can already operate either fast or safe to operating both fast and safe. Learning the target behavior from scratch is possible but inefficient given the source behaviors available. Assuming that the target reward function is a polynomial function of the source reward functions, we propose an approach to reward adaptation by manipulating variants of the Q function for the source behaviors, which are assumed to be accessible and obtained when learning the source behaviors prior to learning the target behavior. It results in a novel method named ``Q-Manipulation'' that enables action pruning before learning the target. We formally prove that our pruning strategy for improving sample complexity does not affect the optimality of the returned policy. Comparison with baselines is performed in a variety of synthetic and simulation domains to demonstrate its effectiveness and generalizability.

###  Requirements: ### 
* PuLP 2.7.0
* pandas
* matplotlib
* numpy

In order to run the code, run python files in each experiment folder.
