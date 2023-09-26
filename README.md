# Modular-RL
We introduce reward adaptation (RA), the problem where  the learning agent adapts to a target reward function based on one or multiple existing behaviors learned a priori with their corresponding source reward functions.  Reward adaptation has many applications, such as adapting an autonomous driving agent that can already operate either fast or safe to operating both fast and safe. We assume that the target reward function is a polynomial function of the source reward functions. 
Learning the target behavior from scratch is  possible but inefficient given the  existing source behaviors.  For a more efficient solution, we propose to boost reward adaptation by manipulating the Q functions of the source behaviors, which are assumed to be accessible. We formally prove that our pruning strategy does not affect the optimality of the returned policy for improving sample efficiency.  Comparison with baselines is performed in a variety of synthetic and simulation domains to demonstrate the effectiveness and generalizability of our approach. 

###  Requirements: ### 
* PuLP 2.7.0
* pandas
* matplotlib
* numpy

In order to run the code, run python files in each experiment folder.