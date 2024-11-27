from InvasiveAgent import InvasiveAgent
from InvasiveEnvironment import InvasiveEnvironment
import os
import numpy as np
from rlglue.types import Action
import random
import matplotlib.pyplot as plt
import tracemalloc

# added to check ram usage
# tracemalloc.start()

if __name__ == "__main__":

    nbrReaches = 3
    habitatSize = 2
    # initialize fixed start state
    S = np.array([random.randint(1, 3) for i in range(nbrReaches * habitatSize)])
    # initialize environment using fixed start state
    Exp = InvasiveEnvironment(simulationParameterObj=None, actionParameterObj=None, Bad_Action_Penalty=-10000,fixedStartState=True, 
                              nbrReaches=nbrReaches, habitatSize=habitatSize, seed=1)
    print("Starting fixed state:, ", S)
    Exp.setAgentState(S)
    task_spec = Exp.env_init()

    # initialize agent
    agent = InvasiveAgent()
    agent.agent_init(task_spec)

    # number of steps per episode
    n_steps = 100

    # number of alternate training and evaluating steps
    n_runs = 1000
    # used to save the mean values for each run
    stored_mean = np.ones(n_runs)
    stored_std_dev = np.ones(n_runs)

    # alternate between training and evaluating agent for every 10 episodes
    for i in range(n_runs):
        # train agent for 10 episodes
        for episode in range(10):
            Exp.setAgentState(S)
            # print("initial state", Exp.state)
            first_obs = Exp.env_start()
            first_action = agent.agent_start(first_obs)
            reward_observation_terminal = Exp.env_step(first_action)
            for k in range(n_steps):
                action = agent.agent_step(reward_observation_terminal.r, reward_observation_terminal.o)
                reward_observation_terminal = Exp.env_step(action)
        
        # freeze learning
        agent.agent_message("freeze learning")
        # freeze exploring
        agent.agent_message("freeze exploring")
                
        # evaluate agent for n episodes (updated)
        n = 10 
        for episode in range(n):
            sum = 0  # reset per episode
            sum_of_squares = 0  # reset per episode

            Exp.setAgentState(S)
            first_obs = Exp.env_start()
            first_action = agent.agent_start(first_obs)
            reward_observation_terminal = Exp.env_step(first_action)

            for k in range(n_steps):
                action = agent.agent_step(reward_observation_terminal.r, reward_observation_terminal.o)
                reward_observation_terminal = Exp.env_step(action)
                sum += reward_observation_terminal.r  # accumulate total reward for the episode
            sum_of_squares += sum**2  # outside the for loop for global variance approach 

        mean = sum / n
        stored_mean[i] = mean
        if n <= 1:
            print("Not enough episodes to calculate variance.")
            variance = 0
        else:
            variance = (sum_of_squares - n * mean * mean) / (n - 1.0)
        standard_dev = np.sqrt(variance)
        stored_std_dev[i] = standard_dev
        

        print("run: ", i)
        print("mean: ", mean)
        print("standard deviation: ", standard_dev)

        # # abby debug
        # print(f"Sum of rewards: {sum}")
        # print(f"Sum of squares: {sum_of_squares}")
        # print(f"Mean: {mean}")
        # print(f"Variance calculation: {sum_of_squares - n * mean * mean}")

        
        # unfreeze learning
        agent.agent_message("unfreeze learning")
        # unfreeze exploring
        agent.agent_message("unfreeze exploring")
        
    # added to check ram usage
    # # Display the current memory usage
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print("[ Top 10 Memory Consumers ]")
    # for stat in top_stats[:10]:
    #     print(stat)
    
    plt.plot(stored_mean)
    plt.title("mean")
    plt.figure()
    plt.plot(stored_std_dev)

    plt.show()