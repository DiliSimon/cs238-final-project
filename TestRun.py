from InvasiveAgent import InvasiveAgent
from InvasiveEnvironment import InvasiveEnvironment
import os
import numpy as np
from rlglue.types import Action
import random


if __name__ == "__main__":
    nbrReaches = 7
    habitatSize = 4
    # initialize fixed start state
    S = np.array([random.randint(1, 3) for i in range(nbrReaches * habitatSize)])
    # initialize environment using fixed start state
    Exp = InvasiveEnvironment(simulationParameterObj=None, actionParameterObj=None, Bad_Action_Penalty=-10000,fixedStartState=True, nbrReaches=7,
            habitatSize=4, seed=1)
    print("Starting fixed state:, ", S)
    Exp.setAgentState(S)
    task_spec = Exp.env_init()

    # initialize agent
    agent = InvasiveAgent()
    agent.agent_init(task_spec)

    # number of steps per episode
    n_steps = 100

    # alternate between training and evaluating agent for every 10 episodes
    for i in range(30):
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
        
        # evaluate agent for n episodes
        n = 10 
        sum = 0
        sum_of_squares = 0
        for episode in range(n):
            Exp.setAgentState(S)
            # print("initial state", Exp.state)
            first_obs = Exp.env_start()
            first_action = agent.agent_start(first_obs)
            reward_observation_terminal = Exp.env_step(first_action)
            for k in range(n_steps):
                action = agent.agent_step(reward_observation_terminal.r, reward_observation_terminal.o)
                reward_observation_terminal = Exp.env_step(action)
                sum += reward_observation_terminal.r
                sum_of_squares += sum**2
        
        mean = sum / n
        variance = (sum_of_squares - n * mean * mean) / (n - 1.0)
        standard_dev = np.sqrt(variance)

        print("run: ", i)
        print("mean: ", mean)
        print("standard deviation: ", standard_dev)
        
        # unfreeze learning
        agent.agent_message("unfreeze learning")