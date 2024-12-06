from InvasiveAgent import InvasiveAgent
from InvasiveAgent2 import InvasiveAgent2
from InvasiveEnvironment import InvasiveEnvironment
from Utilities import SamplingUtility, InvasiveUtility # added for random_policies
import os
import numpy as np
from rlglue.types import Action
import random
import matplotlib.pyplot as plt
import tracemalloc

# added to check ram usage
# tracemalloc.start()


def random_policy(environment, n_steps, initial_state):
    """
    Execute a random policy in the given environment for a specified number of steps.
    Args:
        environment: The environment instance where the policy will run.
        n_steps: Number of steps to run the policy.
    Returns:
        total_reward: Cumulative reward obtained by executing the random policy.
    """
    total_reward = 0

    # Reset the environment to a random initial state
    environment.setAgentState(initial_state)

    # Start the environment
    observation = environment.env_start()

    for _ in range(n_steps):
        # Retrieve valid actions for the current state
        state = observation.intArray  # Current state
        available_actions = InvasiveUtility.getActions(state, environment.nbrReaches, environment.habitatSize)

            # Ensure there are valid actions
        if not available_actions:
            print(f"No valid actions for state {state}. Ending policy execution.")
            break

        # Select a random action
        random_action_tuple = random.choice(available_actions)
                # Wrap the action in the appropriate format
        random_action = Action()
        random_action.intArray = list(random_action_tuple)

        # Take the random action and observe the results
        reward_observation_terminal = environment.env_step(random_action)
        total_reward += reward_observation_terminal.r

        # Stop if the episode ends
        if reward_observation_terminal.terminal:
            break

    return total_reward

# added to check ram usage
# tracemalloc.start()

if __name__ == "__main__":

    nbrReaches = 6
    habitatSize = 2
    # initialize fixed start state
    S = np.array([random.randint(1, 3) for i in range(nbrReaches * habitatSize)])
    # initialize environment using fixed start state
    Exp = InvasiveEnvironment(simulationParameterObj=None, actionParameterObj=None, Bad_Action_Penalty=-10000,fixedStartState=True, 
                              nbrReaches=nbrReaches, habitatSize=habitatSize, seed=1)
    print("Starting fixed state:, ", S)
    Exp.setAgentState(S)
    task_spec = Exp.env_init()

    # initialize agent for SARSA
    agent = InvasiveAgent()
    agent.agent_init(task_spec)

    # initialize agent for Q-learning
    agent2 = InvasiveAgent2()
    agent2.agent_init(task_spec)

    # number of steps per episode
    n_steps = 100

# Evaluate random policy
    random_rewards = [random_policy(Exp, n_steps, S) for _ in range(10)]  # Run for 10 episodes
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)

    print("Random Policy - Mean Reward:", random_mean)
    print("Random Policy - Std Deviation:", random_std)


    # number of alternate training and evaluating steps
    n_runs = 1000*10
    
    '''SARSA'''
    # used to save the mean values for each run
    stored_mean_SARSA = np.ones(n_runs)
    stored_std_dev_SARSA = np.ones(n_runs)

    # alternate between training and evaluating agent for every 10 episodes 
    for i in range(n_runs):

        # Dynamically adjust parameters during training
        # if i % 50 == 0 and i > 0:  # Adjust every 50 runs
        #     new_epsilon = max(0.01, agent.sarsa_epsilon * 0.90)  # Reduce exploration
        #     new_stepsize = max(0.01, agent.sarsa_stepsize * 0.99)  # Reduce learning rate
        #     agent.adjust_parameters(new_epsilon=new_epsilon, new_stepsize=new_stepsize)
        #     print(f"Run {i}: Adjusted sarsa_epsilon to {agent.sarsa_epsilon}, sarsa_stepsize to {agent.sarsa_stepsize}")

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
        stored_mean_SARSA[i] = mean
        if n <= 1:
            print("Not enough episodes to calculate variance.")
            variance = 0
        else:
            variance = (sum_of_squares - n * mean * mean) / (n - 1.0)
        standard_dev = np.sqrt(variance)
        stored_std_dev_SARSA[i] = standard_dev

        print("run: ", i)
        print("mean: ", mean)
        print("standard deviation: ", standard_dev)

        # unfreeze learning
        agent.agent_message("unfreeze learning")
        # unfreeze exploring
        agent.agent_message("unfreeze exploring")


    """Q-Learning"""
    # used to save the mean values for each run
    stored_mean_Q = np.ones(n_runs)
    stored_std_dev_Q = np.ones(n_runs)

    # alternate between training and evaluating agent for every 10 episodes 
    for i in range(n_runs):

        # Dynamically adjust parameters during training
        # if i % 50 == 0 and i > 0:  # Adjust every 50 runs
        #     new_epsilon = max(0.01, agent2.sarsa_epsilon * 0.90)  # Reduce exploration
        #     new_stepsize = max(0.01, agent2.sarsa_stepsize * 0.99)  # Reduce learning rate
        #     agent2.adjust_parameters(new_epsilon=new_epsilon, new_stepsize=new_stepsize)
        #     print(f"Run {i}: Adjusted sarsa_epsilon to {agent2.sarsa_epsilon}, sarsa_stepsize to {agent2.sarsa_stepsize}")

        # train agent for 10 episodes
        for episode in range(10):
            Exp.setAgentState(S)
            # print("initial state", Exp.state)
            first_obs = Exp.env_start()
            first_action = agent2.agent_start(first_obs)
            reward_observation_terminal = Exp.env_step(first_action)
            for k in range(n_steps):
                action = agent2.agent_step(reward_observation_terminal.r, reward_observation_terminal.o)
                reward_observation_terminal = Exp.env_step(action)
        
        # freeze learning
        agent2.agent_message("freeze learning")
        # freeze exploring
        agent2.agent_message("freeze exploring")
                
        # evaluate agent for n episodes (updated)
        n = 10 
        for episode in range(n):
            sum = 0  # reset per episode
            sum_of_squares = 0  # reset per episode

            Exp.setAgentState(S)
            first_obs = Exp.env_start()
            first_action = agent2.agent_start(first_obs)
            reward_observation_terminal = Exp.env_step(first_action)

            for k in range(n_steps):
                action = agent2.agent_step(reward_observation_terminal.r, reward_observation_terminal.o)
                reward_observation_terminal = Exp.env_step(action)
                sum += reward_observation_terminal.r  # accumulate total reward for the episode
            sum_of_squares += sum**2  # outside the for loop for global variance approach 

        mean = sum / n
        stored_mean_Q[i] = mean
        if n <= 1:
            print("Not enough episodes to calculate variance.")
            variance = 0
        else:
            variance = (sum_of_squares - n * mean * mean) / (n - 1.0)
        standard_dev = np.sqrt(variance)
        stored_std_dev_Q[i] = standard_dev
     
        print("run: ", i)
        print("mean: ", mean)
        print("standard deviation: ", standard_dev)

        # unfreeze learning
        agent2.agent_message("unfreeze learning")
        # unfreeze exploring
        agent2.agent_message("unfreeze exploring")
        
    # added to check ram usage
    # # Display the current memory usage
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print("[ Top 10 Memory Consumers ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    # Calculate learned policy stats
    learned_mean_SARSA = np.mean(stored_mean_SARSA)
    learned_std_SARSA = np.std(stored_mean_SARSA)
    learned_mean_Q = np.mean(stored_mean_Q)
    learned_std_Q = np.std(stored_mean_Q)
    
    # Comparison
    print("\nComparison of Policies:")
    print("Random Policy - Mean Reward:", random_mean)
    print("Random Policy - Std Deviation:", random_std)
    print("Learned Policy - Mean Reward:", learned_mean_SARSA)
    print("Learned Policy - Std Deviation:", learned_std_SARSA)
    print("Learned Policy - Mean Reward:", learned_mean_Q)
    print("Learned Policy - Std Deviation:", learned_std_Q)
    if learned_mean_SARSA > random_mean:
        print("Learned policy outperforms random policy.")
    else:
        print("Random policy performs better or equal to learned policy.")

    # Plot
    plt.plot(stored_mean_SARSA)
    plt.plot(stored_mean_Q,alpha=0.5)
    plt.legend(["SARSA", "Q-Learning"])
    plt.title("Mean")
    plt.figure()
    plt.plot(stored_std_dev_SARSA)
    plt.plot(stored_std_dev_Q,alpha=0.5)
    plt.legend(["SARSA", "Q-Learning"])
    plt.title("Standard Deviation")
    plt.figure()

    # Plot Comparison
    plt.plot(range(len(stored_mean_SARSA)), stored_mean_SARSA, label="Learned Policy")
    plt.axhline(y=random_mean, color="red", linestyle="--", label="Random Policy (Mean Reward)")
    plt.xlabel("Run")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.title("Learned Policy vs Random Policy")

    plt.show()

