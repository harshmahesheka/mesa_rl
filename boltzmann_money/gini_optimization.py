# This code implements a multi-agent model called MoneyModel using the Mesa library. 
# The model simulates the distribution of wealth among agents in a grid environment.
# Each agent has a randomly assigned wealth and can move to neighboring cells.
# Agents can also give money to other agents in the same cell if they have greater wealth.
# The model is trained by a scientist who believes in an equal society and wants to minimize the Gini coefficient, which measures wealth inequality.
# The model is trained using the Proximal Policy Optimization (PPO) algorithm from the stable-baselines3 library.
# The trained model is saved as "ppo_money_model".

# Import necessary libraries
import numpy as np
import gymnasium
import argparse
import random
import seaborn as sns
import matplotlib.pyplot as plt
# Import mesa
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# Import stable-baselines3 for RL
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

NUM_AGENTS = 10  # Number of agents

# Define the agent class
class MoneyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Randomly assign wealth to each agent
        self.wealth = np.random.randint(1, NUM_AGENTS)

    def move(self, action):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        action = int(action)  # Convert action to integer
        
        # Uncomment this to test random baselines
        # action = np.random.randint(0, 5)
    
        # Move the agent based on the action
        if action == 0:
            new_position = (self.pos[0] + 1, self.pos[1])
        elif action == 1:
            new_position = (self.pos[0] - 1, self.pos[1])
        elif action == 2:
            new_position = (self.pos[0], self.pos[1] - 1)
        elif action == 3:
            new_position = (self.pos[0], self.pos[1] + 1)
        elif action == 4:
            new_position = self.pos
        # if position is not valid, do not move
        if new_position in possible_steps:
            self.model.grid.move_agent(self, new_position)

    def take_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            # Select a random agent from the same cell
            other_agent = random.choice(cellmates)
            # If other agent has more wealth, take 1 from them
            if other_agent.wealth > self.wealth:
                other_agent.wealth -= 1
                self.wealth += 1

    def step(self, action):
        self.move(action)
        self.take_money()

# Define the model class
class MoneyModel(Model, gymnasium.Env):
    def __init__(self, N, width, height):
        super().__init__()
        self.N = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Gini": self.compute_gini}, agent_reporters={"Wealth": "wealth"}
        )
        self.time = 0
        # Create agents
        for i in range(self.N):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.observation_space = gymnasium.spaces.Box(low=0, high=10 * N, shape=(N, 3))
        self.action_space = gymnasium.spaces.MultiDiscrete([5] * N)
        self.timestep = 0
        self.prev_gini = None
        self.datacollector.collect(self)
        # Visualize the environment for debugging using mesa's built-in visualization
        self.is_visualize = False

    def step(self, action):
        # Apply actions
        for i, a in enumerate(self.schedule.agents):
            a.step(action[i])

        # Collect data
        self.datacollector.collect(self)

        # Calculate reward
        if self.prev_gini is None:
            self.prev_gini = self.compute_gini()
        # Reward is the change in Gini coefficient with negative reward for passing time
        new_gini = self.compute_gini()
        if new_gini < self.prev_gini:
            reward = (self.prev_gini - new_gini) * 20
        else: 
            reward = -0.05
        self.prev_gini = new_gini

        self.timestep += 1
        # Get observations which is the wealth of each agent and their position
        obs = self._get_obs()
        # Check if done and give terminal reward
        if self.timestep > 5 * self.N:
            done = True
            reward = -1
        elif new_gini < 0.1:
            done = True
            reward = 50/self.timestep
        else:
            done = False

        # Prepare empty info dictionary
        info = {}
        truncated = False

        self.time += 1
        return obs, reward, done, truncated, info

    def visualize(self):
        # Current Implementation pauses the program and user needs to close the visualization window to continue
        # TODO: Implement a non-blocking visualization using multiprocessing
        gini = self.datacollector.get_model_vars_dataframe()
        # Plot the Gini coefficient over time
        g = sns.lineplot(data=gini)
        g.set(title="Gini Coefficient over Time", ylabel="Gini Coefficient")
        plt.show()

    def reset(self, *, seed=None, options=None):
        # Visualize the environment for debugging
        if self.is_visualize:
            self.visualize()
        # Reset your environment here
        super().reset()
        self.timestep = 0
        for i, a in enumerate(self.schedule.agents):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.move_agent(a, (x, y))
            a.wealth = np.random.randint(1, self.N)
        self.prev_gini = None
        return self._get_obs(), {}

    def _get_obs(self):
        # The observation is the wealth of each agent and their position
        obs = []
        for a in self.schedule.agents:
            obs.append([a.wealth] + list(a.pos))
        return np.array(obs)

    def compute_gini(self):
        agent_wealths = [agent.wealth for agent in self.schedule.agents]
        x = sorted(agent_wealths)
        N = self.N
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
        return (1 + (1 / N) - 2 * B)

def rl_model(args):
    # Create the environment
    env = MoneyModel(N=NUM_AGENTS, width=NUM_AGENTS, height=NUM_AGENTS)
    eval_env = MoneyModel(N=NUM_AGENTS, width=NUM_AGENTS, height=NUM_AGENTS)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000)
    # Define the PPO model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model = PPO.load("mesa_rl/model/boltzmann_money.zip", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=args.stop_timesteps, callback=[eval_callback])

    # Save the model
    model.save("ppo_money_model")
    

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop-timesteps", type=int, default=NUM_AGENTS * 10, help="Number of timesteps to train."
    )
    args = parser.parse_args()
    rl_model(args)