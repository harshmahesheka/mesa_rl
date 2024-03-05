import mesa
import numpy as np
import gymnasium as gym
from ray import tune, air
import os
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from agents import GrassPatch, Sheep, Wolf
from scheduler import RandomActivationByTypeFiltered
import ray

class WolfSheep(mesa.Model, MultiAgentEnv):
    """
    Wolf-Sheep Predation Model
    """

    verbose = False  # Print-monitoring

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=25,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=True,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.
        """
        super().__init__()
        # Set parameters
        self.width = width
        self.height = height
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.sheep_gain_from_food = sheep_gain_from_food

        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: m.schedule.get_type_count(Wolf),
                "Sheep": lambda m: m.schedule.get_type_count(Sheep),
                "Grass": lambda m: m.schedule.get_type_count(
                    GrassPatch, lambda x: x.fully_grown
                ),
            }
        )

        self.create_intial_agents()

        self.running = True
        self.datacollector.collect(self)
        self.observation_space = gym.spaces.Dict({
            'grid': gym.spaces.Box(low=0, high=1, shape=(10, 10, 4), dtype=bool),  # 3 for sheep, wolf, grass
            'energy': gym.spaces.Box(low=-1, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.action_space =  gym.spaces.Discrete(4)
        self.time = 0

    def create_intial_agents(self):
        # Create sheep:
        for i in range(self.initial_sheep):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.sheep_gain_from_food)
            sheep = Sheep(self.next_id(), (x, y), self, energy)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create wolves
        for i in range(self.initial_wolves):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.wolf_gain_from_food)
            wolf = Wolf(self.next_id(), (x, y), self, energy)
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        # Create grass patches
        if self.grass:
            for agent, (x, y) in self.grid.coord_iter():
                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

    def step(self, action_dict):
        # Check if either wolves or sheep are extinct
        if self.schedule.get_type_count(Wolf) == 0 or self.schedule.get_type_count(Sheep) == 0:
            for agent in self.schedule.agents:
                if isinstance(agent, (Sheep, Wolf)):
                    agent.living = False

        self.datacollector.collect(self)

        rewards = {a.unique_id: 0 for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        # Check for rewards and execute actions
        for agent in self.schedule.agents:
            if isinstance(agent, (Sheep, Wolf)):
                agent.step(action_dict[agent.unique_id])
                if isinstance(agent, Sheep):
                    rewards[agent.unique_id] += min(4, agent.energy - 4)
                else:
                    rewards[agent.unique_id] += min(4, agent.energy/5 - 4)
            else:
                agent.step()

        for agent in self.schedule.agents:
            if agent.unique_id not in rewards:
                rewards[agent.unique_id] = 0
            elif not agent.living:
                agent.done = True
                rewards[agent.unique_id] = min(0, -(25 - agent.time))

        # Get observations
        obs = {a.unique_id: self.get_observation(a) for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}
        # Check if done
        done = {a.unique_id: a.done for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}
        done['__all__'] = np.all(list(done.values()))

        if self.time > 500:
            done['__all__'] = True

        # Prepare info dictionary
        truncated = {a.unique_id: False for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}
        truncated['__all__'] = np.all(list(truncated.values()))

        self.remove_dead_agents()

        return obs, rewards, done, truncated, {}

    def get_subgrid(self, grid, agent, default_value):
        # Identify the agent's position
        agent_position = agent.pos

        # Determine the size of the sub-grid
        sub_grid_size = (10, 10)

        # Calculate the top-left and bottom-right coordinates of the sub-grid
        top_left = (int(agent_position[0] - sub_grid_size[0] / 2), int(agent_position[1] - sub_grid_size[1] / 2))
        bottom_right = (top_left[0] + sub_grid_size[0], top_left[1] + sub_grid_size[1])

        # If the agent is near the edge of the grid, adjust the coordinates and note how much padding is needed
        padding = [0, 0, 0, 0]  # top, bottom, left, right
        if top_left[0] < 0:
            padding[0] = -top_left[0]
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            padding[2] = -top_left[1]
            top_left = (top_left[0], 0)
        if bottom_right[0] > self.grid.width:
            padding[1] = bottom_right[0] - self.grid.width
            bottom_right = (self.grid.width, bottom_right[1])
        if bottom_right[1] > self.grid.height:
            padding[3] = bottom_right[1] - self.grid.height
            bottom_right = (bottom_right[0], self.grid.height)

        # Extract the sub-grid
        sub_grid = grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # If padding is needed, pad the sub-grid with the default value
        if any(padding):
            sub_grid = np.pad(sub_grid, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), 'constant', constant_values=default_value)

        for i in range(len(sub_grid)):
            for j in range(len(sub_grid[i])):
                if all(sub_grid[i][j] == [True, True, True,True]):
                    sub_grid[i][j] = [False, False, False, True]

        return sub_grid

    def remove_dead_agents(self):
        for agent in list(self.schedule.agents):
            if isinstance(agent, (Sheep, Wolf)) and not agent.living:
                self.grid.remove_agent(agent)
                self.schedule.remove(agent)

    def reset(self, *, seed=None, options=None):
        # Reset your environment here
        super().reset()
        self.time = 0
        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.create_intial_agents()
        obs = {a.unique_id: self.get_observation(a) for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}
        return obs , {}

    def get_observation(self, current_agent):
        """
        Get the observation, which is a dictionary.
        The 'grid' key contains a 3D array with information about the agents at each grid location.
        The 'current_agent' key contains a dictionary with the current agent's position and energy.
        """
        grid = np.zeros((self.width, self.height, 4), dtype=bool)  # 3 for sheep, wolf, grass

        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get_cell_list_contents([(x, y)])

                for agent in cell:
                    if isinstance(agent, Sheep):
                        grid[x, y, 0] = True
                    elif isinstance(agent, Wolf):
                        grid[x, y, 1] = True
                    elif isinstance(agent, GrassPatch):
                        grid[x, y, 2] = agent.fully_grown

        sub_grid = self.get_subgrid(grid, current_agent, False)

        observation = {
            'grid': sub_grid,
            'energy': np.array([current_agent.energy])
        }

        return observation

def env_creator(_):
    return WolfSheep()  # Assuming there are 10 sheep

def get_config(framework):
    # Define the configuration for the PPO algorithm
    config = (
        PPOConfig()
        .environment("WorldSheepModel-v0")
        .framework(framework)
        .multi_agent(
            policies={
                "policy_sheep": PolicySpec(
                    config=PPOConfig.overrides(framework_str=framework)
                ),
                "policy_wolf": PolicySpec(
                    config=PPOConfig.overrides(framework_str=framework)
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "policy_sheep" if isinstance(agent_id, Sheep) else "policy_wolf",
            policies_to_train=["policy_sheep", "policy_wolf"],
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")), num_cpus_for_local_worker=2, num_cpus_per_learner_worker=2)
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "policy_sheep": SingleAgentRLModuleSpec(),
                    "policy_wolf": SingleAgentRLModuleSpec()
                }
            ),
        )
    )

    return config

def rl_model(args):

    # ray.init(local_mode=True)
    # Register the environment
    tune.register_env("WorldSheepModel-v0", env_creator)

    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    config = get_config(args.framework)
    results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1)),).fit()


    # Uncomment to run trained checkpoint    
    # checkpoint_path = "../model/wolf_sheep_policy/"

    # trainer = tune.run(
    #     "PPO",
    #     config=config,
    #     stop=stop,
    #     restore=checkpoint_path,  # This is where you specify the checkpoint path
    #     verbose=1
    # )

if __name__ == "__main__":
    class args():
        framework = "torch"
        stop_iters = 60
        stop_timesteps = 10000000
        stop_reward = 10000

    rl_model(args)
