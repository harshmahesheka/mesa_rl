import mesa
import numpy as np
import gymnasium as gym
import functools
from pettingzoo import ParallelEnv
from agents import Wolf, Sheep, GrassPatch
from utils_function import get_observation, remove_dead_agents, create_initial_agents
from scheduler import RandomActivationByTypeFiltered

class WolfSheep(mesa.Model, ParallelEnv):
    """
    Wolf-Sheep Predation Model
    """

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=25,
        sheep_reproduce=0.0,
        wolf_reproduce=0.0,
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

        create_initial_agents(self)

        self.running = True
        self.datacollector.collect(self)
        self.time = 0
        self.agents = [a.unique_id for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))]
        self.possible_agents = self.agents
        self.observation_spaces =  {a: self.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.action_space(a) for a in self.possible_agents}

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
            if isinstance(agent, (Sheep, Wolf)):
                if agent.unique_id not in rewards:
                    rewards[agent.unique_id] = 0
                if not agent.living:
                    agent.done = True
                    rewards[agent.unique_id] = min(0, -(25 - agent.time))

        # Get observations
        obs = {a.unique_id: get_observation(self, a) for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        # Check if done
        done = {a.unique_id: a.done for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        self.time += 1

        if self.time > 500:
            done = {a.unique_id: True for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        # Prepare info dictionary
        truncated = {a.unique_id: False for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        infos = {a.unique_id: {}  for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        remove_dead_agents(self)

        return obs, rewards, done, truncated, infos

    def reset(self, seed=None, options=None):
        # Reset your environment here
        self.time = 0
        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.current_id = 0
        create_initial_agents(self)
        self.agents = [a.unique_id for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))]
        obs = {a.unique_id: get_observation(self, a) for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}
        infos = {a: {} for a in self.agents}
        return obs , infos


    def render(self):
        # Render the environment to the screen
        pass


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return  gym.spaces.Dict({'grid': gym.spaces.Box(low=0, high=1, shape=(10, 10, 4), dtype=bool),
                                 'energy': gym.spaces.Box(low=-1, high=np.inf, shape=(1,), dtype=np.float32)
                                })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(4)
