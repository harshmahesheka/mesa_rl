import mesa
from agent import Citizen, Cop
import mesa
import numpy as np
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv


class EpsteinCivilViolence(mesa.Model, MultiAgentEnv):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full
    Attributes:
        height: grid height
        width: grid width
        citizen_density: approximate % of cells occupied by citizens.
        cop_density: approximate % of cells occupied by cops.
        citizen_vision: number of cells in each direction (N, S, E and W) that
            citizen can inspect
        cop_vision: number of cells in each direction (N, S, E and W) that cop
            can inspect
        legitimacy:  (L) citizens' perception of regime legitimacy, equal
            across all citizens
        max_jail_term: (J_max)
            > threshold, citizen rebels
        arrest_prob_constant: set to ensure agents make plausible arrest
            probability estimates
        movement: binary, whether agents try to move at step end
        max_iters: model may not have a natural stopping point, so we set a
            max.
    """

    def __init__(
        self,
        width=20,
        height=20,
        citizen_density=0.5,
        cop_density=0.05,
        citizen_vision=4,
        cop_vision=4,
        legitimacy=0.8,
        max_jail_term=30,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=200,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.citizen_density = citizen_density
        self.cop_density = cop_density
        self.citizen_vision = citizen_vision
        self.cop_vision = cop_vision
        self.legitimacy = legitimacy
        self.max_jail_term = max_jail_term
        self.arrest_prob_constant = arrest_prob_constant
        self.movement = movement
        self.max_iters = max_iters
        self.iteration = 0
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        model_reporters = {
            "Quiescent": lambda m: self.count_type_citizens(m, "Quiescent"),
            "Active": lambda m: self.count_type_citizens(m, "Active"),
            "Jailed": self.count_jailed,
            "Cops": self.count_cops,
        }
        agent_reporters = {
            "x": lambda a: a.pos[0],
            "y": lambda a: a.pos[1],
            "breed": lambda a: a.breed,
            "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
            "condition": lambda a: getattr(a, "condition", None),
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )
        self.create_intial_agents()
        self.running = True
        self.datacollector.collect(self)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(80, ), dtype=np.float32)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(9), gym.spaces.Discrete(4)))
        self.prev_active_citizens = 0
        self.observation = {}


    def create_intial_agents(self):
        unique_id = 0
        if self.cop_density + self.citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")
        for contents, (x, y) in self.grid.coord_iter():
            if self.random.random() < self.cop_density:
                unique_id_str = f"cop_{unique_id}"
                cop = Cop(unique_id_str, self, (x, y), vision=self.cop_vision)
                unique_id += 1
                self.grid[x][y] = cop
                self.schedule.add(cop)
            elif self.random.random() < (self.cop_density + self.citizen_density):
                unique_id_str = f"citizen_{unique_id}"
                citizen = Citizen(
                    unique_id_str,
                    self,
                    (x, y),
                    hardship=self.random.random(),
                    regime_legitimacy=self.legitimacy,
                    risk_aversion=self.random.random(),
                    vision=self.citizen_vision,
                )
                unique_id += 1
                self.grid[x][y] = citizen
                self.schedule.add(citizen)

    def grid_to_matrix(self):
        self.obs_grid = []
        for i in self.grid._grid:
            row = []
            for j in i:
                if j is None:
                    row.append(0)
                elif isinstance(j, Citizen):
                    if j.condition == "Quiescent":
                        row.append(1)
                    elif j.condition == "Active":
                        row.append(2)
                    else:
                        row.append(3)
                else:
                    row.append(4)
            self.obs_grid.append(row)

    def step(self, action_dict):
        # Check if either wolves or cop are extinct
        self.grid_to_matrix()
        self.action_dict = action_dict
        self.schedule.step()
        self.datacollector.collect(self)

        rewards = {}

        active_citizens = self.count_type_citizens(self, "Active")
        jailed_citizens = self.count_jailed(self)
        quiescent_citizens = self.count_type_citizens(self, "Quiescent")
        cops = self.count_cops(self)
        total_citizens = active_citizens + jailed_citizens + quiescent_citizens

        # Check for rewards and execute actions
        for agent in self.schedule.agents:
            if isinstance(agent, Cop):
                rewards[agent.unique_id] = (active_citizens - self.prev_active_citizens) / total_citizens
            else:
                if agent.jail_sentence > 0:
                    rewards[agent.unique_id] = -agent.risk_aversion
                else:
                    if agent.condition == "Quiescent":
                        rewards[agent.unique_id] = -agent.grievance

        done = {a.unique_id: False for a in self.schedule.agents}

        # Prepare info dictionary
        truncated = {a.unique_id: False for a in self.schedule.agents}
        truncated['__all__'] = np.all(list(truncated.values()))
        if self.schedule.time > self.max_iters:
            done['__all__'] = True
        else:
            done['__all__'] = False

        return self.observation, rewards, done, truncated, {}
    
    def reset(self, *, seed=None, options=None):
        # Reset your environment here
        super().reset()
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(self.width, self.height, torus=True)
        self.create_intial_agents()
        self.grid_to_matrix()
        self.action_dict = {a.unique_id: (0, 0) for a in self.schedule.agents}
        self.schedule.step()
        return self.observation , {}
    
    @staticmethod
    def count_type_citizens(model, condition, exclude_jailed=True):
        """
        Helper method to count agents by Quiescent/Active.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "cop":
                continue
            if exclude_jailed and agent.jail_sentence > 0:
                continue
            if agent.condition == condition:
                count += 1
        return count

    @staticmethod
    def count_jailed(model):
        """
        Helper method to count jailed agents.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "citizen" and agent.jail_sentence > 0:
                count += 1
        return count

    @staticmethod
    def count_cops(model):
        """
        Helper method to count jailed agents.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "cop":
                count += 1
        return count

