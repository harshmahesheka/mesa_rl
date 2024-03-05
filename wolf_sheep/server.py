import mesa
from agents import GrassPatch, Sheep, Wolf
from model import WolfSheep
import numpy as np
from ray import tune
import mesa
from agents import GrassPatch, Sheep, Wolf
import numpy as np
from model import get_config
from ray import tune
import os
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec


def env_creator(_):
    return WolfSheep()  # Assuming there are 10 sheep

class WolfSheepRL(WolfSheep):
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
        super().__init__(
            width,
            height,
            initial_sheep,
            initial_wolves,
            sheep_reproduce,
            wolf_reproduce,
            wolf_gain_from_food,
            grass,
            grass_regrowth_time,
            sheep_gain_from_food,
        )

        # Register the environment
        tune.register_env("WorldSheepModel-v0", env_creator)

        config = get_config("torch")
        self.rl_model = config.build()

        # Load the checkpoint
        checkpoint_path = "../model/wolf_sheep_policy/"
        self.rl_model.restore(checkpoint_path)
        self.wolf_policy = self.rl_model.get_policy("policy_wolf")
        self.sheep_policy = self.rl_model.get_policy("policy_sheep")
    
    def step(self):

        if self.schedule.get_type_count(Wolf) == 0 or self.schedule.get_type_count(Sheep) == 0:
            for agent in self.schedule.agents:
                if isinstance(agent, (Sheep, Wolf)) :
                    agent.living = False

        self.datacollector.collect(self)
        # Apply actions
        rewards = {a.unique_id: 0 for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))}

        # Check for rewards
        for agent in self.schedule.agents:
            if isinstance(agent, (Sheep, Wolf)) :
                # agent.step(action_dict[agent.unique_id])
                if isinstance(agent, Sheep):
                    rewards[agent.unique_id] += min(4, agent.energy - 4)
                else:
                    rewards[agent.unique_id] += min(4, agent.energy/5 - 4)

        for agent in self.schedule.agents:
            if agent.unique_id not in rewards:
                rewards[agent.unique_id] = 0
            elif not agent.living:
                agent.done = True
                rewards[agent.unique_id] = min( 0, -(25 - agent.time))

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

        self.agents = {agent.unique_id: agent for agent in self.schedule.agents}
        # Compute actions using the appropriate policy for each agent
        actions = {}
        for agent_id, observation in obs.items():
            agent = self.agents[agent_id]
            if isinstance(agent, Sheep):
                actions[agent_id] = self.sheep_policy.compute_single_action(observation)[0]
            else:
                actions[agent_id] = self.wolf_policy.compute_single_action(observation)[0]

        # Apply actions
        for agent_id, action in actions.items():
            self.agents[agent_id].step(action)
        for agent in self.schedule.agents:
            if isinstance(agent, (GrassPatch)):
                agent.step()

        self.remove_dead_agents()



def wolf_sheep_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Sheep:
        portrayal["Shape"] = "mobile_rl/resources/sheep.png"
        # https://icons8.com/web-app/433/sheep
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1
        portrayal["text"] = round(agent.energy, 1)

    elif type(agent) is Wolf:
        portrayal["Shape"] = "mobile_rl/resources/wolf.png"
        # https://icons8.com/web-app/36821/German-Shepherd
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 2
        portrayal["text"] = round(agent.energy, 1)
        portrayal["text_color"] = "White"

    elif type(agent) is GrassPatch:
        if agent.fully_grown:
            portrayal["Color"] = ["#00FF00", "#00CC00", "#009900"]
        else:
            portrayal["Color"] = ["#84e184", "#adebad", "#d6f5d6"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal

# ray.init(local_mode=True)
canvas_element = mesa.visualization.CanvasGrid(wolf_sheep_portrayal, 20, 20, 500, 500)
chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "Wolves", "Color": "#AA0000"},
        {"Label": "Sheep", "Color": "#666666"},
        {"Label": "Grass", "Color": "#00AA00"},
    ]
)

model_params = {
    # The following line is an example to showcase StaticText.
    "title": mesa.visualization.StaticText("Parameters:"),
    "grass": mesa.visualization.Checkbox("Grass Enabled", True),
    "grass_regrowth_time": mesa.visualization.Slider("Grass Regrowth Time", 20, 1, 50),
    "initial_sheep": mesa.visualization.Slider(
        "Initial Sheep Population", 100, 10, 300
    ),
    "sheep_reproduce": mesa.visualization.Slider(
        "Sheep Reproduction Rate", 0.04, 0.01, 1.0, 0.01
    ),
    "initial_wolves": mesa.visualization.Slider("Initial Wolf Population", 25, 10, 300),
    "wolf_reproduce": mesa.visualization.Slider(
        "Wolf Reproduction Rate",
        0.05,
        0.01,
        1.0,
        0.01,
        description="The rate at which wolf agents reproduce.",
    ),
    "wolf_gain_from_food": mesa.visualization.Slider(
        "Wolf Gain From Food Rate", 20, 1, 50
    ),
    "sheep_gain_from_food": mesa.visualization.Slider("Sheep Gain From Food", 4, 1, 10),
}

server = mesa.visualization.ModularServer(
    WolfSheepRL, [canvas_element, chart_element], "Wolf Sheep Predation", model_params
)
server.port = 6002
server.launch(open_browser=True)
