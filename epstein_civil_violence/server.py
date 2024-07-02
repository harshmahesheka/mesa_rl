import mesa
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from agent import Citizen, Cop
from model import EpsteinCivilViolence
from ray import tune

COP_COLOR = "#000000"
AGENT_QUIET_COLOR = "#648FFF"
AGENT_REBEL_COLOR = "#FE6100"
JAIL_COLOR = "#808080"
JAIL_SHAPE = "rect"

ray.init(local_mode=True)

def env_creator(_):
    return EpsteinCivilViolence()

class EpsteinCivilViolenceServer(EpsteinCivilViolence):

    def __init__(self, height=20, width=20, citizen_density=0.5, cop_density=0.05, citizen_vision=4, cop_vision=4, legitimacy=0.82, max_jail_term=30):
        super().__init__(height, width, citizen_density, cop_density, citizen_vision, cop_vision, legitimacy, max_jail_term)
        self.running = True
        self.iteration = 0
        tune.register_env("WorldcopModel-v0", env_creator)
        algo = Algorithm.from_checkpoint("policy/")
        self.cop_policy = algo.get_policy("policy_cop")
        self.citizen_policy = algo.get_policy("policy_citizen")

    def step(self):
        # Check if either wolves or cop are extinct
        self.grid_to_matrix()
        action_dict = {}
        for agent in self.schedule.agents:
            if agent.unique_id[0:3] == "cop":
                action_dict[agent.unique_id] = self.cop_policy.compute_single_action(self.observation[agent.unique_id], explore=False)
            else:
                action_dict[agent.unique_id] = self.citizen_policy.compute_single_action(self.observation[agent.unique_id], explore=False)
        self.action_dict = action_dict
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False

def citizen_cop_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "Shape": "circle",
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Filled": "true",
    }

    if type(agent) is Citizen:
        color = (
            AGENT_QUIET_COLOR if agent.condition == "Quiescent" else AGENT_REBEL_COLOR
        )
        color = JAIL_COLOR if agent.jail_sentence else color
        shape = JAIL_SHAPE if agent.jail_sentence else "circle"
        portrayal["Color"] = color
        portrayal["Shape"] = shape
        if shape == "rect":
            portrayal["w"] = 0.9
            portrayal["h"] = 0.9
        else:
            portrayal["r"] = 0.5
            portrayal["Filled"] = "false"
        portrayal["Layer"] = 0

    elif type(agent) is Cop:
        portrayal["Color"] = COP_COLOR
        portrayal["r"] = 0.9
        portrayal["Layer"] = 1

    return portrayal


model_params = {
    "height": 20,
    "width": 20,
    "citizen_density": mesa.visualization.Slider(
        "Initial Agent Density", 0.5, 0.0, 0.9, 0.1
    ),
    "cop_density": mesa.visualization.Slider(
        "Initial Cop Density", 0.05, 0.0, 0.1, 0.01
    ),
    "citizen_vision": mesa.visualization.Slider("Citizen Vision", 4, 1, 10, 1),
    "cop_vision": mesa.visualization.Slider("Cop Vision", 4, 1, 10, 1),
    "legitimacy": mesa.visualization.Slider(
        "Government Legitimacy", 0.82, 0.0, 1, 0.01
    ),
    "max_jail_term": mesa.visualization.Slider("Max Jail Term", 30, 0, 50, 1),
}
canvas_element = mesa.visualization.CanvasGrid(citizen_cop_portrayal, 20, 20, 480, 480)
chart = mesa.visualization.ChartModule(
    [
        {"Label": "Quiescent", "Color": "#648FFF"},
        {"Label": "Active", "Color": "#FE6100"},
        {"Label": "Jailed", "Color": "#808080"},
    ],
    data_collector_name="datacollector",
)
server = mesa.visualization.ModularServer(
    EpsteinCivilViolenceServer,
    [
        canvas_element,
        chart,
    ],
    "Epstein Civil Violence",
    model_params,
)

server.port = 6002
server.launch(open_browser=True)
