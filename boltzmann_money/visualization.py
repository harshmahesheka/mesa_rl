from gini_optimization import MoneyModel
from stable_baselines3 import PPO
import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

# Modify the MoneyModel class to take actions from the RL model
class MoneyModelRL(MoneyModel):
    def __init__(self, N, width, height):
        super().__init__(N, width, height)
        self.rl_model = model = PPO.load("mesa_rl/model/boltzmann_money.zip")

    def step(self):
        # Collect data
        self.datacollector.collect(self)

        # Calculate reward
        if self.prev_gini is None:
            self.prev_gini = self.compute_gini()
        new_gini = self.compute_gini()
        if new_gini < self.prev_gini:
            reward = (self.prev_gini - new_gini) * 20
        else: 
            reward = -0.05
        self.prev_gini = new_gini

        # Get observations which is the wealth of each agent and their position
        obs = self._get_obs()
        
        action, _states = self.rl_model.predict(obs)
        for i, a in enumerate(self.schedule.agents):
            a.step(action[i])

# Define the agent portrayal with different colors for different wealth levels
def agent_portrayal(agent):
    if agent.wealth > 10:
        color = "purple"
    elif agent.wealth > 7:
        color = "red"
    elif agent.wealth > 5:
        color = "orange"
    elif agent.wealth > 3:
        color = "yellow"
    else:
        color = "blue"

    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": color,
                 "r": 0.5}
    return portrayal

if __name__ == "__main__":

    # Define a grid visualization
    grid = mesa.visualization.CanvasGrid(agent_portrayal, 10, 10, 500, 500)

    # Define a chart visualization
    chart = ChartModule([{"Label": "Gini", "Color": "Black"}], 
                        data_collector_name='datacollector')

    # Create a modular server
    server = ModularServer(MoneyModelRL,
                        [grid, chart],
                        "Money Model",
                        {"N":10, "width":10, "height":10})
    server.port = 8521 # The default
    server.launch()
