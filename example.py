from epstein_civil_violence.model import EpsteinCivilViolence_RL
from wolf_sheep.server import run_model
from epstein_civil_violence.train import config
from train import train_model

# env = EpsteinCivilViolence_RL()
# observation, info = env.reset(seed=42)
# for _ in range(10):
#     action_dict = {}
#     for agent in env.schedule.agents:
#         action_dict[agent.unique_id] = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action_dict)

#     if terminated or truncated:
#         observation, info = env.reset()

# # Training a model
# train_model(config, num_iterations=1, result_path='results.txt', checkpoint_dir='checkpoints')

# Running the model
server = run_model(model_path='/home/iris/mesa/mesa_rl/model/wolf_sheep')
server.port = 6005
server.launch(open_browser=True)