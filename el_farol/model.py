import mesa
import numpy as np
import ray
from agents import BarCustomer
import gymnasium as gym
import mesa
import numpy as np
from ray import tune, air
import os
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.utils.typing import  EnvConfigDict

class ElFarolBar(mesa.Model, MultiAgentEnv):
    def __init__(
        self,
        crowd_threshold=60,
        num_strategies=10,
        memory_size=10,
        N=100,
    ):
        super().__init__()
        self.running = True
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.num_strategies = num_strategies

        # Initialize the previous attendance randomly so the agents have a history
        # to work with from the start.
        self.history = np.random.randint(0, N, size=memory_size).tolist()
        self.attendance = self.history[-1]
        for i in range(self.num_agents):
            a = BarCustomer(i, self, memory_size, crowd_threshold)
            self.schedule.add(a)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Customers": "attendance"},
            agent_reporters={"Attendance": "attend"},
        )
        self.max_time = 100
        self.time = 0
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(memory_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action_dict):
        self.datacollector.collect(self)
        self.attendance = 0
        self.action_dict = action_dict
        self.schedule.step()
            
        # We ensure that the length of history is constant
        # after each step.
        self.history.pop(0)
        self.history.append(self.attendance)

        rewards = {}
        for agent in self.schedule.agents:
            if agent.attend:
                strategy_index = agent.unique_id % self.num_strategies
                if self.attendance > agent.crowd_threshold:
                    reward = - (strategy_index%self.num_strategies / self.num_strategies) - 0.5
                else:
                    reward = 1
            else:
                reward = 0
            rewards[agent.unique_id] = reward

        # Get observations
        obs = {a.unique_id: self.history for a in self.schedule.agents}

        # Check if done
        done = {a.unique_id: False for a in self.schedule.agents}
        if self.time > self.max_time:
            done['__all__'] = True
        else:
            done['__all__'] = False
            self.time += 1

        # Prepare info dictionary
        truncated = {a.unique_id: False for a in self.schedule.agents}
        truncated['__all__'] = False

        return obs, rewards, done, truncated, {}
    
    def reset(self, *, seed=None, options=None):
        # Reset your environment here
        super().reset()
        self.time = 0
        # print(self.history)
        obs = {a.unique_id: self.history for a in self.schedule.agents}
        return obs , {}
    
def env_creator(_):
    return ElFarolBar()

def get_config(framework, num_strategies):
    # Define the configuration for the PPO algorithm
    config = (
        PPOConfig()
        .environment("ElFarolBar-v0")
        .framework(framework)
        .multi_agent(
            policies={
                f"policy_{i}": PolicySpec(
                    config=PPOConfig.overrides(framework_str=framework)
                ) for i in range(num_strategies)
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id%num_strategies}",
            policies_to_train=[f"policy_{i}" for i in range(num_strategies)],
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")), num_cpus_for_local_worker=2, num_cpus_per_learner_worker=2)
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    f"policy_{i}": SingleAgentRLModuleSpec() for i in range(num_strategies)
                }
            ),
        )
    )

    config.exploration_config= {'epsilon_timesteps': 10000,
                                    'final_epsilon': 0.0,
                                    'initial_epsilon': 1.0,
                                    'type': 'EpsilonGreedy'}

    return config

def rl_model(args):

    ray.init(local_mode=True)
    # Register the environment
    tune.register_env("ElFarolBar-v0", env_creator)
    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    config = get_config(args.framework, args.num_strategies)
    results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1)),).fit()


    # Uncomment to run trained checkpoint    
    # checkpoint_path = "/root/ray_results/PPO_2024-06-25_08-03-00/PPO_ElFarolBar-v0_5891e_00000_0_2024-06-25_08-03-02/checkpoint_000001/"

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
        stop_iters = 100
        stop_timesteps = 1000000
        stop_reward = 100
        num_strategies = 10

    rl_model(args)
