from pettingzoo.test import parallel_api_test
from environment import WolfSheep
from ray.tune.registry import register_env
from ray import tune, air
import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from agents import Sheep
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import ray

def env_creator():
    return WolfSheep()

if __name__ == "__main__":
    # Uncomment to debug
    # ray.init(local_mode=True)

    env = WolfSheep()
    parallel_api_test(env, num_cycles=1_000_000)

    # Register the environment under an rllib name
    register_env('WorldSheepModel', lambda config: ParallelPettingZooEnv(env_creator()))

    # Define the configuration for the PPO algorithm
    config = (
        PPOConfig()
        .environment("WorldSheepModel")
        .framework("torch")
        .multi_agent(
            policies={
                "policy_sheep": PolicySpec(
                    config=PPOConfig.overrides(framework_str="torch")
                ),
                "policy_wolf": PolicySpec(
                    config=PPOConfig.overrides(framework_str="torch")
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

    stop = {
        "training_iteration": 20,
        "episode_reward_mean": 1000,
        "timesteps_total": 1000000,
    }

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1)),
    ).fit()
