from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray import tune
import os
from model import EpsteinCivilViolence
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

def env_creator(_):
    return EpsteinCivilViolence()  

# ray.init(local_mode=True)
tune.register_env("WorldcopModel-v0", env_creator)

def get_config(framework="torch"):
    # Define the configuration for the PPO algorithm
    config = (
        PPOConfig()
        .environment("WorldcopModel-v0")
        .framework(framework)
        .training(train_batch_size=2000)
        .multi_agent(
            policies={
                "policy_cop": PolicySpec(
                    config=PPOConfig.overrides(framework_str=framework)
                ),
                "policy_citizen": PolicySpec(
                    config=PPOConfig.overrides(framework_str=framework)
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "policy_cop" if agent_id[0:3]=="cop" else "policy_citizen",
            policies_to_train=["policy_cop", "policy_citizen"],
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .learners(num_learners=50)
        .env_runners(num_env_runners=20, num_envs_per_env_runner=1, batch_mode="truncate_episodes", rollout_fragment_length=20)
    )

    return config

algo = get_config().build()

for i in range(5):
    result = algo.train()
    print(pretty_print(result))
with open('/home/iris/mesa/mesa_rl/epstein_civil_violence/results.txt', 'w') as file:
        file.write(pretty_print(result))
checkpoint_dir = algo.save().checkpoint.path
print(f"Checkpoint saved in directory {checkpoint_dir}")