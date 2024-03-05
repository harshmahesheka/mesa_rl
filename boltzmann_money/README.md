# Balancing Wealth Inequality
This folder showcases how to solve the Boltzmann wealth model with Proximal Policy Optimization (PPO) from Stable Baselines.

## Key features:

- Boltzmann Wealth Model: Agents with varying wealth navigate a grid, aiming to minimize inequality measured by the Gini coefficient.
- PPO Training: A PPO agent is trained to achieve this goal, receiving sparse rewards based on Gini coefficient improvement and a large terminal reward for achieving low inequality.
- Mesa Data Collection and Visualization: The Mesa data collector tool tracks Gini values during training, allowing for real-time visualization.
- Visualization Script: Visualize the trained agent's behavior with Mesa's visualization tools, presenting agent movement and Gini values within the grid. 
- Pre-trained Model: A model trained for 2 hours on 3 CPU cores is included, achieving equality (Gini < 0.1) in roughly 16-18 steps (significantly faster than a random agent which takes over 150 steps).

## Model Behaviour
The adgent as seen below learns to move towards a corner of the grid. These brings all the agents together allowing exchange of money between them resulting in reward maximization. This is not optimal behaviour but still a lot better than random agent.


<img src="ppo_agent.gif" width="500" height="400">