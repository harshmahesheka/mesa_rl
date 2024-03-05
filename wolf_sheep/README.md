# Collaborative Survival (Wolf-Sheep Predation Model)
This project demonstrates the use of the RLlib library to implement Multi-Agent Reinforcement Learning (MARL) on the classic Wolf-Sheep predation problem. The environment details can be found on the Mesa project's GitHub repository [here](https://github.com/projectmesa/mesa-examples/tree/main/examples/wolf_sheep). Overall the model showcases an example on how to easily inegrate multi agent libraries like RLlib with mesa and use it's visualziation tools to debug and simulate the policy.


# Key Features:

- RLlib and Multi-Agent Learning: The project leverages the RLlib library to concurrently train two independent PPO (Proximal Policy Optimization) agents, one representing the wolf and the other the sheep.
- Observation Space: Each agent's policy receives a 10x10 grid centered on itself as input. This grid incorporates information about:
Presence of other agents (wolves, sheep, and grass) within the grid
Agent's current energy level
- Server Code Adaptation: The provided server code has been modified to seamlessly integrate outputs from trained models, facilitating visualization for analysis and debugging.
- Training Results:
    - Random agent: Average episode length of 76
    - Trained agent (after 3 hours): Average episode length significantly increased to 215
- Visualization and Debuggability: The model offers a practical example of how to effectively integrate multi-agent RL libraries like RLlib with Mesa's visualization tools, enabling:
Monitoring of simulated agent behavior
Debugging and refining trained policies
Future Work:

TODO: Investigate and address the performance gap observed when interfacing the trained model compared to its performance during training.

<img src="wolf_sheep.gif" width="500" height="400">