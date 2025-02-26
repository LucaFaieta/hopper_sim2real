# Active Domain Randomization for Reinforcement Learning

## Overview
This repository contains an implementation and analysis of **Active Domain Randomization (ADR)** and **Uniform Domain Randomization (UDR)** for reinforcement learning in robotics. The aim is to bridge the **Reality Gap**, where policies trained in simulations struggle to generalize to real-world applications.

## Problem Statement
Reinforcement Learning (RL) is widely used to train robotic agents. However, training directly in the real world is expensive and risky due to the unpredictable nature of RL exploration. Simulators provide a safer environment, but policies trained in simulation often fail in real-world settings due to differences between the simulated and real domains (**Reality Gap**).

To address this, we analyze **Uniform Domain Randomization (UDR)** and an instance of **Active Domain Randomization (ADR)** to improve policy transferability to real-world scenarios.

## Methodology
### Environment: Mujoco Hopper
We use the **Mujoco Hopper** environment, a 2D bipedal robot consisting of:
- Torso
- Thigh
- Leg
- Foot

The goal is to optimize forward locomotion while maintaining stability.

### Uniform Domain Randomization (UDR)
UDR introduces random variations to environment parameters during training, ensuring that the agent is exposed to a wide range of scenarios. The parameters are sampled uniformly:

$$\xi \sim U(a, b)$$

where \( U(a, b) \) is a uniform distribution over a predefined range \([a, b]\).

### Active Domain Randomization (ADR)
ADR improves upon UDR by **guiding** the randomization process based on the agent's performance in the target environment. Instead of uniform sampling, ADR uses a Gaussian distribution:

$$\xi \sim N(\mu_\xi, \sigma^2_\xi)$$

where \( N(\mu_\xi, \sigma^2_\xi) \) is a normal distribution with mean \( \mu_\xi \) and variance \( \sigma^2_\xi \). These parameters are updated iteratively to maximize rewards in the target environment.

### Training Algorithm
The agent is trained using the **Soft Actor-Critic (SAC)** algorithm with domain randomization. The policy updates iteratively based on:
- Actions taken in the randomized environment
- Rewards obtained from the environment
- Adjustments to parameter distributions (for ADR)

## Experimental Setup
We conducted experiments with two Mujoco Hopper environments:
- **Environment Source:** Default parameters with a reduced torso mass (-1kg)
- **Environment Target:** Default Mujoco Hopper settings

We evaluated the trained policies on **50 test episodes** after **1 million timesteps** of training.

## Results
### Key Findings:
- **UDR:** Best performance was achieved by randomizing floor inclination, enhancing robustness to environmental variations.
- **ADR:** Improved training efficiency and transferability by guiding randomization based on performance.
- **Mixed ADR & UDR:** Combining both methods led to suboptimal performance due to convergence issues.

### Evaluation Metrics:
- **Mean Reward over 50 Episodes**
- **Average Forward Velocity**
- **Energy Consumption**
- **Stability Metrics** (joint angle variance, frequency of falls)

### Hyperparameter Tuning:
We tested various learning rates and discount factors (\(\gamma\)). The best configuration was:
- **Learning Rate:** 0.003
- **Discount Factor (\(\gamma\))**: 0.99

## Conclusion
We demonstrated that **Active Domain Randomization (ADR)** enhances the adaptability of reinforcement learning agents to real-world variations. Future work includes optimizing ADR strategies for broader robotic applications.

## Acknowledgements
Special thanks to the instructors and assistants of the **Robot Learning course at Politecnico di Torino (2023/24)** for their support.

## References
1. Tuomas Haarnoja et al. *Soft Actor-Critic: Off-policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. 2017.
2. Bhairav Mehta et al. *Active Domain Randomization*. 2019.
3. OpenAI et al. *Solving Rubik’s Cube with a Robot Hand*. 2019.
4. Gabriele Tiboni et al. *Dropo: Sim-to-Real Transfer with Offline Domain Randomization*. 2023.
5. Viktor Makoviychuk et al. *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real-World Experience*. 2018.
