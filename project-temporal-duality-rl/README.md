# README

## Project Overview

This project explores the concept of **temporal duality in Reinforcement Learning (RL)** by implementing and comparing three Deep Q-Network (DQN) variants in the challenging Frozen Lake environment:

1. **DQN_Frozen_Lake**: A standard DQN implementation.
2. **ADQN_Frozen_Lake**: A DQN enhanced with an attention mechanism to improve forward planning.
3. **ADQN-BP_Frozen_Lake**: An attention-based DQN that incorporates backward planning for better navigation toward the goal.

The **ADQN-BP model**, which integrates attention and backward planning, represents the core innovation of this project.

---

## File Descriptions

### `DQN_Frozen_Lake.py`
Implements the basic DQN algorithm with an epsilon-greedy policy, standard Q-network training, and replay buffer management.

### `ADQN_Frozen_Lake.py`
Enhances the DQN with an **attention mechanism**, which prioritizes relevant transitions based on their contributions to reward maximization. This mechanism dynamically weights past states and rewards during training.

### `ADQN_BP_Frozen_Lake.py`
Combines attention-based forward planning with **backward planning**, where a backward model generates synthetic transitions from the goal state. These transitions are added to the replay buffer to guide the agent toward optimal trajectories. This file includes:
- A **policy network** for forward planning.
- A **backward model** for generating backward trajectories.

---

## Key Features

### Trained Models
Each script includes pre-trained models saved during the training loop:

1. **`DQN_policy_net.pth`** for `DQN_Frozen_Lake`.
2. **`Attention_DQN_policy_net.pth`** for `ADQN_Frozen_Lake`.
3. **`Backward_DQN_policy_net.pth`** (policy network) and **`Backward_DQN_backward_model.pth`** (backward model) for `ADQN_BP_Frozen_Lake`.

These models are loaded during the testing phase to ensure consistent evaluation without retraining, as RL models often produce varying results across runs.

**Important**: Ensure the correct paths to the trained model files (`DQN_policy_net.pth`, `Backward_DQN_backward_model.pth`) are specified in the scripts before running the testing code. Replace the default paths with the locations of your saved models if they differ.

---

## Usage Instructions

### Prerequisites
Ensure the following dependencies are installed:

- Python (≥3.8)
- PyTorch (≥1.9)
- OpenAI Gym (≥0.21)
- NumPy

Install dependencies using:
```bash
pip install torch gym numpy
```

### Key Parameters
- **Discount Factor (γ)**: 0.99
- **Epsilon Decay**: 0.999 (minimum epsilon: 0.1)
- **Replay Buffer Size**: 10,000
- **Batch Size**: 64
- **History Size**: 15
- **Backward Steps**: 5 (ADQN-BP only)

---

## Notes
1. **Variability in RL**: Due to the stochastic nature of RL and exploration strategies, results can vary across runs. Pre-trained models mitigate this by providing consistent outputs.
2. **Backward Planning**: Only the ADQN-BP model integrates backward planning, used during both training and testing for enhanced performance. 

---