# Spider Solitaire Reinforcement Learning

This project implements various reinforcement learning algorithms to play Spider Solitaire, a popular card game. The implementation includes DQN, A2C, and PPO algorithms with both full and simplified versions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Files](#environment-files)
- [Training Scripts](#training-scripts)
- [Comparison Scripts](#comparison-scripts)
- [Utility Scripts](#utility-scripts)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Call Chain Diagram](#call-chain-diagram)

## Overview

The project provides a complete reinforcement learning framework for Spider Solitaire with:
- Custom Gymnasium environments with action masking
- Three RL algorithms: DQN, A2C, and PPO
- Both full and simplified network architectures
- Comprehensive comparison and visualization tools
- Fixed environments addressing common RL training issues

## Project Structure

```
SpiderSolitair_RL_Python/
├── environments/
│   ├── spider_solitaire_env_fixed.py      # Base environment with fixes
│   └── spider_solitaire_masked_env_fixed.py # Environment with action masking
├── training/
│   ├── train_dqn.py                       # Full DQN implementation
│   ├── train_dqn_simple.py                # Simplified DQN
│   ├── train_a2c.py                       # Full A2C implementation
│   ├── train_a2c_simple.py                # Simplified A2C
│   ├── train_ppo.py                       # Full PPO implementation
│   └── train_ppo_simple.py                # Simplified PPO
├── comparison/
│   ├── quick_compare_fixed.py             # Quick comparison of full algorithms
│   ├── quick_compare_simple_fixed.py      # Quick comparison of simplified algorithms
│   └── compare_algorithms_fixed.py        # Comprehensive algorithm comparison
├── utilities/
│   ├── example_usage.py                   # Demo of environment usage
│   ├── visualize_training.py              # Training visualization tools
│   ├── demo_comparison.py                 # Simulated comparison demo
│   └── environment_usage_summary.py       # Environment usage documentation
└── results/                               # Training results and plots
```

## Environment Files

### spider_solitaire_env_fixed.py
The base Spider Solitaire environment with bug fixes:
- **Features**: 
  - 2-suit Spider Solitaire (104 cards)
  - Maximum steps per episode (truncation)
  - Proper handling of stuck games
  - Reward structure: move (-1), invalid move (-10), reveal card (+5), complete sequence (+100), win (+1000)
- **Key Classes**: `SpiderSolitaireEnvFixed`

### spider_solitaire_masked_env_fixed.py
Environment with action masking for valid moves:
- **Features**: 
  - Extends base environment with action masking
  - Computes valid actions dynamically
  - Provides action mask in observation
- **Key Classes**: `MaskedSpiderSolitaireEnvFixed`, `ActionMasker`
- **Important Fix**: ActionMasker inherits from `gym.Wrapper` (not `gym.ActionWrapper`) to avoid action conversion issues

## Training Scripts

### Full Network Implementations

#### train_dqn.py
Deep Q-Network with experience replay:
```bash
python train_dqn.py
```
- **Network**: 3 Conv layers (32→64→128) + 3 FC layers (512→256→actions)
- **Features**: Experience replay buffer, target network, epsilon-greedy exploration
- **Output**: Saves model to `dqn_spider_final.pt`

#### train_a2c.py
Advantage Actor-Critic with parallel environments:
```bash
python train_a2c.py
```
- **Network**: 3 Conv layers (32→64→128) + shared FC (512→256) + separate actor/critic heads
- **Features**: GAE for advantage estimation, parallel environment collection
- **Output**: Saves model to `a2c_spider_final.pt`

#### train_ppo.py
Proximal Policy Optimization:
```bash
python train_ppo.py
```
- **Network**: 3 Conv layers (32→64→64) + policy and value networks (256→256)
- **Features**: Clipped objective, multiple epochs per batch
- **Output**: Saves model to `models/spider_solitaire_*/final_model`
- **Note**: Doesn't use action masking, learns from reward signals

### Simplified Network Implementations

These use lighter networks for faster training:

#### train_dqn_simple.py
```bash
python train_dqn_simple.py
```
- **Network**: 2 Conv layers (16→32) + 1 FC layer (128)
- **Training**: 10,000 episodes by default

#### train_a2c_simple.py
```bash
python train_a2c_simple.py
```
- **Network**: 2 Conv layers (16→32) + 1 FC layer (128)
- **Training**: 1,000,000 timesteps by default

#### train_ppo_simple.py
```bash
python train_ppo_simple.py
```
- **Network**: 2 Conv layers (16→32) + simplified policy/value networks
- **Training**: 1,000,000 timesteps by default

## Comparison Scripts

### quick_compare_fixed.py
Quick comparison of full algorithms with minimal training:
```bash
python quick_compare_fixed.py --episodes 100 --max-steps 500
```
- **Purpose**: Fast testing of all algorithms
- **Output**: Comparison plots and report in `results/quick_comparison_fixed/`

### quick_compare_simple_fixed.py
Quick comparison of simplified algorithms:
```bash
python quick_compare_simple_fixed.py --episodes 100 --max-steps 500
```
- **Purpose**: Test simplified implementations
- **Output**: Comparison plots and report in `results/simple_comparison_fixed/`

### compare_algorithms_fixed.py
Comprehensive comparison with longer training:
```bash
python compare_algorithms_fixed.py --timesteps 100000 --max-steps 500 --algorithms dqn a2c ppo
```
- **Purpose**: Detailed performance analysis
- **Features**: Training curves, win rates, efficiency metrics
- **Output**: Detailed plots and report in `results/algorithm_comparison_fixed/`

### demo_comparison.py
Simulated comparison for demonstration:
```bash
python demo_comparison.py
```
- **Purpose**: Generate example comparison without actual training
- **Output**: Demo plots and report in `results/demo_comparison/`

## Utility Scripts

### example_usage.py
Demonstrates environment usage:
```bash
python example_usage.py test   # Run environment tests
python example_usage.py play   # Play interactive game
python example_usage.py        # Watch random game
```

### visualize_training.py
Visualize training metrics from TensorBoard logs:
```bash
python visualize_training.py --log-dir ./logs/spider_solitaire_* --model-path models/final_model
```

### environment_usage_summary.py
Check environment usage across files:
```bash
python environment_usage_summary.py
```

## Installation

1. Install Python 3.8+
2. Install dependencies:
```bash
pip install gymnasium numpy torch matplotlib
pip install stable-baselines3  # For PPO support
```

## Quick Start

1. **Test the environment**:
```bash
python example_usage.py test
```

2. **Run a quick comparison**:
```bash
python quick_compare_simple_fixed.py --episodes 10 --max-steps 500
```

3. **Train a specific algorithm**:
```bash
python train_dqn_simple.py
```

4. **Run comprehensive comparison**:
```bash
python compare_algorithms_fixed.py --timesteps 10000
```

## Call Chain Diagram

```
User Entry Points
    │
    ├─── quick_compare_fixed.py ─────┐
    │                                │
    ├─── quick_compare_simple_fixed.py ──┐
    │                                    │
    ├─── compare_algorithms_fixed.py ────┤
    │                                    │
    └─── Individual Training Scripts     │
         (train_*.py)                    │
              │                          │
              ▼                          ▼
         Training Functions         Uses train_* functions
              │                          │
              ├─── train_dqn() ◄─────────┤
              ├─── train_a2c() ◄─────────┤
              └─── train_ppo() ◄─────────┘
                    │
                    ▼
              Environments
                    │
    ┌───────────────┴────────────────┐
    │                                │
    ▼                                ▼
MaskedSpiderSolitaireEnvFixed   SpiderSolitaireEnvFixed
    │                                │
    └──► ActionMasker Wrapper        │
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
                   RL Algorithms
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    DQNAgent         A2CAgent      PPO (SB3)
    DQNNetwork       A2CNetwork    Custom Extractor
```

## Key Bug Fixes Implemented

1. **ActionMasker Fix**: Changed inheritance from `gym.ActionWrapper` to `gym.Wrapper` to prevent automatic action conversion
2. **A2C Environment Reset**: Added proper environment reset when episodes end
3. **DQN Random Action**: Fixed fallback to return 0 instead of invalid random action
4. **Tensor Dimension Fix**: Fixed A2C implementation for proper tensor handling
5. **PPO Progress Bar**: Disabled to avoid tqdm/rich dependency issues

## Performance Notes

- **DQN & A2C**: Achieve positive rewards (~440) with action masking
- **PPO**: Initially gets negative rewards as it learns to avoid invalid actions
- **Training Speed**: PPO > A2C > DQN
- **Stability**: A2C = DQN > PPO (without action masking)

## Future Improvements

- Implement PPO with explicit action masking
- Add LSTM for handling partial observability
- Implement Rainbow DQN
- Add distributed training support
- Create Google Colab notebooks in `colab_notebooks/`

## License

This project is for educational and research purposes.