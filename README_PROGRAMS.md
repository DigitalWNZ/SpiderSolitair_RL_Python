# Spider Solitaire RL - Program Documentation

This document provides a comprehensive guide to all programs in the Spider Solitaire Reinforcement Learning project.

## Table of Contents
1. [Environment Programs](#environment-programs)
2. [Training Programs](#training-programs)
3. [Comparison Programs](#comparison-programs)
4. [Utility Programs](#utility-programs)
5. [Test Programs](#test-programs)

---

## Environment Programs

### `spider_solitaire_env.py`
**Function**: Base Spider Solitaire game environment following OpenAI Gym interface.
- Implements game rules and logic
- Provides observation space (tableau, stock, foundation)
- Handles card movements and scoring
- **Note**: Original version without truncation (can cause infinite loops)

**Dependencies**: 
- `gymnasium`, `numpy`

**Called by**: All training scripts when not using fixed versions

---

### `spider_solitaire_env_fixed.py`
**Function**: Fixed version of Spider Solitaire environment with proper game truncation.
- Adds maximum steps per episode (default: 500)
- Detects stuck games and truncates appropriately
- Prevents infinite loops during training

**How to use**:
```python
from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
env = SpiderSolitaireEnvFixed(max_steps=500)
```

**Dependencies**: 
- `gymnasium`, `numpy`

**Called by**: Fixed training and comparison scripts

---

### `spider_solitaire_masked_env.py`
**Function**: Adds action masking to the base environment.
- Computes valid actions for each state
- Provides action mask in observations
- Includes `ActionMasker` wrapper for discrete action space

**Dependencies**: 
- `spider_solitaire_env.py`, `gymnasium`, `numpy`

**Called by**: Training scripts that use action masking

---

### `spider_solitaire_masked_env_fixed.py`
**Function**: Fixed version with both action masking and truncation.
- Combines features of masked environment and fixed environment
- Properly handles edge cases when no valid actions exist
- Includes `ActionMasker` wrapper

**How to use**:
```python
from spider_solitaire_masked_env_fixed import create_masked_env
env = create_masked_env(max_steps=500)
```

**Dependencies**: 
- `spider_solitaire_env_fixed.py`, `gymnasium`, `numpy`

**Called by**: Fixed training scripts

---

## Training Programs

### `train_dqn.py`
**Function**: Trains a Deep Q-Network (DQN) agent on Spider Solitaire.
- Implements DQN with experience replay and target network
- Uses CNN for tableau feature extraction
- Supports epsilon-greedy exploration

**How to run**:
```bash
python train_dqn.py
```

**Key classes**:
- `DQNNetwork`: Neural network architecture
- `ReplayBuffer`: Experience replay storage
- `DQNAgent`: Main training agent

**Dependencies**:
- `spider_solitaire_env.py`, `spider_solitaire_masked_env.py`
- `torch`, `gymnasium`, `numpy`, `matplotlib`

---

### `train_dqn_simple.py`
**Function**: Simplified DQN implementation with lighter network architecture.
- Reduced CNN layers (2 conv layers instead of 3)
- Smaller fully connected layers (128 neurons)
- Faster training but potentially lower performance

**How to run**:
```bash
python train_dqn_simple.py
```

**Key differences from `train_dqn.py`:
- Smaller network: 16→32 conv channels, 128 FC neurons
- Smaller replay buffer (50,000 vs 100,000)

**Dependencies**: Same as `train_dqn.py`

---

### `train_a2c.py`
**Function**: Trains an Advantage Actor-Critic (A2C) agent.
- Implements actor-critic architecture
- Uses Generalized Advantage Estimation (GAE)
- Supports multiple parallel environments

**How to run**:
```bash
python train_a2c.py
```

**Key classes**:
- `A2CNetwork`: Shared network with actor and critic heads
- `A2CAgent`: Handles rollout collection and training

**Dependencies**:
- `spider_solitaire_env.py`, `spider_solitaire_masked_env.py`
- `torch`, `gymnasium`, `numpy`, `matplotlib`

---

### `train_a2c_simple.py`
**Function**: Simplified A2C with lighter architecture.
- Reduced network complexity for faster training
- Same algorithm logic as original A2C

**How to run**:
```bash
python train_a2c_simple.py
```

**Key differences from `train_a2c.py`:
- Smaller network architecture
- Single shared layer (128 neurons)

**Dependencies**: Same as `train_a2c.py`

---

### `train_ppo.py`
**Function**: Trains a PPO agent using Stable Baselines3.
- Uses custom feature extractor for Spider Solitaire
- Implements clipped objective for stable training
- Includes training callbacks for monitoring

**How to run**:
```bash
# Training mode
python train_ppo.py --mode train

# Evaluation mode
python train_ppo.py --mode eval --model-path models/best/best_model.zip
```

**Key classes**:
- `SpiderSolitaireFeaturesExtractor`: Custom CNN feature extractor
- `TrainingCallback`: Tracks training metrics

**Dependencies**:
- `spider_solitaire_env.py`
- `stable_baselines3`, `torch`, `gymnasium`

---

### `train_ppo_simple.py`
**Function**: Simplified PPO with lighter feature extractor.
- Reduced CNN and policy network size
- Faster training with fewer parameters

**How to run**:
```bash
python train_ppo_simple.py --mode train
```

**Dependencies**: Same as `train_ppo.py`

---

## Comparison Programs

### `quick_compare.py`
**Function**: Quick comparison of DQN, A2C, and PPO algorithms.
- Trains each algorithm for a small number of episodes
- Generates comparison plots and reports
- **Warning**: Uses original environments (may deadlock)

**How to run**:
```bash
python quick_compare.py
```

**Output**:
- `results/quick_comparison/comparison.png`: Performance plots
- `results/quick_comparison/report.md`: Summary report

**Calls**:
- `train_dqn.py` (imports DQNAgent)
- `train_a2c.py` (imports A2CAgent)
- Uses PPO from stable_baselines3

---

### `quick_compare_fixed.py`
**Function**: Fixed version of quick comparison with truncation.
- Uses fixed environments to prevent deadlocks
- Configurable episodes and max steps

**How to run**:
```bash
# Default: 100 episodes, 500 max steps
python quick_compare_fixed.py

# Custom settings
python quick_compare_fixed.py --episodes 50 --max-steps 200
```

**Arguments**:
- `--episodes`: Number of episodes per algorithm (default: 100)
- `--max-steps`: Maximum steps per episode (default: 500)

**Output**:
- `results/quick_comparison_fixed/`: Results directory

**Calls**:
- `spider_solitaire_env_fixed.py`
- `spider_solitaire_masked_env_fixed.py`
- Imports agents from training scripts

---

### `quick_compare_simple.py`
**Function**: Compares simplified versions of all algorithms.
- Uses lightweight network architectures
- Faster training for rapid experimentation

**How to run**:
```bash
python quick_compare_simple.py --episodes 100
```

**Output**:
- `results/simple_comparison/`: Results directory

**Calls**:
- `train_dqn_simple.py` (imports SimpleDQNAgent)
- `train_a2c_simple.py` (imports SimpleA2CAgent)
- `train_ppo_simple.py` (imports SimpleSpiderSolitaireFeaturesExtractor)

---

### `quick_compare_simple_fixed.py`
**Function**: Fixed version comparing simplified algorithms.
- Combines simplified networks with fixed environments
- Most reliable for quick testing

**How to run**:
```bash
python quick_compare_simple_fixed.py --episodes 10 --max-steps 100
```

**Output**:
- `results/simple_comparison_fixed/`: Results directory

**Calls**:
- Fixed environment modules
- Simple training modules

---

### `compare_algorithms.py`
**Function**: Comprehensive comparison with longer training runs.
- Trains algorithms for specified total timesteps
- Generates detailed performance analysis
- **Warning**: Uses original environments

**How to run**:
```bash
# Train all algorithms for 1M timesteps
python compare_algorithms.py --timesteps 1000000

# Train specific algorithms
python compare_algorithms.py --timesteps 500000 --algorithms dqn a2c
```

**Arguments**:
- `--timesteps`: Total training timesteps (default: 100,000)
- `--algorithms`: Which algorithms to compare (default: all)

**Output**:
- `results/algorithm_comparison/`: Comprehensive results

---

### `compare_algorithms_fixed.py`
**Function**: Fixed version for comprehensive comparison.
- Uses fixed environments with truncation
- Suitable for long training runs without deadlock

**How to run**:
```bash
python compare_algorithms_fixed.py --timesteps 1000000 --max-steps 500
```

**Arguments**:
- `--timesteps`: Total training timesteps
- `--max-steps`: Maximum steps per episode
- `--algorithms`: Algorithms to compare [dqn, a2c, ppo]

**Output**:
- `results/algorithm_comparison_fixed/`: Results directory

---

## Utility Programs

### `play_spider_solitaire.py`
**Function**: Manual play interface for Spider Solitaire.
- Human-playable version of the game
- Useful for understanding game mechanics
- Text-based interface

**How to run**:
```bash
python play_spider_solitaire.py
```

**Controls**:
- `m <from> <to> <num>`: Move cards
- `d`: Deal from stock
- `q`: Quit

**Calls**: `spider_solitaire_env.py`

---

### `evaluate_model.py`
**Function**: Evaluate a trained model's performance.
- Load and test saved models
- Generate performance statistics
- Option to render games

**How to run**:
```bash
# Evaluate DQN model
python evaluate_model.py --model-path models/dqn_spider_final.pt --algorithm dqn

# Evaluate with rendering
python evaluate_model.py --model-path models/a2c_spider_final.pt --algorithm a2c --render
```

**Calls**: Corresponding training modules for model loading

---

## Test Programs

### `test_fixed_env.py`
**Function**: Tests the fixed environment implementations.
- Verifies truncation works correctly
- Tests action masking functionality
- Quick sanity check for environments

**How to run**:
```bash
python test_fixed_env.py
```

**Calls**:
- `spider_solitaire_masked_env_fixed.py`
- `spider_solitaire_env_fixed.py`

---

### `test_quick_compare_simple.py`
**Function**: Quick test for simplified comparison script.
- Runs comparison with minimal episodes
- Verifies script functionality

**How to run**:
```bash
python test_quick_compare_simple.py
```

**Calls**: `quick_compare_simple.py`

---

### `test_comparison_scripts.py`
**Function**: Tests all comparison scripts.
- Verifies fixed comparison scripts work
- Runs with minimal settings for quick validation

**How to run**:
```bash
python test_comparison_scripts.py
```

**Calls**:
- `quick_compare_fixed.py`
- `compare_algorithms_fixed.py`

---

## Recommended Usage Flow

### For Quick Testing:
1. Start with `test_fixed_env.py` to verify environments work
2. Run `quick_compare_simple_fixed.py --episodes 10` for rapid testing
3. Check results in `results/simple_comparison_fixed/`

### For Serious Training:
1. Use `compare_algorithms_fixed.py --timesteps 1000000` for comprehensive comparison
2. Train individual algorithms with their respective scripts
3. Evaluate best models with `evaluate_model.py`

### For Development:
1. Test changes with simplified versions first
2. Use fixed environments to avoid deadlocks
3. Monitor training with generated plots and reports

## Common Issues and Solutions

1. **Deadlock/Infinite Loop**: Use `*_fixed.py` versions of scripts
2. **Out of Memory**: Use simplified networks or reduce batch size
3. **Slow Training**: Start with `*_simple.py` versions
4. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install gymnasium numpy torch matplotlib stable-baselines3
   ```

## Output Directory Structure

```
results/
├── quick_comparison/          # Original quick compare results
├── quick_comparison_fixed/    # Fixed quick compare results
├── simple_comparison/         # Simplified algorithm results
├── simple_comparison_fixed/   # Fixed simplified results
├── algorithm_comparison/      # Comprehensive comparison
└── algorithm_comparison_fixed/# Fixed comprehensive results

models/
├── dqn_spider_*.pt           # DQN checkpoints
├── a2c_spider_*.pt           # A2C checkpoints
├── ppo_spider_*/             # PPO checkpoints
└── simple_*/                 # Simplified model checkpoints
```