# PPO Training System for Spider Solitaire

This repository contains a complete Proximal Policy Optimization (PPO) training system for learning to play Spider Solitaire using reinforcement learning.

## Overview

The system implements a gymnasium-compatible Spider Solitaire environment and trains a PPO agent to play the game effectively. The implementation includes custom feature extraction, action masking for invalid moves, comprehensive monitoring, and visualization tools.

## Components

### 1. Environment (`spider_solitaire_env.py`)

The base Spider Solitaire environment following the Gymnasium API standard:

- **Game Rules**: Single-suit Spider Solitaire with 104 cards
- **State Space**: 
  - Tableau: 10 columns with up to 19 visible cards each
  - Stock count: Number of remaining deal piles (0-5)
  - Foundation count: Completed sequences (0-8)
- **Action Space**: Multi-discrete `[action_type, from_col, to_col, num_cards]`
  - `action_type`: 0 for moving cards, 1 for dealing from stock
  - `from_col/to_col`: Column indices (0-9)
  - `num_cards`: Number of cards to move (1-13)

### 2. Masked Environment (`spider_solitaire_masked_env.py`)

An enhanced version with action masking to improve training efficiency:

- Prevents the agent from attempting invalid moves
- Includes action mask in observations
- Significantly reduces training time by avoiding invalid action exploration

### 3. PPO Training Script (`train_ppo.py`)

The main training script with the following features:

#### Custom Feature Extractor
```python
class SpiderSolitaireFeaturesExtractor(BaseFeaturesExtractor):
    # CNN for processing the 10x19 tableau grid
    # Combines visual features with game state information
    # Outputs 256-dimensional feature vector
```

#### PPO Hyperparameters
- **Learning rate**: 3e-4
- **Batch size**: 64
- **N steps**: 2048
- **N epochs**: 10
- **Gamma**: 0.99
- **GAE lambda**: 0.95
- **Clip range**: 0.2
- **Entropy coefficient**: 0.01

#### Training Features
- Multi-environment training (4 parallel environments)
- Custom callbacks for monitoring win rate and episode statistics
- Automatic checkpointing every 50,000 steps
- Evaluation callback for best model selection
- TensorBoard logging for all metrics

### 4. Visualization Tools (`visualize_training.py`)

Comprehensive analysis and visualization:

- **Training Metrics**:
  - Episode rewards over time
  - Episode lengths
  - Training loss
  - Learning rate schedule

- **Game Statistics**:
  - Win rate tracking
  - Move distribution
  - Score analysis
  - Foundation completion rates
  - Valid moves availability

## Reward System

The environment uses a carefully designed reward structure:

| Event | Reward | Purpose |
|-------|--------|---------|
| Valid move | -1 | Encourages efficiency |
| Invalid move | -10 | Strongly discourages illegal actions |
| Card reveal | +5 | Rewards uncovering hidden cards |
| Sequence completion | +100 | Major milestone reward |
| Game win | +1000 | Ultimate goal achievement |
| Starting score | 500 | Baseline score |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SpiderSolitaire

# Install dependencies
pip install -r requirements.txt
```

Required packages:
- gymnasium>=0.29.0
- numpy>=1.24.0
- stable-baselines3>=2.0.0
- torch>=2.0.0
- tensorboard>=2.13.0
- matplotlib>=3.7.0

## Usage

### Training a New Model

```bash
# Start training with default parameters
python train_ppo.py --mode train

# Training will:
# - Run for 2,000,000 timesteps
# - Save checkpoints every 50,000 steps
# - Evaluate every 10,000 steps
# - Log to TensorBoard
```

### Evaluating a Trained Model

```bash
# Evaluate with rendering
python train_ppo.py --mode eval --model-path models/spider_solitaire_*/final_model --episodes 10

# Evaluate without rendering (faster)
python train_ppo.py --mode eval --model-path models/spider_solitaire_*/final_model --episodes 100 --no-render
```

### Visualizing Training Progress

```bash
# View training metrics
python visualize_training.py --log-dir logs/spider_solitaire_* 

# Analyze game statistics
python visualize_training.py --log-dir logs/spider_solitaire_* --model-path models/spider_solitaire_*/final_model --n-episodes 100

# Save plots to files
python visualize_training.py --log-dir logs/spider_solitaire_* --save-plots
```

### Interactive Testing

```bash
# Test environment manually
python example_usage.py play

# Run environment tests
python example_usage.py test

# Watch random gameplay
python example_usage.py
```

## Training Strategy

The PPO agent learns several key strategies:

1. **Card Revelation Priority**: The +5 reward for revealing face-down cards encourages the agent to uncover hidden information early.

2. **Sequence Building**: The agent learns to build valid descending sequences to maximize mobility.

3. **Foundation Completion**: The +100 reward for completing sequences incentivizes finishing K-to-A runs.

4. **Move Efficiency**: The -1 penalty per move encourages finding optimal paths rather than excessive card shuffling.

5. **Invalid Move Avoidance**: The -10 penalty and action masking help the agent learn legal moves quickly.

## Model Architecture

The neural network architecture consists of:

1. **Feature Extraction Layer**:
   - CNN for tableau processing (1→32→64 channels)
   - Concatenation with game state features
   - Final 256-dimensional representation

2. **Policy Network**:
   - Two hidden layers (256 units each)
   - ReLU activation
   - Outputs action probabilities

3. **Value Network**:
   - Separate two hidden layers (256 units each)
   - Estimates state values for advantage calculation

## Performance Expectations

After training for 2M timesteps:
- **Win Rate**: 5-15% (highly dependent on game difficulty)
- **Average Moves**: 150-250 per game
- **Foundation Completion**: 2-4 sequences on average
- **Invalid Move Rate**: <5% with action masking

## Extending the System

### Adding Multi-Suit Variants

Modify `spider_solitaire_env.py`:
```python
# Add suit parameter to initialization
def __init__(self, num_suits=1, render_mode=None):
    self.num_suits = num_suits  # 1, 2, or 4 suits
    # Adjust deck creation and validation logic
```

### Implementing Curriculum Learning

Start with easier games and gradually increase difficulty:
```python
# In training script
env = SpiderSolitaireEnv(difficulty='easy')  # More face-up cards
# Gradually transition to harder variants
```

### Adding Monte Carlo Tree Search

Combine PPO with MCTS for improved performance:
```python
# Use PPO policy as prior for MCTS
# Select actions using tree search during evaluation
```

## Troubleshooting

### Low Win Rate
- Increase training timesteps (try 5M or 10M)
- Adjust reward values (increase sequence completion bonus)
- Use curriculum learning approach

### High Invalid Move Rate
- Ensure action masking is properly implemented
- Check that the masked environment is being used
- Verify action space boundaries

### Training Instability
- Reduce learning rate
- Increase batch size
- Adjust clip range parameter

## Citation

If you use this code in your research, please cite:
```
@software{spider_solitaire_ppo,
  title = {PPO Training System for Spider Solitaire},
  year = {2024},
  url = {https://github.com/yourusername/spider-solitaire-rl}
}
```

## License

This project is provided for educational purposes. See LICENSE file for details.