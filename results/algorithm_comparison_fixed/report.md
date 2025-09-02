# Spider Solitaire RL Algorithm Comparison Report (Fixed)

Generated on: 2025-09-02 15:15:07

## Environment Fixes
- Added maximum steps per episode (truncation)
- Proper handling of stuck games
- Fixed action masking edge cases

## Training Configuration
- Total timesteps per algorithm: 1,000
- Number of parallel environments: 4 (A2C, PPO), 1 (DQN)

## Summary Results

| Algorithm | Episodes | Win Rate | Avg Reward (Last 100) | Training Time | Steps/Second |
|-----------|----------|----------|-----------------------|---------------|--------------|
| DQN | 20 | 0.0% | -550.0 | 36.7s | 27 |

## Detailed Analysis

- **Best Win Rate**: DQN (0.0%)
- **Best Average Reward**: DQN (-550.0)
- **Fastest Training**: DQN (27 steps/second)

## Recommendations
- For highest win rate: Use the algorithm with best win rate
- For fastest experimentation: Use the algorithm with highest steps/second
- For best sample efficiency: Compare episodes needed to reach target performance