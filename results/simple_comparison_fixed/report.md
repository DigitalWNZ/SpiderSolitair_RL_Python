# Spider Solitaire Simplified RL Comparison Report (Fixed)

Generated on: 2025-09-03 15:46:28

## Summary Results (1000 episodes each)

### Simplified Network Architecture:
- **Conv Layers**: 2 layers (16→32 channels)
- **FC Layers**: Single layer (128 neurons)
- **Total Parameters**: ~10x fewer than original networks

### Environment Fixes:
- Added maximum steps per episode (truncation)
- Proper handling of stuck games
- Fixed action masking edge cases

| Algorithm | Win Rate | Avg Reward | Final 20 Avg | Training Time |
|-----------|----------|------------|--------------|---------------|
| Simple DQN | 0.0% | -906.3 | -872.6 | 6849.0s |
| Simple A2C | 0.0% | -939.6 | -720.5 | 2474.9s |
| Simple PPO | 0.0% | -10049.5 | -10049.5 | 1124.7s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple DQN (-906.3)
- **Fastest Training**: Simple PPO (1124.7s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Invalid Moves | Sequences |
|-----------|-------------|-------------|---------------|-----------|
| Simple DQN | TRUNCATED | 1000 | 0 | 0/8 |
| Simple PPO | TRUNCATED | 0 | 1000 | 0/8 |