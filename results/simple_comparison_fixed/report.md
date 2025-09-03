# Spider Solitaire Simplified RL Comparison Report (Fixed)

Generated on: 2025-09-03 20:50:50

## Summary Results (5 episodes each)

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
| Simple DQN | 0.0% | -465.0 | -465.0 | 17.4s |
| Simple A2C | 0.0% | -488.2 | -488.2 | 4.7s |
| Simple PPO | 0.0% | -5050.0 | -5050.0 | 7.7s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple DQN (-465.0)
- **Fastest Training**: Simple A2C (4.7s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Invalid Moves | Sequences |
|-----------|-------------|-------------|---------------|-----------|
| Simple DQN | TRUNCATED | 500 | 0 | 0/8 |
| Simple PPO | TRUNCATED | 0 | 500 | 0/8 |