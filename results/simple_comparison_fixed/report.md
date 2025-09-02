# Spider Solitaire Simplified RL Comparison Report (Fixed)

Generated on: 2025-09-02 15:39:43

## Summary Results (2 episodes each)

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
| Simple DQN | 0.0% | -550.0 | -550.0 | 1.3s |
| Simple A2C | 0.0% | -141.7 | -141.7 | 0.2s |
| Simple PPO | 0.0% | -100.0 | -100.0 | 0.2s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple PPO (-100.0)
- **Fastest Training**: Simple A2C (0.2s)