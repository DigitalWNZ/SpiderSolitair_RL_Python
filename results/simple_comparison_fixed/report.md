# Spider Solitaire Simplified RL Comparison Report (Fixed)

Generated on: 2025-09-02 13:04:37

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
| Simple DQN | 0.0% | -5050.0 | -5050.0 | 8.7s |
| Simple A2C | 0.0% | -891.7 | -891.7 | 1.2s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple A2C (-891.7)
- **Fastest Training**: Simple A2C (1.2s)