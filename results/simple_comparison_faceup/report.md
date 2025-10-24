# Spider Solitaire Simplified RL Comparison Report (Faceup)

Generated on: 2025-10-24 11:52:38

## Summary Results (30 episodes each)

### Simplified Network Architecture:
- **Conv Layers**: 2 layers (16→32 channels)
- **FC Layers**: Single layer (128 neurons)
- **Total Parameters**: ~10x fewer than original networks

### Environment Features:
- Faceup environment with all cards visible
- Maximum steps per episode (truncation)
- Proper handling of stuck games
- Action masking for valid moves

| Algorithm | Win Rate | Avg Reward | Final 20 Avg | Training Time |
|-----------|----------|------------|--------------|---------------|
| Simple DQN | 0.0% | 1160.8 | 1122.7 | 66.2s |
| Simple A2C | 0.0% | 4067.6 | 4418.0 | 37.9s |
| Simple PPO | 0.0% | 827.6 | 831.9 | 28.0s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple A2C (4067.6)
- **Fastest Training**: Simple PPO (28.0s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Sequences |
|-----------|-------------|-------------|-----------|
| Simple DQN | TRUNCATED | 9 | 0/2 |
| Simple PPO | TRUNCATED | 6 | 0/2 |