# Spider Solitaire Simplified RL Comparison Report (Faceup)

Generated on: 2025-10-11 19:23:26

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
| Simple DQN | 0.0% | 1183.6 | 1180.0 | 73.2s |
| Simple A2C | 0.0% | 1134.4 | 848.0 | 32.7s |
| Simple PPO | 0.0% | 861.9 | 949.4 | 27.0s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple DQN (1183.6)
- **Fastest Training**: Simple PPO (27.0s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Sequences |
|-----------|-------------|-------------|-----------|
| Simple DQN | TRUNCATED | 15 | 0/2 |
| Simple PPO | TRUNCATED | 6 | 0/2 |