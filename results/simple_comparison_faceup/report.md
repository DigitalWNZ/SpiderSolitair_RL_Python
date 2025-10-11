# Spider Solitaire Simplified RL Comparison Report (Faceup)

Generated on: 2025-10-10 21:17:07

## Summary Results (10 episodes each)

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
| Simple DQN | 0.0% | 1537.0 | 1537.0 | 68.6s |
| Simple A2C | 0.0% | 1382.0 | 1382.0 | 25.0s |
| Simple PPO | 0.0% | -10050.0 | -10050.0 | 21.6s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple DQN (1537.0)
- **Fastest Training**: Simple PPO (21.6s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Sequences |
|-----------|-------------|-------------|-----------|
| Simple DQN | TRUNCATED | 12 | 0/4 |
| Simple PPO | TRUNCATED | 13 | 0/4 |