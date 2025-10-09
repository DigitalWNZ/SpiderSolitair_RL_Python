# Spider Solitaire Simplified RL Comparison Report (Faceup)

Generated on: 2025-10-09 21:28:40

## Summary Results (5 episodes each)

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
| Simple DQN | 40.0% | 4.0 | 4.0 | 21.9s |
| Simple A2C | 40.0% | 18.0 | 18.0 | 6.3s |
| Simple PPO | 0.0% | -5050.0 | -5050.0 | 7.7s |

## Analysis

- **Best Win Rate**: Simple DQN (40.0%)
- **Best Average Reward**: Simple A2C (18.0)
- **Fastest Training**: Simple A2C (6.3s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Sequences |
|-----------|-------------|-------------|-----------|
| Simple DQN | TRUNCATED | 6 | 0/8 |
| Simple PPO | TRUNCATED | 9 | 0/8 |