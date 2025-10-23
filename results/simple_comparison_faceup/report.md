# Spider Solitaire Simplified RL Comparison Report (Faceup)

Generated on: 2025-10-23 18:56:22

## Summary Results (50 episodes each)

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
| Simple DQN | 0.0% | 1465.9 | 1639.7 | 177.5s |
| Simple A2C | 0.0% | 1066.5 | 1204.7 | 69.2s |
| Simple PPO | 0.0% | 1207.8 | 1207.8 | 46.4s |

## Analysis

- **Best Win Rate**: Simple DQN (0.0%)
- **Best Average Reward**: Simple DQN (1465.9)
- **Fastest Training**: Simple PPO (46.4s)

### Last Episode Details

| Algorithm | Game Result | Valid Moves | Sequences |
|-----------|-------------|-------------|-----------|
| Simple DQN | TRUNCATED | 20 | 0/2 |
| Simple PPO | TRUNCATED | 13 | 0/2 |