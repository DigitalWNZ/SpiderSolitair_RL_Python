"""
Diagnostic script to analyze action space validity in faceup environment.
"""
import numpy as np
from spide_solitaire_env_faceup import SpiderSolitaireEnv

def analyze_action_space(num_episodes=10):
    """Analyze what percentage of actions are valid."""
    env = SpiderSolitaireEnv(max_steps=500)

    total_states = 0
    total_valid_moves = 0
    total_action_space = 2 * 10 * 10 * 13  # 2600

    valid_move_counts = []
    valid_percentages = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        step = 0

        while not done and step < 100:  # Sample 100 steps per episode
            valid_moves = info.get('valid_moves', 0)
            valid_move_counts.append(valid_moves)
            valid_percentage = (valid_moves / total_action_space) * 100
            valid_percentages.append(valid_percentage)

            total_states += 1
            total_valid_moves += valid_moves

            # Take a random action
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

    print("="*60)
    print("ACTION SPACE ANALYSIS - Faceup Environment")
    print("="*60)
    print(f"Total action space size: {total_action_space}")
    print(f"States sampled: {total_states}")
    print(f"\nValid Moves Statistics:")
    print(f"  Average valid moves per state: {np.mean(valid_move_counts):.2f}")
    print(f"  Min valid moves: {np.min(valid_move_counts)}")
    print(f"  Max valid moves: {np.max(valid_move_counts)}")
    print(f"  Std dev: {np.std(valid_move_counts):.2f}")
    print(f"\nValid Action Percentage:")
    print(f"  Average: {np.mean(valid_percentages):.4f}%")
    print(f"  Min: {np.min(valid_percentages):.4f}%")
    print(f"  Max: {np.max(valid_percentages):.4f}%")
    print(f"\nRandom Exploration Success Rate:")
    print(f"  Expected valid action every {total_action_space / np.mean(valid_move_counts):.1f} attempts")
    print("="*60)

    # Analyze action distribution
    print("\nAction Type Distribution Analysis:")
    env.reset()

    # Count valid actions by type
    deal_valid = 1 if len(env.stock) > 0 else 0
    move_valid = env._count_valid_moves() - deal_valid

    print(f"  Deal from stock: {deal_valid} valid actions (0.04% of space)")
    print(f"  Move cards: ~{move_valid} valid actions (0.4% of space)")
    print(f"  Invalid actions: ~{total_action_space - deal_valid - move_valid} (99.6% of space)")

    return {
        'avg_valid_moves': np.mean(valid_move_counts),
        'avg_valid_percentage': np.mean(valid_percentages),
        'total_action_space': total_action_space
    }

if __name__ == "__main__":
    results = analyze_action_space(num_episodes=10)
