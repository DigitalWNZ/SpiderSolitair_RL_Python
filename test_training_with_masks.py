"""
Test training behavior with action masking to understand why learning is slow.
"""
import numpy as np
from spider_solitaire_masked_env_faceup import create_masked_faceup_env
from train_dqn_simple_faceup import SimpleDQNAgent

def test_training_behavior():
    """Test a few episodes to see reward patterns."""
    env = create_masked_faceup_env(max_steps=500)

    # Create agent
    agent = SimpleDQNAgent(
        env,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=0.2,  # Lower epsilon to use more greedy actions
        epsilon_end=0.1,
        epsilon_decay=0.99,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
    )

    print("="*60)
    print("TRAINING BEHAVIOR TEST (5 episodes)")
    print("="*60)

    for episode in range(5):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        valid_moves = 0
        invalid_moves = 0
        reward_breakdown = {'revealing': 0, 'sequence': 0, 'invalid': 0, 'neutral': 0}

        while True:
            mask = state.get('action_mask', None)
            action = agent.select_action(state, mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track reward types
            if reward == -10:
                invalid_moves += 1
                reward_breakdown['invalid'] += 1
            elif reward == 0:
                valid_moves += 1
                reward_breakdown['neutral'] += 1
            elif reward == 5:
                valid_moves += 1
                reward_breakdown['revealing'] += 1
            elif reward >= 100:
                valid_moves += 1
                reward_breakdown['sequence'] += reward // 100

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        valid_pct = (valid_moves / episode_length) * 100 if episode_length > 0 else 0
        foundation_count = int(info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0))

        print(f"\nEpisode {episode + 1}:")
        print(f"  Length: {episode_length}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Valid moves: {valid_moves} ({valid_pct:.1f}%)")
        print(f"  Invalid moves: {invalid_moves}")
        print(f"  Sequences completed: {foundation_count}/8")
        print(f"  Reward breakdown:")
        print(f"    - Cards revealed: {reward_breakdown['revealing']}")
        print(f"    - Sequences: {reward_breakdown['sequence']}")
        print(f"    - Neutral moves: {reward_breakdown['neutral']}")
        print(f"    - Invalid penalties: {reward_breakdown['invalid']}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_training_behavior()
