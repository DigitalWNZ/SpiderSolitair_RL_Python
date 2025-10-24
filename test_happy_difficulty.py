"""
Test the new 'happy' difficulty level (guaranteed winnable layout)
"""

from spide_solitaire_env_faceup import SpiderSolitaireEnv
from spider_solitaire_masked_env_faceup import create_masked_faceup_env
import numpy as np


def compare_difficulties():
    """Compare different difficulty levels."""

    print("="*70)
    print("COMPARING DIFFICULTY LEVELS")
    print("="*70)

    difficulties = ['happy', 'easy', 'medium']

    for diff in difficulties:
        print(f"\n{'='*70}")
        print(f"Testing: {diff.upper()} Difficulty")
        print(f"{'='*70}")

        # Create environment with specific difficulty
        env = SpiderSolitaireEnv(
            render_mode='human',
            use_strategic_deal=True,
            difficulty=diff,
            max_steps=500
        )

        obs, info = env.reset(seed=42)

        env.render()
        print(f"\nInitial valid moves: {info['valid_moves']}")
        print(f"Stock count: {obs['stock_count'][0]}")
        print(f"Foundation count: {obs['foundation_count'][0]}")

        # Analyze the tableau structure
        print(f"\nTableau analysis:")
        for col in range(10):
            face_up_count = 0
            for card in env.tableau[col]:
                if card['face_up']:
                    face_up_count += 1

            # Get the face-up card rank
            if env.tableau[col]:
                top_card = env.tableau[col][-1]
                rank_name = env.RANKS[top_card['rank']]
                print(f"  Col {col}: {len(env.tableau[col])} cards, top={rank_name}, face_up={face_up_count}")


def test_happy_with_random_actions():
    """Test happy difficulty with random actions to see if it's easier."""

    print("\n" + "="*70)
    print("TESTING HAPPY DIFFICULTY WITH RANDOM ACTIONS")
    print("="*70)

    env = create_masked_faceup_env(
        max_steps=500,
        use_strategic_deal=True,
        difficulty='happy'
    )

    num_episodes = 10
    total_rewards = []
    sequences_completed = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)

        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 500:
            # Get valid actions
            valid_actions = np.where(obs['action_mask'] > 0)[0]

            if len(valid_actions) == 0:
                break

            # Take random valid action
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        foundation = info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)
        sequences_completed.append(foundation)

        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, Steps={step_count}, Sequences={foundation}/1")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Average Reward: {np.mean(total_rewards):.1f}")
    print(f"Max Reward: {np.max(total_rewards):.1f}")
    print(f"Average Sequences Completed: {np.mean(sequences_completed):.2f}")
    print(f"Sequences Completed (total): {np.sum(sequences_completed)}")
    print(f"Win Rate: {np.sum(np.array(sequences_completed) >= 1) / num_episodes * 100:.1f}%")


def demo_happy_vs_easy():
    """Quick comparison of happy vs easy."""

    print("\n" + "="*70)
    print("HAPPY vs EASY - Initial State Comparison")
    print("="*70)

    for diff in ['happy', 'easy']:
        env = SpiderSolitaireEnv(
            use_strategic_deal=True,
            difficulty=diff,
            max_steps=500
        )

        obs, info = env.reset(seed=42)

        print(f"\n{diff.upper()} Difficulty:")
        print(f"  Initial valid moves: {info['valid_moves']}")

        # Count cards that can form sequences
        sequence_potential = 0
        for col in range(10):
            if len(env.tableau[col]) >= 2:
                # Check for potential sequences
                for i in range(len(env.tableau[col]) - 1):
                    if env.tableau[col][i]['rank'] == env.tableau[col][i+1]['rank'] + 1:
                        sequence_potential += 1

        print(f"  Sequential card pairs: {sequence_potential}")


if __name__ == "__main__":
    # Compare all difficulty levels
    compare_difficulties()

    # Quick comparison
    demo_happy_vs_easy()

    # Test with random actions
    test_happy_with_random_actions()
