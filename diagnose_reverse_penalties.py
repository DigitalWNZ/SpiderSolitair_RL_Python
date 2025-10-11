"""
Diagnose reverse move penalties by running random agent.
This helps understand if the penalty is too harsh.
"""
import numpy as np
from spider_solitaire_masked_env_faceup import create_masked_faceup_env

def test_random_agent_penalties(episodes=10):
    """Test with random valid actions to see penalty distribution."""

    env = create_masked_faceup_env(max_steps=300)

    print("="*80)
    print("REVERSE PENALTY DIAGNOSIS (Random Agent)")
    print("="*80)

    total_reverse_penalties = 0
    total_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        step = 0

        episode_reward = 0
        reverse_count = 0
        move_history = []

        while not done and step < 300:
            # Get valid actions
            mask = state.get('action_mask', None)

            if mask is not None:
                valid_actions = np.where(mask > 0)[0]

                if len(valid_actions) > 0:
                    # Random valid action
                    action = np.random.choice(valid_actions)
                else:
                    break
            else:
                action = 0

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Decode action
            action_type = action // (10 * 10 * 13)
            remainder = action % (10 * 10 * 13)
            from_col = remainder // (10 * 13)
            remainder = remainder % (10 * 13)
            to_col = remainder // 13
            num_cards = remainder % 13

            # Track moves to detect reverses
            if action_type == 0:  # Card move
                current_move = (from_col, to_col, num_cards)
                move_history.append(current_move)

                # Check for reverse in last 3 moves
                if len(move_history) >= 2:
                    for i in range(max(0, len(move_history) - 4), len(move_history) - 1):
                        prev_from, prev_to, prev_num = move_history[i]
                        curr_from, curr_to, curr_num = current_move

                        if (prev_from == curr_to and prev_to == curr_from and
                            prev_num == curr_num):
                            reverse_count += 1
                            print(f"  Episode {episode+1}, Step {step}: REVERSE detected!")
                            print(f"    Previous: {prev_num} cards col{prev_from}→col{prev_to}")
                            print(f"    Current:  {curr_num} cards col{curr_from}→col{curr_to}")
                            break
            else:
                # Reset history after dealing
                move_history = []

            episode_reward += reward
            state = next_state
            step += 1

        total_rewards.append(episode_reward)
        total_reverse_penalties += reverse_count

        print(f"\nEpisode {episode + 1}:")
        print(f"  Total Reward: {episode_reward:.1f}")
        print(f"  Steps: {step}")
        print(f"  Reverse Penalties: {reverse_count}")
        print(f"  Estimated Loss from Reverses: {reverse_count * -5:.1f}")
        print()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Episodes: {episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.1f}")
    print(f"Total Reverse Penalties: {total_reverse_penalties}")
    print(f"Avg Reverse Penalties per Episode: {total_reverse_penalties / episodes:.2f}")
    print(f"Estimated Avg Loss per Episode: {(total_reverse_penalties / episodes) * -5:.1f}")
    print("="*80)

if __name__ == "__main__":
    test_random_agent_penalties(episodes=10)
