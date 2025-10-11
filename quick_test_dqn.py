"""Quick test of DQN with improved settings."""
import numpy as np
from spider_solitaire_masked_env_faceup import create_masked_faceup_env
from train_dqn_simple_faceup import SimpleDQNAgent

def quick_test(num_episodes=100):
    """Test DQN for specified number of episodes."""
    env = create_masked_faceup_env(max_steps=1000)

    agent = SimpleDQNAgent(
        env,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
    )

    wins = 0
    episode_rewards = []
    episode_lengths = []
    sequences_completed = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            mask = state.get('action_mask', None)
            action = agent.select_action(state, mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(
                {k: v for k, v in state.items() if k != 'action_mask'},
                action,
                reward,
                {k: v for k, v in next_state.items() if k != 'action_mask'},
                done
            )

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train_step()

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                foundation_count_val = info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)
                foundation_count = int(foundation_count_val)

                if foundation_count >= 2:  # Win condition: 2 sequences (changed from 4)
                    wins += 1
                    print(f"Episode {episode + 1}: WON! Reward: {episode_reward:.2f}, Length: {episode_length}, Sequences: {foundation_count}/2")

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                sequences_completed.append(foundation_count)
                break

        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_sequences = np.mean(sequences_completed[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: Avg Reward: {avg_reward:.2f}, "
                  f"Win Rate: {wins/(episode+1):.2%}, Avg Sequences: {avg_sequences:.2f}/2, "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total Wins: {wins}/{num_episodes} ({wins/num_episodes:.1%})")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Average Sequences Completed: {np.mean(sequences_completed):.2f}/2")
    print(f"Best Sequences: {np.max(sequences_completed)}/2")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print("="*60)

if __name__ == "__main__":
    import sys
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    quick_test(episodes)
