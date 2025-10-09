"""
Play Spider Solitaire using a trained A2C model from train_a2c_simple_faceup.py
"""
import torch
import numpy as np
import argparse
from spide_solitaire_env_faceup import SpiderSolitaireEnv
from spider_solitaire_masked_env_faceup import ActionMasker, create_masked_faceup_env
from train_a2c_simple_faceup import SimpleA2CNetwork
import torch.nn.functional as F


def load_trained_model(model_path, env):
    """Load a trained A2C model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create network
    network = SimpleA2CNetwork(env.observation_space, env.action_space).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    print(f"Model loaded from {model_path}")
    print(f"Training stats: {checkpoint['total_episodes']} episodes, "
          f"{checkpoint['wins']} wins ({checkpoint['wins']/checkpoint['total_episodes']:.1%} win rate)")

    return network, device


def play_episode(network, env, device, render=True, deterministic=True):
    """Play one episode using the trained model."""
    state, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    print("\n" + "="*80)
    print("Starting new episode...")
    print("="*80)

    if render:
        env.render()

    while not done:
        # Convert state to tensor
        state_dict = {
            k: torch.FloatTensor(v).unsqueeze(0).to(device)
            for k, v in state.items() if k != 'action_mask'
        }

        # Get action from policy
        with torch.no_grad():
            policy_logits, value = network(state_dict)

            # Apply action mask if available
            if 'action_mask' in state:
                mask = torch.FloatTensor(state['action_mask']).to(device)
                policy_logits = policy_logits.masked_fill(mask == 0, float('-inf'))

            # Select action
            if deterministic:
                # Greedy action
                action = policy_logits.argmax().item()
            else:
                # Sample from policy
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1

        # Display step information
        if episode_length % 50 == 0:
            print(f"Step {episode_length}: Reward={episode_reward:.1f}, "
                  f"Valid moves={info.get('valid_moves', 0)}, "
                  f"Sequences={info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)}/8")

        if render and episode_length % 10 == 0:
            env.render()

        state = next_state

    # Extract final info
    foundation_count_val = info.get('foundation_count', 0)
    if isinstance(foundation_count_val, np.ndarray):
        foundation_count = int(foundation_count_val[0])
    else:
        foundation_count = int(foundation_count_val)

    final_score = info.get('score', 0)
    valid_moves = info.get('valid_moves', 0)
    current_step = info.get('current_step', episode_length)
    max_step = info.get('max_step', 500)

    # Determine game result
    if foundation_count >= 8:
        game_result = 'WON'
        result_symbol = '🏆'
    elif current_step >= max_step:
        game_result = 'TRUNCATED'
        result_symbol = '⏱️'
    else:
        game_result = 'LOST'
        result_symbol = '❌'

    print("\n" + "="*80)
    print(f"{result_symbol} EPISODE COMPLETE - {game_result} {result_symbol}")
    print("="*80)
    print(f"Total Steps: {episode_length}")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Final Score: {final_score}")
    print(f"Valid Moves Available: {valid_moves}")
    print(f"Sequences Completed: {foundation_count}/8")
    print(f"Steps: {current_step}/{max_step}")
    print("="*80)

    if render:
        print("\nFinal board state:")
        env.render()

    return {
        'result': game_result,
        'reward': episode_reward,
        'length': episode_length,
        'score': final_score,
        'sequences': foundation_count,
        'valid_moves': valid_moves,
    }


def play_multiple_episodes(model_path, n_episodes=10, max_steps=500, render=False, deterministic=True):
    """Play multiple episodes and show statistics."""
    # Create environment with proper masking
    env = create_masked_faceup_env(max_steps=max_steps)

    # Load model
    network, device = load_trained_model(model_path, env)

    print(f"\nPlaying {n_episodes} episodes...")
    print(f"Render: {render}, Deterministic: {deterministic}")
    print(f"Max steps per episode: {max_steps}")

    results = []
    for i in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {i+1}/{n_episodes}")
        print(f"{'='*80}")

        result = play_episode(network, env, device, render=render, deterministic=deterministic)
        results.append(result)

    env.close()

    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    wins = sum(1 for r in results if r['result'] == 'WON')
    losses = sum(1 for r in results if r['result'] == 'LOST')
    truncated = sum(1 for r in results if r['result'] == 'TRUNCATED')

    avg_reward = np.mean([r['reward'] for r in results])
    avg_length = np.mean([r['length'] for r in results])
    avg_sequences = np.mean([r['sequences'] for r in results])
    avg_score = np.mean([r['score'] for r in results])

    print(f"\nGame Results:")
    print(f"  Won: {wins} ({wins/n_episodes:.1%})")
    print(f"  Lost: {losses} ({losses/n_episodes:.1%})")
    print(f"  Truncated: {truncated} ({truncated/n_episodes:.1%})")

    print(f"\nPerformance Metrics:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_length:.1f}")
    print(f"  Average Score: {avg_score:.1f}")
    print(f"  Average Sequences: {avg_sequences:.2f}/8")

    # Best episode
    best_episode = max(results, key=lambda x: x['reward'])
    best_idx = results.index(best_episode)
    print(f"\nBest Episode (#{best_idx+1}):")
    print(f"  Result: {best_episode['result']}")
    print(f"  Reward: {best_episode['reward']:.2f}")
    print(f"  Sequences: {best_episode['sequences']}/8")

    print("\n" + "="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Play Spider Solitaire with trained A2C model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model checkpoint (.pt file)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play (default: 1)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--render', action='store_true',
                        help='Render the game state')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (sample from distribution) instead of deterministic (greedy)')

    args = parser.parse_args()

    # Play episodes
    results = play_multiple_episodes(
        model_path=args.model_path,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
