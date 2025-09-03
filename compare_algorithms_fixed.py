import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from typing import Dict, List, Tuple

# Import fixed environments
from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
from spider_solitaire_masked_env_fixed import MaskedSpiderSolitaireEnvFixed, ActionMasker

# Import training modules
from train_dqn import DQNAgent
from train_a2c import A2CAgent

# For PPO, we'll use stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: stable-baselines3 not available, skipping PPO")


class MetricsCallback(BaseCallback):
    """Callback to collect metrics during training."""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.wins = 0
        self.total_episodes = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            reward = self.locals['rewards'][0]
            
            self.total_episodes += 1
            self.episode_rewards.append(info.get('episode', {}).get('r', reward))
            self.episode_lengths.append(info.get('episode', {}).get('l', 1))
            self.episode_times.append(time.time() - self.start_time)
            
            if reward > 0:  # Win condition
                self.wins += 1
            
            # Log episode details periodically
            if self.total_episodes % 100 == 0:
                game_result = info.get('game_result', 'UNKNOWN')
                valid_moves = info.get('valid_moves', 0)
                invalid_moves = info.get('invalid_moves', 0)
                foundation_count = info.get('foundation_count', 0)
                print(f"  [PPO] Episode {self.total_episodes}: {game_result}, "
                      f"Valid/Invalid: {valid_moves}/{invalid_moves}, "
                      f"Sequences: {foundation_count}/8")
                
        return True


def train_dqn(total_timesteps: int, max_steps_per_episode: int = 500) -> Dict:
    """Train DQN agent with fixed environment."""
    print("\n" + "="*60)
    print("Training DQN (Deep Q-Network)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60)
    
    start_time = time.time()
    
    # Create environment
    env = ActionMasker(MaskedSpiderSolitaireEnvFixed(max_steps=max_steps_per_episode))
    
    # Create agent
    agent = DQNAgent(
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
    )
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    wins = 0
    episodes = 0
    total_steps = 0
    
    while total_steps < total_timesteps:
        episodes += 1
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_start = time.time()
        
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
            total_steps += 1
            
            if done:
                # Extract detailed episode info
                game_result = info.get('game_result', 'UNKNOWN')
                valid_moves = info.get('valid_moves', 0)
                invalid_moves = info.get('invalid_moves', 0)
                foundation_count = info.get('foundation_count', 0)
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_times.append(time.time() - episode_start)
        
        if episode_reward > 0:
            wins += 1
        
        # Update epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Progress update
        if episodes % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            win_rate = wins / episodes
            print(f"Episode {episodes}, Steps: {total_steps:,}/{total_timesteps:,}, "
                  f"Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2%}")
            if 'game_result' in locals():
                print(f"  Last episode: {game_result}, Valid/Invalid: {valid_moves}/{invalid_moves}, "
                      f"Sequences: {foundation_count}/8")
    
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/dqn_spider_solitaire_fixed.pt")
    
    env.close()
    
    return {
        'algorithm': 'DQN',
        'total_timesteps': total_steps,
        'episodes': episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_times': episode_times,
        'wins': wins,
        'win_rate': wins / episodes if episodes > 0 else 0,
    }


def train_a2c(total_timesteps: int, max_steps_per_episode: int = 500) -> Dict:
    """Train A2C agent with fixed environment."""
    print("\n" + "="*60)
    print("Training A2C (Advantage Actor-Critic)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60)
    
    start_time = time.time()
    
    # For single environment to avoid dimension issues
    n_envs = 1
    env = ActionMasker(MaskedSpiderSolitaireEnvFixed(max_steps=max_steps_per_episode))
    
    # Create agent with single environment
    agent = A2CAgent(
        env,
        n_envs=n_envs,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=5,
    )
    
    # Initialize states
    states = [env.reset()[0]]
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    wins = 0
    episodes = 0
    total_steps = 0
    
    # Episode tracking
    current_reward = 0
    current_length = 0
    episode_start = time.time()
    
    while total_steps < total_timesteps:
        # Collect rollouts
        rollout_data, states = agent.collect_rollouts(states, agent.n_steps)
        
        # Process rollout data
        for step in range(len(rollout_data['rewards'])):
            reward = rollout_data['rewards'][step][0]  # Single environment
            done = rollout_data['dones'][step][0]
            current_reward += reward
            current_length += 1
            total_steps += 1
            
            if done:
                episodes += 1
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                episode_times.append(time.time() - episode_start)
                
                if current_reward > 0:
                    wins += 1
                
                # Reset tracking
                current_reward = 0
                current_length = 0
                episode_start = time.time()
        
        # Train on rollouts
        agent.train_step(rollout_data)
        
        # Progress update
        if episodes > 0 and episodes % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            win_rate = wins / episodes
            print(f"Episodes: {episodes}, Steps: {total_steps:,}/{total_timesteps:,}, "
                  f"Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2%}")
    
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/a2c_spider_solitaire_fixed.pt")
    
    # Close environment
    env.close()
    
    return {
        'algorithm': 'A2C',
        'total_timesteps': total_steps,
        'episodes': episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_times': episode_times,
        'wins': wins,
        'win_rate': wins / episodes if episodes > 0 else 0,
    }


def train_ppo(total_timesteps: int, max_steps_per_episode: int = 500) -> Dict:
    """Train PPO agent with fixed environment."""
    if not PPO_AVAILABLE:
        print("PPO not available. Please install stable-baselines3.")
        return None
        
    print("\n" + "="*60)
    print("Training PPO (Proximal Policy Optimization)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60)
    
    start_time = time.time()
    
    # Create vectorized environment
    # Note: PPO doesn't use explicit action masking but learns to avoid invalid actions through rewards
    n_envs = 4
    env = make_vec_env(
        lambda: Monitor(SpiderSolitaireEnvFixed(max_steps=max_steps_per_episode)),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv  # Use DummyVecEnv for stability
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dict observation spaces
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,  # Reduce verbosity
    )
    
    # Metrics callback
    callback = MetricsCallback()
    
    # Train with progress bar disabled
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_spider_solitaire_fixed")
    
    env.close()
    
    return {
        'algorithm': 'PPO',
        'total_timesteps': total_timesteps,
        'episodes': callback.total_episodes,
        'training_time': training_time,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'episode_times': callback.episode_times,
        'wins': callback.wins,
        'win_rate': callback.wins / callback.total_episodes if callback.total_episodes > 0 else 0,
    }


def plot_results(results: List[Dict]):
    """Generate comparison plots."""
    os.makedirs("results/algorithm_comparison_fixed", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spider Solitaire RL Algorithm Comparison (Fixed)', fontsize=16)
    
    colors = {'PPO': 'blue', 'DQN': 'red', 'A2C': 'green'}
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # 1. Learning Curves
    ax = axes[0, 0]
    for result in results:
        alg = result['algorithm']
        rewards = result['episode_rewards']
        # Smooth rewards
        window = min(100, len(rewards) // 10)
        if window > 1 and len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = np.arange(len(smoothed))
            ax.plot(episodes, smoothed, label=alg, color=colors[alg])
    ax.set_title('Learning Curves (Smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Win Rate Over Time
    ax = axes[0, 1]
    for result in results:
        alg = result['algorithm']
        n_episodes = len(result['episode_rewards'])
        wins_cumsum = np.cumsum([1 if r > 0 else 0 for r in result['episode_rewards']])
        episodes = np.arange(1, n_episodes + 1)
        win_rates = wins_cumsum / episodes
        ax.plot(episodes, win_rates, label=alg, color=colors[alg])
    ax.set_title('Win Rate Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Length Distribution
    ax = axes[0, 2]
    for result in results:
        alg = result['algorithm']
        lengths = result['episode_lengths']
        ax.hist(lengths, bins=50, alpha=0.5, label=alg, color=colors[alg])
    ax.set_title('Episode Length Distribution')
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Performance Comparison
    ax = axes[1, 0]
    algorithms = [r['algorithm'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    bars = ax.bar(algorithms, win_rates, color=[colors[alg] for alg in algorithms])
    ax.set_title('Final Win Rates')
    ax.set_ylabel('Win Rate')
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Efficiency
    ax = axes[1, 1]
    steps_per_second = [r['total_timesteps'] / r['training_time'] for r in results]
    bars = ax.bar(algorithms, steps_per_second, color=[colors[alg] for alg in algorithms])
    ax.set_title('Training Speed')
    ax.set_ylabel('Steps per Second')
    for bar, speed in zip(bars, steps_per_second):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{speed:.0f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Average Reward (Last 100 Episodes)
    ax = axes[1, 2]
    avg_rewards_last_100 = []
    for result in results:
        rewards = result['episode_rewards']
        if len(rewards) >= 100:
            avg_rewards_last_100.append(np.mean(rewards[-100:]))
        else:
            avg_rewards_last_100.append(np.mean(rewards))
    bars = ax.bar(algorithms, avg_rewards_last_100, color=[colors[alg] for alg in algorithms])
    ax.set_title('Average Reward (Last 100 Episodes)')
    ax.set_ylabel('Average Reward')
    for bar, reward in zip(bars, avg_rewards_last_100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{reward:.0f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison_fixed/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved to results/algorithm_comparison_fixed/comparison.png")


def generate_report(results: List[Dict]):
    """Generate a comprehensive comparison report."""
    os.makedirs("results/algorithm_comparison_fixed", exist_ok=True)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    report = []
    report.append("# Spider Solitaire RL Algorithm Comparison Report (Fixed)")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## Environment Fixes")
    report.append("- Added maximum steps per episode (truncation)")
    report.append("- Proper handling of stuck games")
    report.append("- Fixed action masking edge cases")
    
    report.append("\n## Training Configuration")
    if results:
        report.append(f"- Total timesteps per algorithm: {results[0]['total_timesteps']:,}")
        report.append(f"- Number of parallel environments: 1 (DQN, A2C), 4 (PPO)")
    
    report.append("\n## Summary Results\n")
    report.append("| Algorithm | Episodes | Win Rate | Avg Reward (Last 100) | Training Time | Steps/Second |")
    report.append("|-----------|----------|----------|-----------------------|---------------|--------------|")
    
    for r in results:
        avg_reward_last_100 = np.mean(r['episode_rewards'][-100:]) if len(r['episode_rewards']) >= 100 else np.mean(r['episode_rewards'])
        steps_per_second = r['total_timesteps'] / r['training_time']
        report.append(f"| {r['algorithm']} | {r['episodes']} | {r['win_rate']:.1%} | "
                     f"{avg_reward_last_100:.1f} | {r['training_time']:.1f}s | {steps_per_second:.0f} |")
    
    report.append("\n## Detailed Analysis\n")
    
    # Add last episode details if available
    report.append("\n### Training Progress\n")
    report.append("The environment now tracks detailed episode information including:")
    report.append("- Game result (WON/LOST/TRUNCATED)")
    report.append("- Number of valid and invalid moves")
    report.append("- Foundation count (completed sequences)")
    report.append("- Episode length and rewards")
    
    # Best performer analysis
    if results:
        best_win_rate = max(r['win_rate'] for r in results)
        best_win_alg = [r['algorithm'] for r in results if r['win_rate'] == best_win_rate][0]
        
        best_reward = max(np.mean(r['episode_rewards'][-100:]) if len(r['episode_rewards']) >= 100 else np.mean(r['episode_rewards']) for r in results)
        best_reward_alg = [r['algorithm'] for r in results if (np.mean(r['episode_rewards'][-100:]) if len(r['episode_rewards']) >= 100 else np.mean(r['episode_rewards'])) == best_reward][0]
        
        fastest_steps = max(r['total_timesteps'] / r['training_time'] for r in results)
        fastest_alg = [r['algorithm'] for r in results if r['total_timesteps'] / r['training_time'] == fastest_steps][0]
        
        report.append(f"- **Best Win Rate**: {best_win_alg} ({best_win_rate:.1%})")
        report.append(f"- **Best Average Reward**: {best_reward_alg} ({best_reward:.1f})")
        report.append(f"- **Fastest Training**: {fastest_alg} ({fastest_steps:.0f} steps/second)")
    
    report.append("\n## Recommendations")
    report.append("- For highest win rate: Use the algorithm with best win rate")
    report.append("- For fastest experimentation: Use the algorithm with highest steps/second")
    report.append("- For best sample efficiency: Compare episodes needed to reach target performance")
    report.append("- Monitor valid/invalid move ratios to assess learning quality")
    report.append("- Track foundation counts to measure progress toward winning")
    
    # Save report
    with open('results/algorithm_comparison_fixed/report.md', 'w') as f:
        f.write('\n'.join(report))
    
    # Save raw results
    with open('results/algorithm_comparison_fixed/raw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nReport saved to results/algorithm_comparison_fixed/report.md")
    print('\n'.join(report))


def main():
    """Run comprehensive comparison of all algorithms."""
    parser = argparse.ArgumentParser(description='Compare RL algorithms on Spider Solitaire')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps to train each algorithm (default: 100000)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--algorithms', nargs='+', default=['dqn', 'a2c', 'ppo'],
                        choices=['dqn', 'a2c', 'ppo'],
                        help='Algorithms to compare (default: all)')
    
    args = parser.parse_args()
    
    print("Spider Solitaire RL Algorithm Comparison (Fixed)")
    print(f"Training each algorithm for {args.timesteps:,} timesteps")
    print(f"Maximum steps per episode: {args.max_steps}")
    print("="*60)
    
    results = []
    
    # Train selected algorithms
    if 'dqn' in args.algorithms:
        try:
            results.append(train_dqn(args.timesteps, args.max_steps))
        except Exception as e:
            print(f"DQN training failed: {e}")
            import traceback
            traceback.print_exc()
    
    if 'a2c' in args.algorithms:
        try:
            results.append(train_a2c(args.timesteps, args.max_steps))
        except Exception as e:
            print(f"A2C training failed: {e}")
            import traceback
            traceback.print_exc()
    
    if 'ppo' in args.algorithms and PPO_AVAILABLE:
        try:
            results.append(train_ppo(args.timesteps, args.max_steps))
        except Exception as e:
            print(f"PPO training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate plots and report
    if results:
        plot_results(results)
        generate_report(results)
    else:
        print("No successful training runs!")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("Results saved to results/algorithm_comparison_fixed/")


if __name__ == "__main__":
    import argparse
    main()