import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob


def plot_training_metrics(log_dir, save_path=None):
    """
    Visualize training metrics from TensorBoard logs.
    """
    # Find event files
    event_files = glob.glob(os.path.join(log_dir, 'PPO_*', 'events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    # Load the most recent event file
    event_file = sorted(event_files)[-1]
    print(f"Loading metrics from: {event_file}")
    
    # Create event accumulator
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Spider Solitaire PPO Training Metrics', fontsize=16)
    
    # Plot episode reward
    if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
        rewards = event_acc.Scalars('rollout/ep_rew_mean')
        steps = [r.step for r in rewards]
        values = [r.value for r in rewards]
        axes[0, 0].plot(steps, values, 'b-', alpha=0.7)
        axes[0, 0].set_title('Average Episode Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode length
    if 'rollout/ep_len_mean' in event_acc.Tags()['scalars']:
        lengths = event_acc.Scalars('rollout/ep_len_mean')
        steps = [l.step for l in lengths]
        values = [l.value for l in lengths]
        axes[0, 1].plot(steps, values, 'g-', alpha=0.7)
        axes[0, 1].set_title('Average Episode Length')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot loss
    if 'train/loss' in event_acc.Tags()['scalars']:
        losses = event_acc.Scalars('train/loss')
        steps = [l.step for l in losses]
        values = [l.value for l in losses]
        axes[1, 0].plot(steps, values, 'r-', alpha=0.7)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'train/learning_rate' in event_acc.Tags()['scalars']:
        lrs = event_acc.Scalars('train/learning_rate')
        steps = [l.step for l in lrs]
        values = [l.value for l in lrs]
        axes[1, 1].plot(steps, values, 'm-', alpha=0.7)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def analyze_game_statistics(env, model, n_episodes=100):
    """
    Analyze detailed game statistics from trained model.
    """
    stats = {
        'rewards': [],
        'moves': [],
        'scores': [],
        'foundations': [],
        'win': [],
        'valid_moves_per_step': [],
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_valid_moves = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            step_valid_moves.append(info['valid_moves'])
        
        stats['rewards'].append(episode_reward)
        stats['moves'].append(info['moves'])
        stats['scores'].append(info['score'])
        stats['foundations'].append(obs['foundation_count'][0])
        stats['win'].append(1 if episode_reward > 0 else 0)
        stats['valid_moves_per_step'].append(np.mean(step_valid_moves))
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Spider Solitaire Game Statistics Analysis', fontsize=16)
    
    # Reward distribution
    axes[0, 0].hist(stats['rewards'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(stats['rewards']), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(stats["rewards"]):.1f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Moves distribution
    axes[0, 1].hist(stats['moves'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(stats['moves']), color='r', linestyle='--',
                       label=f'Mean: {np.mean(stats["moves"]):.1f}')
    axes[0, 1].set_title('Moves per Game Distribution')
    axes[0, 1].set_xlabel('Number of Moves')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Foundations completed
    axes[0, 2].hist(stats['foundations'], bins=9, range=(-0.5, 8.5), 
                    edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Foundations Completed Distribution')
    axes[0, 2].set_xlabel('Number of Foundations')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_xticks(range(9))
    
    # Win rate over time
    win_rate_window = 20
    win_rates = [np.mean(stats['win'][max(0, i-win_rate_window):i+1]) 
                 for i in range(len(stats['win']))]
    axes[1, 0].plot(win_rates, 'b-', alpha=0.7)
    axes[1, 0].set_title(f'Win Rate (Rolling {win_rate_window} games)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Score vs Moves scatter
    axes[1, 1].scatter(stats['moves'], stats['scores'], alpha=0.5)
    axes[1, 1].set_title('Score vs Moves')
    axes[1, 1].set_xlabel('Number of Moves')
    axes[1, 1].set_ylabel('Final Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Valid moves availability
    axes[1, 2].hist(stats['valid_moves_per_step'], bins=20, 
                    edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Average Valid Moves per Step')
    axes[1, 2].set_xlabel('Average Valid Moves')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Game Statistics Summary ===")
    print(f"Total Episodes: {n_episodes}")
    print(f"Win Rate: {np.mean(stats['win']):.2%}")
    print(f"Average Reward: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    print(f"Average Moves: {np.mean(stats['moves']):.1f} ± {np.std(stats['moves']):.1f}")
    print(f"Average Score: {np.mean(stats['scores']):.1f} ± {np.std(stats['scores']):.1f}")
    print(f"Average Foundations: {np.mean(stats['foundations']):.2f} ± {np.std(stats['foundations']):.2f}")
    
    return stats


if __name__ == "__main__":
    import argparse
    from stable_baselines3 import PPO
    from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
    
    parser = argparse.ArgumentParser(description='Visualize Spider Solitaire training')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model for statistics analysis')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots instead of displaying')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Number of episodes for statistics analysis')
    
    args = parser.parse_args()
    
    # Plot training metrics
    save_path = 'training_metrics.png' if args.save_plots else None
    plot_training_metrics(args.log_dir, save_path)
    
    # Analyze game statistics if model provided
    if args.model_path:
        print(f"\nAnalyzing model: {args.model_path}")
        model = PPO.load(args.model_path)
        env = SpiderSolitaireEnvFixed()
        
        stats = analyze_game_statistics(env, model, args.n_episodes)
        
        if args.save_plots:
            plt.savefig('game_statistics.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        env.close()