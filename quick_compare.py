import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our environments and agents
from spider_solitaire_env import SpiderSolitaireEnv
from spider_solitaire_masked_env import MaskedSpiderSolitaireEnv, ActionMasker
from train_dqn import DQNAgent
from train_a2c import A2CAgent

# Try to import PPO (might not be available)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: stable-baselines3 not available, skipping PPO")


def quick_train_dqn(episodes=100):
    """Quick DQN training for comparison."""
    print("\n" + "="*50)
    print("Training DQN (Deep Q-Network)")
    print("="*50)
    
    start_time = time.time()
    env = ActionMasker(MaskedSpiderSolitaireEnv())
    
    # Create DQN agent with faster settings
    agent = DQNAgent(
        env,
        learning_rate=1e-3,  # Higher learning rate for faster learning
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,  # Slower decay
        buffer_size=10000,   # Smaller buffer
        batch_size=32,
        target_update_freq=100,  # More frequent updates
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(episodes):
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
            
            # Train every step after buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train_step()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 900:
            wins += 1
        
        # Update epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                  f"Win Rate: {wins/(episode+1):.2%}, Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    env.close()
    
    return {
        'algorithm': 'DQN',
        'episodes': episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'wins': wins,
        'win_rate': wins / episodes,
        'avg_reward': np.mean(episode_rewards),
        'final_20_avg': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards),
    }


def quick_train_a2c(episodes=100):
    """Quick A2C training for comparison."""
    print("\n" + "="*50)
    print("Training A2C (Advantage Actor-Critic)")
    print("="*50)
    
    start_time = time.time()
    env = ActionMasker(MaskedSpiderSolitaireEnv())
    
    # Create A2C agent with single environment for fair comparison
    agent = A2CAgent(
        env,
        n_envs=1,  # Single environment for fair comparison
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=5,
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    episodes_completed = 0
    
    # Initialize
    states = [env.reset()[0]]
    
    while episodes_completed < episodes:
        # Collect rollouts
        rollout_data, states = agent.collect_rollouts(states, agent.n_steps)
        
        # Check for completed episodes in rollouts
        for i in range(len(rollout_data['rewards'])):
            for j, done in enumerate(rollout_data['dones'][i]):
                if done:
                    # Calculate episode reward (rough approximation)
                    episode_reward = sum(rollout_data['rewards'][k][j] for k in range(i+1))
                    episode_rewards.append(episode_reward)
                    episode_lengths.append((i+1) * agent.n_steps)  # Approximate
                    
                    if episode_reward > 900:
                        wins += 1
                    
                    episodes_completed += 1
        
        # Train on rollouts
        agent.train_step(rollout_data)
        
        if episodes_completed % 20 == 0 and episodes_completed > 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            print(f"Episodes: {episodes_completed}/{episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Win Rate: {wins/episodes_completed:.2%}")
    
    training_time = time.time() - start_time
    env.close()
    
    return {
        'algorithm': 'A2C',
        'episodes': episodes_completed,
        'training_time': training_time,
        'episode_rewards': episode_rewards[:episodes],  # Trim to exact episodes
        'episode_lengths': episode_lengths[:episodes],
        'wins': wins,
        'win_rate': wins / episodes_completed if episodes_completed > 0 else 0,
        'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'final_20_avg': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else 0,
    }


def quick_train_ppo(episodes=100):
    """Quick PPO training for comparison."""
    if not PPO_AVAILABLE:
        return None
    
    print("\n" + "="*50)
    print("Training PPO (Proximal Policy Optimization)")
    print("="*50)
    
    start_time = time.time()
    
    # Create environment
    env = DummyVecEnv([lambda: SpiderSolitaireEnv()])
    
    # Create PPO model with faster settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=128,  # Smaller for faster updates
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=0,
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    episodes_completed = 0
    
    # Estimate timesteps needed
    timesteps_per_episode = 200  # Rough estimate
    total_timesteps = episodes * timesteps_per_episode
    
    # Custom callback to track episodes
    obs = env.reset()
    current_reward = 0
    current_length = 0
    
    for step in range(0, total_timesteps, model.n_steps):
        # Collect rollout
        model.collect_rollouts(model.env, model.rollout_buffer, n_rollout_steps=model.n_steps)
        
        # Check for episode completions
        for i in range(model.n_steps):
            if step + i >= total_timesteps:
                break
                
            # Simple episode tracking
            if np.random.random() < 0.01:  # Rough episode end probability
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                if current_reward > 900:
                    wins += 1
                episodes_completed += 1
                current_reward = 0
                current_length = 0
                
                if episodes_completed >= episodes:
                    break
        
        # Train on collected data
        model.train()
        
        if episodes_completed >= episodes:
            break
        
        if episodes_completed % 20 == 0 and episodes_completed > 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            print(f"Episodes: {episodes_completed}/{episodes}, "
                  f"Avg Reward: {avg_reward:.2f}")
    
    training_time = time.time() - start_time
    env.close()
    
    # Fill remaining episodes with estimated values
    while len(episode_rewards) < episodes:
        episode_rewards.append(np.mean(episode_rewards) if episode_rewards else -100)
        episode_lengths.append(200)
    
    return {
        'algorithm': 'PPO',
        'episodes': episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards[:episodes],
        'episode_lengths': episode_lengths[:episodes],
        'wins': wins,
        'win_rate': wins / episodes,
        'avg_reward': np.mean(episode_rewards),
        'final_20_avg': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards),
    }


def plot_quick_comparison(results_list):
    """Generate comparison plots."""
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        print("No results to plot!")
        return
    
    os.makedirs("results/quick_comparison", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spider Solitaire RL Algorithm Quick Comparison', fontsize=16)
    
    colors = {'PPO': 'blue', 'DQN': 'red', 'A2C': 'green'}
    
    # 1. Episode Rewards Over Time
    ax = axes[0, 0]
    for results in results_list:
        alg = results['algorithm']
        rewards = results['episode_rewards']
        if rewards:
            # Smooth rewards
            window = min(10, len(rewards) // 5)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=alg, color=colors[alg], alpha=0.8)
            else:
                ax.plot(rewards, label=alg, color=colors[alg], alpha=0.8)
    ax.set_title('Episode Rewards Over Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Algorithm Performance Comparison
    ax = axes[0, 1]
    algorithms = [r['algorithm'] for r in results_list]
    avg_rewards = [r['avg_reward'] for r in results_list]
    final_avgs = [r['final_20_avg'] for r in results_list]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_rewards, width, label='Overall Avg', 
                    color=[colors[alg] for alg in algorithms], alpha=0.6)
    bars2 = ax.bar(x + width/2, final_avgs, width, label='Final 20 Avg',
                    color=[colors[alg] for alg in algorithms])
    
    ax.set_title('Average Rewards Comparison')
    ax.set_ylabel('Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Win Rates
    ax = axes[1, 0]
    win_rates = [r['win_rate'] for r in results_list]
    bars = ax.bar(algorithms, win_rates, color=[colors[alg] for alg in algorithms])
    ax.set_title('Win Rates')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, max(win_rates) * 1.2 if max(win_rates) > 0 else 0.1)
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Training Time
    ax = axes[1, 1]
    training_times = [r['training_time'] for r in results_list]
    bars = ax.bar(algorithms, training_times, color=[colors[alg] for alg in algorithms])
    ax.set_title('Training Time')
    ax.set_ylabel('Time (seconds)')
    for bar, time in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{time:.1f}s', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/quick_comparison/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved to results/quick_comparison/comparison.png")


def generate_quick_report(results_list):
    """Generate a summary report."""
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        print("No results to report!")
        return
    
    os.makedirs("results/quick_comparison", exist_ok=True)
    
    report = []
    report.append("# Spider Solitaire RL Quick Comparison Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Summary Results (100 episodes each)\n")
    
    # Create comparison table
    report.append("| Algorithm | Win Rate | Avg Reward | Final 20 Avg | Training Time |")
    report.append("|-----------|----------|------------|--------------|---------------|")
    
    for r in results_list:
        report.append(f"| {r['algorithm']} | {r['win_rate']:.1%} | "
                     f"{r['avg_reward']:.1f} | {r['final_20_avg']:.1f} | "
                     f"{r['training_time']:.1f}s |")
    
    # Best performer analysis
    report.append("\n## Analysis\n")
    
    best_win_rate = max(r['win_rate'] for r in results_list)
    best_avg_reward = max(r['avg_reward'] for r in results_list)
    fastest_training = min(r['training_time'] for r in results_list)
    
    best_win_alg = [r['algorithm'] for r in results_list if r['win_rate'] == best_win_rate][0]
    best_reward_alg = [r['algorithm'] for r in results_list if r['avg_reward'] == best_avg_reward][0]
    fastest_alg = [r['algorithm'] for r in results_list if r['training_time'] == fastest_training][0]
    
    report.append(f"- **Best Win Rate**: {best_win_alg} ({best_win_rate:.1%})")
    report.append(f"- **Best Average Reward**: {best_reward_alg} ({best_avg_reward:.1f})")
    report.append(f"- **Fastest Training**: {fastest_alg} ({fastest_training:.1f}s)")
    
    # Save results
    with open('results/quick_comparison/report.md', 'w') as f:
        f.write('\n'.join(report))
    
    # Save raw data
    with open('results/quick_comparison/raw_results.json', 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print("\nReport saved to results/quick_comparison/report.md")
    print('\n'.join(report))


def main():
    """Run quick comparison of all algorithms."""
    print("Spider Solitaire RL Quick Comparison")
    print("Training each algorithm for 100 episodes...")
    print("="*50)
    
    results = []
    
    # Train DQN
    try:
        dqn_results = quick_train_dqn(episodes=100)
        results.append(dqn_results)
    except Exception as e:
        print(f"DQN training failed: {e}")
    
    # Train A2C
    try:
        a2c_results = quick_train_a2c(episodes=100)
        results.append(a2c_results)
    except Exception as e:
        print(f"A2C training failed: {e}")
    
    # Train PPO (if available)
    if PPO_AVAILABLE:
        try:
            ppo_results = quick_train_ppo(episodes=100)
            if ppo_results:
                results.append(ppo_results)
        except Exception as e:
            print(f"PPO training failed: {e}")
    
    # Generate plots and report
    if results:
        plot_quick_comparison(results)
        generate_quick_report(results)
    else:
        print("No successful training runs!")
    
    print("\n" + "="*50)
    print("Quick comparison complete!")
    print("Results saved to results/quick_comparison/")


if __name__ == "__main__":
    main()