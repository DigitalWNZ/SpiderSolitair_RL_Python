import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from typing import Dict, List, Tuple

# Import training modules
from train_ppo import train_spider_solitaire as train_ppo_full
from train_dqn import DQNAgent, ActionMasker, MaskedSpiderSolitaireEnv
from train_a2c import A2CAgent
from spider_solitaire_env import SpiderSolitaireEnv

# For PPO, we'll use stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


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
            
            if reward > 900:  # Win condition
                self.wins += 1
                
        return True


def train_ppo_comparison(timesteps: int = 100000) -> Dict:
    """Train PPO with limited timesteps for comparison."""
    print("\n" + "="*50)
    print("Training PPO (Proximal Policy Optimization)")
    print("="*50)
    
    start_time = time.time()
    
    # Create environment
    n_envs = 4
    env = make_vec_env(SpiderSolitaireEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Create model with same hyperparameters as full training
    model = PPO(
        "MlpPolicy",  # Using MLP for fair comparison
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
        verbose=1,
    )
    
    # Create callback
    callback = MetricsCallback()
    
    # Train
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # Save model
    model.save("models/comparison/ppo_model")
    
    training_time = time.time() - start_time
    
    # Collect results
    results = {
        'algorithm': 'PPO',
        'timesteps': timesteps,
        'training_time': training_time,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'episode_times': callback.episode_times,
        'total_episodes': callback.total_episodes,
        'wins': callback.wins,
        'win_rate': callback.wins / callback.total_episodes if callback.total_episodes > 0 else 0,
    }
    
    env.close()
    return results


def train_dqn_comparison(episodes: int = 1000) -> Dict:
    """Train DQN with limited episodes for comparison."""
    print("\n" + "="*50)
    print("Training DQN (Deep Q-Network)")
    print("="*50)
    
    start_time = time.time()
    
    # Create environment
    env = ActionMasker(MaskedSpiderSolitaireEnv())
    
    # Create agent
    agent = DQNAgent(
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,  # Smaller buffer for faster training
        batch_size=32,
        target_update_freq=1000,
    )
    
    # Modified training loop to collect metrics
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    wins = 0
    timesteps = 0
    
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
            
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            timesteps += 1
            
            if timesteps % agent.target_update_freq == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_times.append(time.time() - start_time)
        
        if episode_reward > 900:
            wins += 1
        
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Win Rate: {wins/(episode+1):.2%}, Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    os.makedirs("models/comparison", exist_ok=True)
    agent.save_model("models/comparison/dqn_model.pt")
    
    training_time = time.time() - start_time
    
    results = {
        'algorithm': 'DQN',
        'timesteps': timesteps,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_times': episode_times,
        'total_episodes': episodes,
        'wins': wins,
        'win_rate': wins / episodes,
    }
    
    env.close()
    return results


def train_a2c_comparison(timesteps: int = 100000) -> Dict:
    """Train A2C with limited timesteps for comparison."""
    print("\n" + "="*50)
    print("Training A2C (Advantage Actor-Critic)")
    print("="*50)
    
    start_time = time.time()
    
    # Create environment
    env = ActionMasker(MaskedSpiderSolitaireEnv())
    
    # Create agent
    agent = A2CAgent(
        env,
        n_envs=4,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=5,
    )
    
    # Modified training to match timesteps
    print(f"Training A2C for {timesteps} timesteps...")
    
    # Initialize environments
    envs = [ActionMasker(MaskedSpiderSolitaireEnv()) for _ in range(agent.n_envs)]
    states = [env.reset()[0] for env in envs]
    agent.env.envs = envs
    
    num_updates = timesteps // (agent.n_steps * agent.n_envs)
    episode_times = []
    
    for update in range(num_updates):
        # Collect rollouts
        rollout_data, states = agent.collect_rollouts(states, agent.n_steps)
        
        # Record times for episodes
        if agent.total_episodes > len(episode_times):
            for _ in range(agent.total_episodes - len(episode_times)):
                episode_times.append(time.time() - start_time)
        
        # Train on rollouts
        agent.train_step(rollout_data)
        
        # Logging
        if update % 100 == 0:
            avg_reward = np.mean(list(agent.episode_rewards)) if agent.episode_rewards else 0
            win_rate = agent.wins / agent.total_episodes if agent.total_episodes > 0 else 0
            print(f"Update {update}/{num_updates}, Episodes: {agent.total_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2%}")
    
    # Save model
    agent.save_model("models/comparison/a2c_model.pt")
    
    training_time = time.time() - start_time
    
    results = {
        'algorithm': 'A2C',
        'timesteps': timesteps,
        'training_time': training_time,
        'episode_rewards': list(agent.episode_rewards),
        'episode_lengths': list(agent.episode_lengths),
        'episode_times': episode_times,
        'total_episodes': agent.total_episodes,
        'wins': agent.wins,
        'win_rate': agent.wins / agent.total_episodes if agent.total_episodes > 0 else 0,
    }
    
    env.close()
    return results


def evaluate_model(algorithm: str, model_path: str, n_episodes: int = 100) -> Dict:
    """Evaluate a trained model."""
    print(f"\nEvaluating {algorithm} model...")
    
    env = SpiderSolitaireEnv()
    
    if algorithm == 'PPO':
        model = PPO.load(model_path)
        
        def get_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
            
    elif algorithm == 'DQN':
        from train_dqn import DQNNetwork
        agent = DQNAgent(env)
        agent.load_model(model_path)
        agent.epsilon = 0  # No exploration during evaluation
        
        def get_action(obs):
            return agent.select_action(obs)
            
    elif algorithm == 'A2C':
        from train_a2c import A2CNetwork
        agent = A2CAgent(env, n_envs=1)
        agent.load_model(model_path)
        
        def get_action(obs):
            state_dict = {
                k: torch.FloatTensor(v).unsqueeze(0).to(agent.device)
                for k, v in obs.items() if k != 'action_mask'
            }
            with torch.no_grad():
                policy_logits, _ = agent.network(state_dict)
                probs = torch.softmax(policy_logits, dim=-1)
                return probs.argmax().item()
    
    # Run evaluation
    rewards = []
    lengths = []
    wins = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        if episode_reward > 900:
            wins += 1
    
    env.close()
    
    return {
        'algorithm': algorithm,
        'eval_rewards': rewards,
        'eval_lengths': lengths,
        'eval_wins': wins,
        'eval_win_rate': wins / n_episodes,
        'eval_avg_reward': np.mean(rewards),
        'eval_std_reward': np.std(rewards),
        'eval_avg_length': np.mean(lengths),
    }


def plot_comparison(results_list: List[Dict], save_dir: str = "results/comparison"):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Spider Solitaire RL Algorithm Comparison', fontsize=16)
    
    colors = {'PPO': 'blue', 'DQN': 'red', 'A2C': 'green'}
    
    # 1. Episode Rewards Over Time
    ax = axes[0, 0]
    for results in results_list:
        alg = results['algorithm']
        rewards = results['episode_rewards']
        if len(rewards) > 0:
            # Smooth rewards
            window = min(50, len(rewards) // 10)
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
    
    # 2. Win Rate Progress
    ax = axes[0, 1]
    for results in results_list:
        alg = results['algorithm']
        total_eps = results['total_episodes']
        if total_eps > 0:
            # Calculate cumulative win rate
            rewards = results['episode_rewards']
            wins = [1 if r > 900 else 0 for r in rewards]
            win_rates = []
            for i in range(1, len(wins) + 1):
                win_rates.append(sum(wins[:i]) / i)
            ax.plot(win_rates, label=alg, color=colors[alg])
    ax.set_title('Win Rate Progress')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training Efficiency
    ax = axes[0, 2]
    for results in results_list:
        alg = results['algorithm']
        times = results['episode_times']
        rewards = results['episode_rewards']
        if len(times) > 0 and len(rewards) > 0:
            # Plot rewards vs wall-clock time
            ax.scatter(times, rewards, label=alg, color=colors[alg], alpha=0.5, s=10)
    ax.set_title('Training Efficiency (Rewards vs Time)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Performance Comparison
    ax = axes[1, 0]
    algorithms = [r['algorithm'] for r in results_list]
    win_rates = [r['win_rate'] for r in results_list]
    bars = ax.bar(algorithms, win_rates, color=[colors[alg] for alg in algorithms])
    ax.set_title('Final Win Rates')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, max(win_rates) * 1.2 if max(win_rates) > 0 else 0.1)
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 5. Average Episode Length
    ax = axes[1, 1]
    avg_lengths = []
    for results in results_list:
        lengths = results['episode_lengths']
        avg_lengths.append(np.mean(lengths) if lengths else 0)
    bars = ax.bar(algorithms, avg_lengths, color=[colors[alg] for alg in algorithms])
    ax.set_title('Average Episode Length')
    ax.set_ylabel('Steps')
    for bar, length in zip(bars, avg_lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{length:.0f}', ha='center', va='bottom')
    
    # 6. Training Time Comparison
    ax = axes[1, 2]
    training_times = [r['training_time'] / 60 for r in results_list]  # Convert to minutes
    bars = ax.bar(algorithms, training_times, color=[colors[alg] for alg in algorithms])
    ax.set_title('Training Time')
    ax.set_ylabel('Time (minutes)')
    for bar, time in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time:.1f}m', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create evaluation comparison if available
    eval_data = [r for r in results_list if 'eval_avg_reward' in r]
    if eval_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Evaluation Performance Comparison', fontsize=16)
        
        algorithms = [r['algorithm'] for r in eval_data]
        
        # Evaluation win rates
        ax = axes[0]
        eval_win_rates = [r['eval_win_rate'] for r in eval_data]
        bars = ax.bar(algorithms, eval_win_rates, color=[colors[alg] for alg in algorithms])
        ax.set_title('Evaluation Win Rates (100 episodes)')
        ax.set_ylabel('Win Rate')
        for bar, rate in zip(bars, eval_win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Average rewards
        ax = axes[1]
        avg_rewards = [r['eval_avg_reward'] for r in eval_data]
        std_rewards = [r['eval_std_reward'] for r in eval_data]
        bars = ax.bar(algorithms, avg_rewards, yerr=std_rewards, 
                      color=[colors[alg] for alg in algorithms], capsize=10)
        ax.set_title('Average Evaluation Reward')
        ax.set_ylabel('Reward')
        
        # Average episode lengths
        ax = axes[2]
        avg_lengths = [r['eval_avg_length'] for r in eval_data]
        bars = ax.bar(algorithms, avg_lengths, color=[colors[alg] for alg in algorithms])
        ax.set_title('Average Episode Length')
        ax.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(results_list: List[Dict], save_path: str = "results/comparison/report.md"):
    """Generate a detailed comparison report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("# Spider Solitaire RL Algorithm Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Find best performer
        best_win_rate = max(r['win_rate'] for r in results_list)
        best_algorithm = [r['algorithm'] for r in results_list if r['win_rate'] == best_win_rate][0]
        
        f.write(f"**Best Performing Algorithm**: {best_algorithm} (Win Rate: {best_win_rate:.1%})\n\n")
        
        f.write("## Training Results\n\n")
        
        for results in results_list:
            alg = results['algorithm']
            f.write(f"### {alg}\n\n")
            f.write(f"- **Total Episodes**: {results['total_episodes']}\n")
            f.write(f"- **Total Timesteps**: {results['timesteps']:,}\n")
            f.write(f"- **Training Time**: {results['training_time']/60:.1f} minutes\n")
            f.write(f"- **Win Rate**: {results['win_rate']:.1%}\n")
            f.write(f"- **Wins**: {results['wins']}/{results['total_episodes']}\n")
            
            if results['episode_rewards']:
                f.write(f"- **Average Reward**: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}\n")
                f.write(f"- **Max Reward**: {max(results['episode_rewards']):.2f}\n")
                f.write(f"- **Final 100 Episodes Avg**: {np.mean(results['episode_rewards'][-100:]):.2f}\n")
            
            if results['episode_lengths']:
                f.write(f"- **Average Episode Length**: {np.mean(results['episode_lengths']):.1f} steps\n")
            
            f.write("\n")
        
        # Evaluation results if available
        eval_data = [r for r in results_list if 'eval_avg_reward' in r]
        if eval_data:
            f.write("## Evaluation Results (100 episodes)\n\n")
            
            for results in eval_data:
                alg = results['algorithm']
                f.write(f"### {alg}\n\n")
                f.write(f"- **Evaluation Win Rate**: {results['eval_win_rate']:.1%}\n")
                f.write(f"- **Evaluation Wins**: {results['eval_wins']}/100\n")
                f.write(f"- **Average Reward**: {results['eval_avg_reward']:.2f} ± {results['eval_std_reward']:.2f}\n")
                f.write(f"- **Average Episode Length**: {results['eval_avg_length']:.1f} steps\n\n")
        
        f.write("## Algorithm Comparison\n\n")
        
        f.write("| Metric | PPO | DQN | A2C |\n")
        f.write("|--------|-----|-----|-----|\n")
        
        metrics = {
            'Training Win Rate': lambda r: f"{r['win_rate']:.1%}",
            'Training Time': lambda r: f"{r['training_time']/60:.1f}m",
            'Episodes': lambda r: str(r['total_episodes']),
            'Timesteps': lambda r: f"{r['timesteps']:,}",
            'Avg Reward': lambda r: f"{np.mean(r['episode_rewards']):.1f}" if r['episode_rewards'] else "N/A",
        }
        
        for metric_name, metric_func in metrics.items():
            row = f"| {metric_name} |"
            for alg in ['PPO', 'DQN', 'A2C']:
                result = next((r for r in results_list if r['algorithm'] == alg), None)
                if result:
                    row += f" {metric_func(result)} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        
        f.write("\n## Conclusions\n\n")
        
        # Performance analysis
        f.write("### Performance Analysis\n\n")
        
        win_rates = [(r['algorithm'], r['win_rate']) for r in results_list]
        win_rates.sort(key=lambda x: x[1], reverse=True)
        
        f.write("**Win Rate Ranking**:\n")
        for i, (alg, rate) in enumerate(win_rates, 1):
            f.write(f"{i}. {alg}: {rate:.1%}\n")
        
        f.write("\n### Efficiency Analysis\n\n")
        
        # Calculate episodes per minute
        efficiency = [(r['algorithm'], r['total_episodes'] / (r['training_time'] / 60)) for r in results_list]
        efficiency.sort(key=lambda x: x[1], reverse=True)
        
        f.write("**Training Efficiency** (episodes per minute):\n")
        for i, (alg, eff) in enumerate(efficiency, 1):
            f.write(f"{i}. {alg}: {eff:.1f} episodes/min\n")
        
        f.write("\n### Recommendations\n\n")
        f.write(f"1. For best performance: Use **{best_algorithm}**\n")
        f.write(f"2. For fastest training: Use **{efficiency[0][0]}**\n")
        f.write("3. For sample efficiency: Use **DQN** (off-policy learning)\n")
        f.write("4. For stability: Use **PPO** (clipped objectives)\n")
    
    print(f"Report saved to {save_path}")


def main():
    """Run the complete comparison."""
    print("Spider Solitaire RL Algorithm Comparison")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models/comparison", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    
    results = []
    
    # Train each algorithm
    # Note: Using smaller timesteps for demonstration
    # For full comparison, use timesteps=1000000 or more
    
    try:
        # Train PPO
        ppo_results = train_ppo_comparison(timesteps=50000)
        results.append(ppo_results)
    except Exception as e:
        print(f"PPO training failed: {e}")
    
    try:
        # Train DQN
        dqn_results = train_dqn_comparison(episodes=500)
        results.append(dqn_results)
    except Exception as e:
        print(f"DQN training failed: {e}")
    
    try:
        # Train A2C
        a2c_results = train_a2c_comparison(timesteps=50000)
        results.append(a2c_results)
    except Exception as e:
        print(f"A2C training failed: {e}")
    
    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating Trained Models")
    print("="*50)
    
    eval_results = []
    
    if os.path.exists("models/comparison/ppo_model.zip"):
        try:
            eval_result = evaluate_model("PPO", "models/comparison/ppo_model.zip", n_episodes=100)
            # Merge with training results
            for r in results:
                if r['algorithm'] == 'PPO':
                    r.update(eval_result)
        except Exception as e:
            print(f"PPO evaluation failed: {e}")
    
    if os.path.exists("models/comparison/dqn_model.pt"):
        try:
            eval_result = evaluate_model("DQN", "models/comparison/dqn_model.pt", n_episodes=100)
            for r in results:
                if r['algorithm'] == 'DQN':
                    r.update(eval_result)
        except Exception as e:
            print(f"DQN evaluation failed: {e}")
    
    if os.path.exists("models/comparison/a2c_model.pt"):
        try:
            eval_result = evaluate_model("A2C", "models/comparison/a2c_model.pt", n_episodes=100)
            for r in results:
                if r['algorithm'] == 'A2C':
                    r.update(eval_result)
        except Exception as e:
            print(f"A2C evaluation failed: {e}")
    
    # Save raw results
    with open("results/comparison/raw_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            json_r = r.copy()
            for key, value in json_r.items():
                if isinstance(value, np.ndarray):
                    json_r[key] = value.tolist()
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    # Generate plots
    if results:
        plot_comparison(results)
        
        # Generate report
        generate_report(results)
    
    print("\n" + "="*50)
    print("Comparison Complete!")
    print("Results saved to results/comparison/")
    print("="*50)


if __name__ == "__main__":
    main()