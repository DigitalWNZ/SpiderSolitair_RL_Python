import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
from datetime import datetime

from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed


class SimpleSpiderSolitaireFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simplified feature extractor for Spider Solitaire.
    Uses a lighter architecture for faster training.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict):
        # Use smaller feature dimension
        super().__init__(observation_space, features_dim=128)
        
        # Simplified CNN for tableau
        self.tableau_cnn = nn.Sequential(
            # Input: (batch, 10, 19) -> (batch, 1, 10, 19)
            nn.Unflatten(1, (1, 10)),
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_tableau = torch.zeros(1, 10, 19)
            cnn_output_dim = self.tableau_cnn(sample_tableau).shape[1]
        
        # Combine all features
        total_input_dim = cnn_output_dim + 2  # +2 for stock_count and foundation_count
        
        # Simplified linear layer
        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
        )
        
    def forward(self, observations) -> torch.Tensor:
        # Extract components
        tableau = observations['tableau'].float()
        stock_count = observations['stock_count'].float()
        foundation_count = observations['foundation_count'].float()
        
        # Process tableau through CNN
        tableau_features = self.tableau_cnn(tableau)
        
        # Concatenate all features
        combined = torch.cat([
            tableau_features,
            stock_count,
            foundation_count
        ], dim=1)
        
        return self.linear(combined)


class TrainingCallback(BaseCallback):
    """
    Custom callback for logging training metrics.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones')[0]:
            self.total_episodes += 1
            episode_reward = self.locals['rewards'][0]
            self.episode_rewards.append(episode_reward)
            
            # Check if won (high reward indicates win)
            if episode_reward > 900:
                self.wins += 1
            
            # Log every 100 episodes
            if self.total_episodes % 100 == 0:
                win_rate = self.wins / self.total_episodes
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episodes: {self.total_episodes}, "
                      f"Win Rate: {win_rate:.2%}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}")
                
        return True


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = SpiderSolitaireEnvFixed()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_spider_solitaire_simple(total_timesteps=1_000_000, n_envs=4, learning_rate=3e-4):
    """
    Train a simplified PPO agent on Spider Solitaire.
    
    Args:
        total_timesteps: Total timesteps to train for
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        
    Returns:
        model: Trained PPO model
        env: Training environment
    """
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/simple_spider_solitaire_{timestamp}"
    model_dir = f"./models/simple_spider_solitaire_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Training Simplified PPO")
    print("Network architecture: 2 Conv layers (16, 32 channels), 1 FC layer (128 neurons)")
    print("Policy/Value networks: 128→128 (single layer each)")
    
    # Environment setup
    env = make_vec_env(SpiderSolitaireEnvFixed, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Evaluation environment
    eval_env = SpiderSolitaireEnvFixed()
    eval_env = Monitor(eval_env)
    
    # Simplified PPO hyperparameters
    policy_kwargs = dict(
        features_extractor_class=SimpleSpiderSolitaireFeaturesExtractor,
        net_arch=[dict(pi=[128], vf=[128])],  # Simplified architecture
        activation_fn=nn.ReLU,
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=1024,  # Reduced from 2048
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Callbacks
    training_callback = TrainingCallback()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix='simple_spider_solitaire_ppo',
    )
    
    callbacks = [training_callback, eval_callback, checkpoint_callback]
    
    # Train the model
    print("Starting training...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,  # Disable progress bar to avoid dependency issues
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'final_model'))
    print(f"Training completed! Model saved to {model_dir}")
    
    return model, env


def evaluate_model(model_path, n_episodes=10, render=True):
    """
    Evaluate a trained model.
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = SpiderSolitaireEnvFixed(render_mode="human" if render else None)
    
    wins = 0
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        if episode_reward > 900:  # Win condition
            wins += 1
            
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Score = {info['score']}, Moves = {info['moves']}")
    
    print(f"\nEvaluation Results:")
    print(f"Win Rate: {wins/n_episodes:.2%}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Std Reward: {np.std(total_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate Simplified Spider Solitaire PPO agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Train a new model or evaluate existing one')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, env = train_spider_solitaire_simple()
    elif args.mode == 'eval':
        if args.model_path is None:
            print("Please provide --model-path for evaluation")
        else:
            evaluate_model(args.model_path, args.episodes, not args.no_render)