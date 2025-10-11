import gymnasium as gym
from gymnasium import spaces
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

from spide_solitaire_env_faceup import SpiderSolitaireEnv
from spider_solitaire_masked_env_faceup import MaskedSpiderSolitaireEnvFaceup


class ActionMaskingWrapper(gym.Wrapper):
    """
    Wrapper to handle action masking for stable-baselines3.
    Automatically filters invalid actions before they're taken.
    """
    def __init__(self, env):
        super().__init__(env)
        # Remove action_mask from observation space for SB3 compatibility
        self.observation_space = spaces.Dict({
            'tableau': env.observation_space['tableau'],
            'stock_count': env.observation_space['stock_count'],
            'foundation_count': env.observation_space['foundation_count'],
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Store mask internally but don't expose it in observation
        self._current_mask = obs.pop('action_mask', None)
        return obs, info

    def step(self, action):
        # Convert action to scalar if needed
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = int(action)
            elif action.size == 1:
                action = int(action.item())
            else:
                # For multi-dimensional arrays, take first element
                action = int(action.flatten()[0])

        # If we have a mask, validate the action
        if self._current_mask is not None:
            valid_actions = np.where(self._current_mask > 0)[0]
            if len(valid_actions) > 0 and action not in valid_actions:
                # Replace invalid action with random valid action
                action = int(np.random.choice(valid_actions))

        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store mask internally
        self._current_mask = obs.pop('action_mask', None)
        return obs, reward, terminated, truncated, info


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
    Custom callback for detailed episode tracking in Spider Solitaire.
    Tracks game results, valid/invalid moves, and other episode statistics.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_details = []
        self.wins = 0
        self.losses = 0
        self.truncated = 0
        self.total_episodes = 0

        # Aggregate statistics
        self.total_valid_moves = 0
        self.total_invalid_moves = 0
        self.total_moves = 0

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones')[0]:
            self.total_episodes += 1

            # Get episode information from the environment info
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode_reward', 0)

            # Extract detailed episode information
            valid_moves = info.get('valid_moves', 0)
            total_moves = info.get('moves', 0)
            foundation_count_val = info.get('foundation_count', 0)
            # Handle both numpy arrays and scalars
            if isinstance(foundation_count_val, np.ndarray):
                foundation_count = int(foundation_count_val[0])
            else:
                foundation_count = int(foundation_count_val)

            final_score = info.get('score', 0)
            steps = info.get('current_step', 0)
            max_step = info.get('max_step', 500)

            # Determine game result based on foundation count and truncation (1 sequence)
            if foundation_count >= 1:  # Win condition (changed from 2 to 1)
                game_result = 'WON'
                self.wins += 1
            elif steps >= max_step:
                game_result = 'TRUNCATED'
                self.truncated += 1
            else:
                game_result = 'LOST'
                self.losses += 1

            # Store episode details
            episode_detail = {
                'episode': self.total_episodes,
                'result': game_result,
                'reward': episode_reward,
                'score': final_score,
                'steps': steps,
                'valid_moves': valid_moves,
                'total_moves': total_moves,
                'foundation_count': foundation_count,
            }
            self.episode_details.append(episode_detail)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            # Update aggregate statistics
            self.total_valid_moves += valid_moves
            self.total_moves += total_moves

            # Log every 100 episodes
            if self.total_episodes % 100 == 0:
                self._log_statistics()

            # Detailed log every 1000 episodes
            if self.total_episodes % 1000 == 0:
                self._log_detailed_statistics()

        return True

    def _log_statistics(self):
        """Log basic statistics."""
        win_rate = self.wins / self.total_episodes
        loss_rate = self.losses / self.total_episodes
        truncated_rate = self.truncated / self.total_episodes
        avg_reward = np.mean(self.episode_rewards[-100:])
        avg_length = np.mean(self.episode_lengths[-100:])

        print(f"\n{'='*60}")
        print(f"Episodes: {self.total_episodes}")
        print(f"Win Rate: {win_rate:.2%} | Loss Rate: {loss_rate:.2%} | Truncated: {truncated_rate:.2%}")
        print(f"Avg Reward (last 100): {avg_reward:.2f}")
        print(f"Avg Episode Length (last 100): {avg_length:.1f}")
        print(f"{'='*60}\n")

    def _log_detailed_statistics(self):
        """Log detailed statistics including move analysis."""
        recent_details = self.episode_details[-1000:]

        # Calculate detailed statistics
        won_episodes = [d for d in recent_details if d['result'] == 'WON']
        lost_episodes = [d for d in recent_details if d['result'] == 'LOST']
        truncated_episodes = [d for d in recent_details if d['result'] == 'TRUNCATED']

        print(f"\n{'='*80}")
        print(f"DETAILED STATISTICS (Last 1000 episodes)")
        print(f"{'='*80}")

        # Win/Loss/Truncated statistics
        print(f"\nGame Results:")
        print(f"  Won: {len(won_episodes)} ({len(won_episodes)/10:.1%})")
        print(f"  Lost: {len(lost_episodes)} ({len(lost_episodes)/10:.1%})")
        print(f"  Truncated: {len(truncated_episodes)} ({len(truncated_episodes)/10:.1%})")

        # Move statistics
        if self.total_moves > 0:
            print(f"\nMove Statistics (Total):")
            print(f"  Total Moves: {self.total_moves}")
            print(f"  Valid Moves: {self.total_valid_moves} ({self.total_valid_moves/self.total_moves:.1%})")

        # Statistics by outcome
        for outcome, episodes in [('Won', won_episodes), ('Lost', lost_episodes), ('Truncated', truncated_episodes)]:
            if episodes:
                avg_reward = np.mean([e['reward'] for e in episodes])
                avg_steps = np.mean([e['steps'] for e in episodes])
                avg_foundation = np.mean([e['foundation_count'] for e in episodes])

                print(f"\n{outcome} Episodes:")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Steps: {avg_steps:.1f}")
                print(f"  Average Foundations: {avg_foundation:.2f}")

        print(f"{'='*80}\n")

    def get_final_statistics(self):
        """Get final training statistics."""
        return {
            'total_episodes': self.total_episodes,
            'wins': self.wins,
            'losses': self.losses,
            'truncated': self.truncated,
            'win_rate': self.wins / max(self.total_episodes, 1),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'total_valid_moves': self.total_valid_moves,
            'total_moves': self.total_moves,
        }


def make_env(rank, seed=0, max_steps=500):
    """
    Utility function for multiprocessed env with action masking.
    """
    def _init():
        env = MaskedSpiderSolitaireEnvFaceup(max_steps=max_steps)
        env = ActionMaskingWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_spider_solitaire_simple(total_timesteps=1_000_000, n_envs=4, learning_rate=3e-4, max_steps=500):
    """
    Train a simplified PPO agent on Spider Solitaire.

    Args:
        total_timesteps: Total timesteps to train for
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        max_steps: Maximum steps per episode

    Returns:
        model: Trained PPO model
        env: Training environment
    """
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/simple_spider_solitaire_faceup_{timestamp}"
    model_dir = f"./models/simple_spider_solitaire_faceup_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Training Simplified PPO on Faceup Environment")
    print("Network architecture: 2 Conv layers (16, 32 channels), 1 FC layer (128 neurons)")
    print("Policy/Value networks: 128→128 (single layer each)")
    print(f"Max steps per episode: {max_steps}")

    # Environment setup with action masking
    def env_fn():
        env = MaskedSpiderSolitaireEnvFaceup(max_steps=max_steps)
        env = ActionMaskingWrapper(env)
        return env

    env = make_vec_env(
        env_fn,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )

    # Evaluation environment with action masking
    eval_env = MaskedSpiderSolitaireEnvFaceup(max_steps=max_steps)
    eval_env = ActionMaskingWrapper(eval_env)
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
        name_prefix='simple_spider_solitaire_faceup_ppo',
    )

    callbacks = [training_callback, eval_callback, checkpoint_callback]

    # Train the model
    print("Starting training...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,  # Disable progress bar to avoid dependency issues
    )

    # Print final training statistics
    print("\n" + "="*80)
    print("FINAL TRAINING STATISTICS")
    print("="*80)

    final_stats = training_callback.get_final_statistics()
    print(f"\nTotal Episodes: {final_stats['total_episodes']}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"\nGame Results:")
    print(f"  Wins: {final_stats['wins']} ({final_stats['win_rate']:.2%})")
    print(f"  Losses: {final_stats['losses']} ({final_stats['losses']/max(final_stats['total_episodes'], 1):.2%})")
    print(f"  Truncated: {final_stats['truncated']} ({final_stats['truncated']/max(final_stats['total_episodes'], 1):.2%})")
    print(f"\nPerformance Metrics:")
    print(f"  Average Episode Reward: {final_stats['avg_reward']:.2f}")
    print(f"  Total Valid Moves: {final_stats['total_valid_moves']:,}")
    print(f"  Total Moves: {final_stats['total_moves']:,}")

    # Save final model
    model.save(os.path.join(model_dir, 'final_model'))
    print(f"\nTraining completed! Model saved to {model_dir}")
    print("="*80 + "\n")

    return model, env


def evaluate_model(model_path, n_episodes=10, render=True, max_steps=500):
    """
    Evaluate a trained model with detailed episode tracking.
    """
    # Load model
    model = PPO.load(model_path)

    # Create environment with action masking
    env = MaskedSpiderSolitaireEnvFaceup(render_mode="human" if render else None, max_steps=max_steps)
    env = ActionMaskingWrapper(env)

    # Tracking variables
    episode_results = []

    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*80}\n")

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

        # Extract episode information
        valid_moves = info.get('valid_moves', 0)
        total_moves = info.get('moves', 0)
        foundation_count_val = info.get('foundation_count', 0)
        # Handle both numpy arrays and scalars
        if isinstance(foundation_count_val, np.ndarray):
            foundation_count = int(foundation_count_val[0])
        else:
            foundation_count = int(foundation_count_val)

        final_score = info.get('score', 0)
        steps = info.get('current_step', 0)

        # Determine game result (1 sequence)
        if foundation_count >= 1:  # Win condition (changed from 2 to 1)
            game_result = 'WON'
        elif steps >= max_steps:
            game_result = 'TRUNCATED'
        else:
            game_result = 'LOST'

        episode_data = {
            'episode': episode + 1,
            'result': game_result,
            'reward': episode_reward,
            'score': final_score,
            'steps': steps,
            'valid_moves': valid_moves,
            'total_moves': total_moves,
            'foundation_count': foundation_count,
        }
        episode_results.append(episode_data)

        # Print episode details
        print(f"Episode {episode + 1}:")
        print(f"  Result: {game_result}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Score: {final_score}")
        print(f"  Steps: {steps}")
        print(f"  Moves: {total_moves} (Valid: {valid_moves})")
        print(f"  Foundations Completed: {foundation_count}/8")
        print()

    # Calculate and display evaluation statistics
    print(f"{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    wins = sum(1 for r in episode_results if r['result'] == 'WON')
    losses = sum(1 for r in episode_results if r['result'] == 'LOST')
    truncated = sum(1 for r in episode_results if r['result'] == 'TRUNCATED')

    print(f"\nGame Results:")
    print(f"  Won: {wins} ({wins/n_episodes:.2%})")
    print(f"  Lost: {losses} ({losses/n_episodes:.2%})")
    print(f"  Truncated: {truncated} ({truncated/n_episodes:.2%})")

    avg_reward = np.mean([r['reward'] for r in episode_results])
    std_reward = np.std([r['reward'] for r in episode_results])
    avg_steps = np.mean([r['steps'] for r in episode_results])
    avg_foundations = np.mean([r['foundation_count'] for r in episode_results])

    print(f"\nPerformance Metrics:")
    print(f"  Average Reward: {avg_reward:.2f} (±{std_reward:.2f})")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Average Foundations Completed: {avg_foundations:.2f}")

    # Statistics by outcome
    for outcome in ['WON', 'LOST', 'TRUNCATED']:
        outcome_episodes = [r for r in episode_results if r['result'] == outcome]
        if outcome_episodes:
            print(f"\n{outcome} Episodes ({len(outcome_episodes)}):")
            print(f"  Average Reward: {np.mean([r['reward'] for r in outcome_episodes]):.2f}")
            print(f"  Average Steps: {np.mean([r['steps'] for r in outcome_episodes]):.1f}")

    print(f"\n{'='*80}\n")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate Simplified Spider Solitaire PPO agent on Faceup environment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Train a new model or evaluate existing one')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during evaluation')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total timesteps for training')

    args = parser.parse_args()

    if args.mode == 'train':
        model, env = train_spider_solitaire_simple(
            total_timesteps=args.timesteps,
            max_steps=args.max_steps
        )
    elif args.mode == 'eval':
        if args.model_path is None:
            print("Please provide --model-path for evaluation")
        else:
            evaluate_model(args.model_path, args.episodes, not args.no_render, args.max_steps)
