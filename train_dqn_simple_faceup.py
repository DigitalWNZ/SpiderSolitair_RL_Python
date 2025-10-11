import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from spide_solitaire_env_faceup import SpiderSolitaireEnv
from spider_solitaire_masked_env_faceup import ActionMasker
from replay_episode import EpisodeRecorder


class SimpleDQNNetwork(nn.Module):
    """
    Simplified Deep Q-Network for Spider Solitaire.
    Uses a lighter architecture for faster training.
    """

    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Space):
        super(SimpleDQNNetwork, self).__init__()

        # Simplified CNN for tableau
        self.tableau_cnn = nn.Sequential(
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
        total_features = cnn_output_dim + 2  # +2 for stock_count and foundation_count

        # Simplified fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n if hasattr(action_space, 'n') else action_space.nvec.prod()),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract components
        tableau = obs['tableau'].float()
        stock_count = obs['stock_count'].float()
        foundation_count = obs['foundation_count'].float()

        # Process tableau
        tableau_features = self.tableau_cnn(tableau)

        # Combine features
        combined = torch.cat([
            tableau_features,
            stock_count,
            foundation_count
        ], dim=1)

        return self.fc(combined)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class SimpleDQNAgent:
    """
    Simplified Deep Q-Learning agent for Spider Solitaire.
    Uses lighter network architecture for faster training.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,  # Smaller buffer for faster sampling
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = None,
        record_episodes: bool = False,
        record_dir: str = "replays",
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use simplified networks
        self.q_network = SimpleDQNNetwork(env.observation_space, env.action_space).to(self.device)
        self.target_network = SimpleDQNNetwork(env.observation_space, env.action_space).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_episodes = 0

        # Episode recording
        self.record_episodes = record_episodes
        self.recorder = EpisodeRecorder(record_dir) if record_episodes else None

    def select_action(self, state: Dict[str, np.ndarray], mask: np.ndarray = None) -> int:
        """
        Select action using epsilon-greedy policy with optional masking.
        """
        if random.random() < self.epsilon:
            # Random action (respecting mask if provided)
            if mask is not None:
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
                else:
                    # No valid actions - shouldn't happen in properly designed env
                    # but return 0 as fallback
                    return 0
            return self.env.action_space.sample()
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = {
                    k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    for k, v in state.items() if k != 'action_mask'
                }
                q_values = self.q_network(state_tensor)

                # Apply mask if provided
                if mask is not None:
                    mask_tensor = torch.FloatTensor(mask).to(self.device)
                    q_values = q_values.masked_fill(mask_tensor == 0, float('-inf'))

                return q_values.argmax().item()

    def train_step(self):
        """
        Perform one training step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        state_dict = {
            k: torch.FloatTensor(np.array([s[k] for s in states])).to(self.device)
            for k in states[0].keys() if k != 'action_mask'
        }
        next_state_dict = {
            k: torch.FloatTensor(np.array([s[k] for s in next_states])).to(self.device)
            for k in next_states[0].keys() if k != 'action_mask'
        }
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(state_dict).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_dict).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.losses.append(loss.item())

    def train(self, num_episodes: int, save_freq: int = 100):
        """
        Train the DQN agent.
        """
        print(f"Training Simplified DQN on {self.device}")
        print("Network architecture: 2 Conv layers (16, 32 channels), 1 FC layer (128 neurons)")
        print(f"Replay buffer size: {self.replay_buffer.buffer.maxlen}")
        if self.record_episodes:
            print(f"Episode recording ENABLED - saving to {self.recorder.save_dir}/")

        steps = 0

        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0

            # Start recording if enabled
            if self.record_episodes:
                self.recorder.start_episode('DQN', episode)

            while True:
                # Select action
                mask = state.get('action_mask', None)
                action = self.select_action(state, mask)

                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Record step if enabled
                if self.record_episodes:
                    self.recorder.record_step(episode_length, state, action, reward, next_state, done, info)

                # Store transition
                self.replay_buffer.push(
                    {k: v for k, v in state.items() if k != 'action_mask'},
                    action,
                    reward,
                    {k: v for k, v in next_state.items() if k != 'action_mask'},
                    done
                )

                # Train
                self.train_step()

                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                steps += 1

                # Update target network
                if steps % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                if done:
                    break

            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_episodes += 1

            # Check foundation_count to determine win (1 sequence)
            foundation_count_val = info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)
            if int(foundation_count_val) >= 1:  # Win condition (changed from 2 to 1)
                self.wins += 1
                game_result = 'WON'
            elif truncated:
                game_result = 'TRUNCATED'
            else:
                game_result = 'LOST'

            # End recording if enabled
            if self.record_episodes:
                self.recorder.end_episode(game_result)

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                win_rate = self.wins / self.total_episodes
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, Win Rate: {win_rate:.2%}, "
                      f"Epsilon: {self.epsilon:.3f}")

                if self.losses:
                    avg_loss = np.mean(self.losses[-1000:])
                    print(f"  Avg Loss (last 1000): {avg_loss:.4f}")

                # Print detailed info from last episode
                if 'valid_moves' in info:
                    print(f"  Last episode - Valid moves: {info.get('valid_moves', 0)}, "
                          f"Sequences: {info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)}/1, "
                          f"Steps: {info.get('current_step', 0)}/{info.get('max_step', 1000)}")

            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f"simple_dqn_faceup_{episode}.pt")

        return self.episode_rewards, self.episode_lengths

    def save_model(self, filename: str):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'wins': self.wins,
            'total_episodes': self.total_episodes,
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.wins = checkpoint['wins']
        self.total_episodes = checkpoint['total_episodes']
        print(f"Model loaded from {filename}")

    def evaluate(self, n_episodes: int = 10, render: bool = False):
        """
        Evaluate the trained model.
        """
        eval_env = SpiderSolitaireEnv(render_mode="human" if render else None, max_steps=500)

        rewards = []
        wins = 0

        for episode in range(n_episodes):
            state, info = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Convert state
                state_dict = {
                    k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    for k, v in state.items() if k != 'action_mask'
                }

                # Get action (greedy)
                with torch.no_grad():
                    q_values = self.q_network(state_dict)

                    # Apply mask if available
                    if 'action_mask' in state:
                        mask = torch.FloatTensor(state['action_mask']).to(self.device)
                        q_values = q_values.masked_fill(mask == 0, float('-inf'))

                    action = q_values.argmax().item()

                # Take action
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if render:
                    eval_env.render()

            rewards.append(episode_reward)
            # Check foundation_count to determine win (1 sequence)
            foundation_count_val = info.get('foundation_count', [0])[0] if isinstance(info.get('foundation_count'), np.ndarray) else info.get('foundation_count', 0)
            if int(foundation_count_val) >= 1:  # Win condition (changed from 2 to 1)
                wins += 1

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        print(f"\nEvaluation Results:")
        print(f"Win Rate: {wins/n_episodes:.2%}")
        print(f"Average Reward: {np.mean(rewards):.2f}")

        eval_env.close()


def plot_training_results(rewards: List[float], lengths: List[float], save_path: str = None):
    """
    Plot training results.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    ax1.plot(rewards, alpha=0.3, color='blue')
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), color='red', linewidth=2)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)

    # Plot lengths
    ax2.plot(lengths, alpha=0.3, color='green')
    ax2.plot(np.convolve(lengths, np.ones(100)/100, mode='valid'), color='red', linewidth=2)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN agent for Spider Solitaire')
    parser.add_argument('--record', action='store_true', help='Enable episode recording')
    parser.add_argument('--record-dir', type=str, default='replays', help='Directory to save episode replays')
    args = parser.parse_args()

    # Create environment with ActionMasker wrapper
    env = ActionMasker(SpiderSolitaireEnv(max_steps=500))

    # Create agent with simplified network
    agent = SimpleDQNAgent(
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,  # Smaller buffer
        batch_size=32,
        target_update_freq=1000,
        record_episodes=args.record,
        record_dir=args.record_dir,
    )

    # Train
    num_episodes = 10000
    rewards, lengths = agent.train(num_episodes, save_freq=1000)

    # Save final model
    agent.save_model("simple_dqn_faceup_final.pt")

    # Plot results
    plot_training_results(rewards, lengths, "simple_dqn_faceup_training_results.png")

    # Evaluate
    print("\nEvaluating final model...")
    agent.evaluate(n_episodes=10, render=False)

    # Print final training statistics
    print("\n" + "="*60)
    print("Final Training Statistics:")
    print(f"Total episodes: {agent.total_episodes}")
    print(f"Total wins: {agent.wins} ({agent.wins/agent.total_episodes:.2%})")
    print(f"Average reward (last 1000): {np.mean(agent.episode_rewards[-1000:]):.2f}")
    print(f"Average episode length (last 1000): {np.mean(agent.episode_lengths[-1000:]):.2f}")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
