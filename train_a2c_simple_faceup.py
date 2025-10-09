import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import deque

from spide_solitaire_env_faceup import SpiderSolitaireEnv
from spider_solitaire_masked_env_faceup import ActionMasker


class SimpleA2CNetwork(nn.Module):
    """
    Simplified Actor-Critic network for Spider Solitaire.
    Uses a lighter architecture for faster training.
    """

    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Space):
        super(SimpleA2CNetwork, self).__init__()

        # Simplified CNN for tableau processing
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

        # Simplified shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
        )

        # Actor head (policy)
        action_dim = action_space.n if hasattr(action_space, 'n') else action_space.nvec.prod()
        self.actor = nn.Linear(128, action_dim)

        # Critic head (value)
        self.critic = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value estimate.
        """
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

        # Shared processing
        shared = self.shared_fc(combined)

        # Actor and critic outputs
        policy_logits = self.actor(shared)
        value = self.critic(shared)

        return policy_logits, value


class SimpleA2CAgent:
    """
    Simplified A2C agent with same interface as original but lighter network.
    """

    def __init__(
        self,
        env: gym.Env,
        n_envs: int = 4,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        device: str = None,
    ):
        self.env = env
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use simplified network
        self.network = SimpleA2CNetwork(env.observation_space, env.action_space).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Training metrics
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.wins = 0
        self.total_episodes = 0
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []

        # Detailed episode tracking
        self.episode_details = deque(maxlen=1000)  # Store detailed info for recent episodes
        self.env_episode_info = [{
            'start_time': datetime.now(),
            'steps': 0,
            'rewards': [],
            'actions': []
        } for _ in range(n_envs)]

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return returns, advantages

    def display_recent_episodes(self, n: int = 5):
        """
        Display details of the n most recent episodes.
        """
        if not self.episode_details:
            print("No completed episodes yet.")
            return

        print(f"\n=== Recent {min(n, len(self.episode_details))} Episodes ===")
        recent_episodes = list(self.episode_details)[-n:]

        for i, ep in enumerate(recent_episodes):
            print(f"\nEpisode {self.total_episodes - len(recent_episodes) + i + 1}:")
            print(f"  Reward: {ep['reward']:.2f}")
            print(f"  Length: {ep['length']} steps")
            print(f"  Win: {'Yes' if ep['win'] else 'No'}")
            print(f"  Duration: {ep['duration']:.2f}s")
            print(f"  Final moves taken: {ep['final_moves'][-10:]}")  # Last 10 moves
            if ep['foundation_count'] > 0:
                print(f"  Foundation cards: {ep['foundation_count']}")
            if ep['stock_count'] >= 0:
                print(f"  Remaining stock: {ep['stock_count']}")

        # Overall statistics
        all_rewards = [ep['reward'] for ep in self.episode_details]
        all_lengths = [ep['length'] for ep in self.episode_details]
        wins = sum(1 for ep in self.episode_details if ep['win'])

        print(f"\n=== Overall Statistics (last {len(self.episode_details)} episodes) ===")
        print(f"  Average reward: {np.mean(all_rewards):.2f}")
        print(f"  Average length: {np.mean(all_lengths):.1f}")
        print(f"  Win rate: {wins/len(self.episode_details):.2%}")
        print(f"  Best reward: {max(all_rewards):.2f}")
        print(f"  Worst reward: {min(all_rewards):.2f}")

    def collect_rollouts(self, states: List[Dict], n_steps: int) -> Dict:
        """
        Collect experience for n_steps.
        """
        rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': [],
        }

        for step in range(n_steps):
            # Convert states to tensor
            state_dict = {
                k: torch.FloatTensor(np.array([s[k] for s in states])).to(self.device)
                for k in states[0].keys() if k != 'action_mask'
            }

            # Get policy and value
            with torch.no_grad():
                policy_logits, values = self.network(state_dict)

                # Apply action mask if available
                if 'action_mask' in states[0]:
                    masks = torch.FloatTensor(
                        np.array([s['action_mask'] for s in states])
                    ).to(self.device)
                    policy_logits = policy_logits.masked_fill(masks == 0, float('-inf'))

                # Sample actions
                probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Store data
            rollout_data['states'].append(states)
            rollout_data['actions'].append(actions.cpu().numpy())
            rollout_data['values'].append(values.squeeze().cpu().numpy())
            rollout_data['log_probs'].append(log_probs.cpu().numpy())

            # Take actions in environment
            next_states = []
            rewards = []
            dones = []

            for i, (state, action) in enumerate(zip(states, actions.cpu().numpy())):
                if hasattr(self.env, 'envs'):  # Vectorized environment
                    next_state, reward, terminated, truncated, info = self.env.envs[i].step(action)
                else:  # Single environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                # Track episode info for each environment
                self.env_episode_info[i]['steps'] += 1
                self.env_episode_info[i]['rewards'].append(reward)
                self.env_episode_info[i]['actions'].append(int(action))

                # Track metrics when episode completes
                if done:
                    self.total_episodes += 1
                    episode_reward = info.get('episode_reward', sum(self.env_episode_info[i]['rewards']))
                    episode_length = info.get('episode_length', self.env_episode_info[i]['steps'])

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    is_win = episode_reward > 0  # Win condition
                    if is_win:
                        self.wins += 1

                    # Store detailed episode information
                    episode_detail = {
                        'reward': episode_reward,
                        'length': episode_length,
                        'win': is_win,
                        'duration': (datetime.now() - self.env_episode_info[i]['start_time']).total_seconds(),
                        'final_moves': self.env_episode_info[i]['actions'][-20:],  # Last 20 moves
                        'foundation_count': int(next_state.get('foundation_count', [0])[0] if isinstance(next_state.get('foundation_count'), np.ndarray) else next_state.get('foundation_count', 0)),
                        'stock_count': int(next_state.get('stock_count', [-1])[0] if isinstance(next_state.get('stock_count'), np.ndarray) else next_state.get('stock_count', -1))
                    }
                    self.episode_details.append(episode_detail)

                    # Reset episode tracking for this environment
                    self.env_episode_info[i] = {
                        'start_time': datetime.now(),
                        'steps': 0,
                        'rewards': [],
                        'actions': []
                    }

                    # Reset the environment after done
                    if hasattr(self.env, 'envs'):  # Vectorized environment
                        next_state, _ = self.env.envs[i].reset()
                    else:  # Single environment
                        next_state, _ = self.env.reset()
                    next_states[i] = next_state

            rollout_data['rewards'].append(rewards)
            rollout_data['dones'].append(dones)

            # Update states
            states = next_states

        # Get final value for GAE computation
        state_dict = {
            k: torch.FloatTensor(np.array([s[k] for s in states])).to(self.device)
            for k in states[0].keys() if k != 'action_mask'
        }
        with torch.no_grad():
            _, next_values = self.network(state_dict)
            rollout_data['next_value'] = next_values.squeeze().cpu().numpy()

        return rollout_data, states

    def train_step(self, rollout_data: Dict):
        """
        Perform one training update using collected rollout data.
        """
        # Convert to tensors
        states_batch = []
        for states in rollout_data['states']:
            state_dict = {
                k: torch.FloatTensor(np.array([s[k] for s in states]))
                for k in states[0].keys() if k != 'action_mask'
            }
            states_batch.append(state_dict)

        actions = torch.LongTensor(np.array(rollout_data['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(rollout_data['log_probs'])).to(self.device)
        rewards = torch.FloatTensor(np.array(rollout_data['rewards'])).to(self.device)
        values = torch.FloatTensor(np.array(rollout_data['values'])).to(self.device)
        dones = torch.FloatTensor(np.array(rollout_data['dones'])).to(self.device)
        next_value = torch.FloatTensor(rollout_data['next_value']).to(self.device)

        # Compute returns and advantages
        returns, advantages = self.compute_gae(
            rewards.flatten(),
            values.flatten(),
            dones.flatten(),
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Reshape for batch processing
        n_steps, n_envs = actions.shape
        batch_size = n_steps * n_envs

        returns = returns.reshape(batch_size)
        advantages = advantages.reshape(batch_size)
        actions = actions.reshape(batch_size)
        old_log_probs = old_log_probs.reshape(batch_size)

        # Forward pass for all states
        all_policy_logits = []
        all_values = []

        for state_dict in states_batch:
            # Move to device
            state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            policy_logits, values = self.network(state_dict)
            all_policy_logits.append(policy_logits)
            all_values.append(values)

        # Stack results
        policy_logits = torch.cat(all_policy_logits, dim=0)
        values = torch.cat(all_values, dim=0).squeeze()

        # Apply action mask if available
        if 'action_mask' in rollout_data['states'][0][0]:
            all_masks = []
            for states in rollout_data['states']:
                masks = torch.FloatTensor(
                    np.array([s['action_mask'] for s in states])
                ).to(self.device)
                all_masks.append(masks)
            masks = torch.cat(all_masks, dim=0)
            policy_logits = policy_logits.masked_fill(masks == 0, float('-inf'))

        # Compute losses
        # Actor loss
        probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -(advantages.detach() * log_probs).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, returns.detach())

        # Entropy loss
        entropy = dist.entropy().mean()
        entropy_loss = -entropy

        # Total loss
        loss = (
            actor_loss +
            self.value_coef * critic_loss +
            self.entropy_coef * entropy_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Store metrics
        self.losses.append(loss.item())
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy.item())

    def train(self, total_timesteps: int, log_interval: int = 100):
        """
        Train the A2C agent.
        """
        print(f"Training Simplified A2C on {self.device}")
        print("Network architecture: 2 Conv layers (16, 32 channels), 1 FC layer (128 neurons)")

        # Initialize environments
        if self.n_envs > 1:
            # Create multiple environments with ActionMasker wrapper
            envs = [ActionMasker(SpiderSolitaireEnv(max_steps=500)) for _ in range(self.n_envs)]
            states = [env.reset()[0] for env in envs]
            self.env.envs = envs  # Store for rollout collection
        else:
            states = [self.env.reset()[0]]

        num_updates = total_timesteps // (self.n_steps * self.n_envs)

        for update in range(num_updates):
            # Collect rollouts
            rollout_data, states = self.collect_rollouts(states, self.n_steps)

            # Train on rollouts
            self.train_step(rollout_data)

            # Logging
            if update % log_interval == 0 and self.total_episodes > 0:
                avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
                avg_length = np.mean(list(self.episode_lengths)) if self.episode_lengths else 0
                win_rate = self.wins / self.total_episodes if self.total_episodes > 0 else 0

                print(f"\n{'='*60}")
                print(f"Update {update}/{num_updates}, Episodes: {self.total_episodes}, "
                      f"Timesteps: {(update + 1) * self.n_steps * self.n_envs}")
                print(f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                      f"Win Rate: {win_rate:.2%}")

                if self.losses:
                    print(f"  Loss: {np.mean(self.losses[-100:]):.4f}, "
                          f"Actor: {np.mean(self.actor_losses[-100:]):.4f}, "
                          f"Critic: {np.mean(self.critic_losses[-100:]):.4f}, "
                          f"Entropy: {np.mean(self.entropy_losses[-100:]):.4f}")

                # Display recent episode details
                if update % (log_interval * 5) == 0:  # Every 5 log intervals
                    self.display_recent_episodes(n=5)

            # Save checkpoint
            if update % 1000 == 0 and update > 0:
                self.save_model(f"simple_a2c_faceup_{update}.pt")

        # Display final statistics
        self.display_final_statistics()

        return self.episode_rewards, self.episode_lengths

    def display_final_statistics(self):
        """
        Display comprehensive statistics at the end of training.
        """
        print(f"\n{'='*80}")
        print("FINAL TRAINING STATISTICS")
        print(f"{'='*80}")

        print(f"\nTotal episodes completed: {self.total_episodes}")
        print(f"Total wins: {self.wins}")
        print(f"Overall win rate: {self.wins/self.total_episodes:.2%}" if self.total_episodes > 0 else "N/A")

        if self.episode_rewards:
            all_rewards = list(self.episode_rewards)
            all_lengths = list(self.episode_lengths)

            print(f"\nReward Statistics:")
            print(f"  Mean: {np.mean(all_rewards):.2f}")
            print(f"  Std: {np.std(all_rewards):.2f}")
            print(f"  Min: {np.min(all_rewards):.2f}")
            print(f"  Max: {np.max(all_rewards):.2f}")
            print(f"  Median: {np.median(all_rewards):.2f}")

            print(f"\nEpisode Length Statistics:")
            print(f"  Mean: {np.mean(all_lengths):.1f}")
            print(f"  Std: {np.std(all_lengths):.1f}")
            print(f"  Min: {np.min(all_lengths)}")
            print(f"  Max: {np.max(all_lengths)}")

            # Recent performance (last 100 episodes)
            if len(all_rewards) >= 100:
                recent_rewards = all_rewards[-100:]
                recent_wins = sum(1 for r in recent_rewards if r > 0)
                print(f"\nLast 100 Episodes:")
                print(f"  Average reward: {np.mean(recent_rewards):.2f}")
                print(f"  Win rate: {recent_wins/100:.2%}")

        # Display last 10 episodes in detail
        print(f"\n{'='*80}")
        print("LAST 10 EPISODES DETAIL")
        print(f"{'='*80}")
        self.display_recent_episodes(n=10)

    def save_model(self, filename: str):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'wins': self.wins,
            'total_episodes': self.total_episodes,
            'episode_details': list(self.episode_details)[-100:],  # Save last 100 episodes
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=1000)
        self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=1000)
        self.wins = checkpoint['wins']
        self.total_episodes = checkpoint['total_episodes']

        # Load episode details if available
        if 'episode_details' in checkpoint:
            self.episode_details = deque(checkpoint['episode_details'], maxlen=1000)

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

                # Get action
                with torch.no_grad():
                    policy_logits, _ = self.network(state_dict)

                    # Apply mask if available
                    if 'action_mask' in state:
                        mask = torch.FloatTensor(state['action_mask']).to(self.device)
                        policy_logits = policy_logits.masked_fill(mask == 0, float('-inf'))

                    probs = F.softmax(policy_logits, dim=-1)
                    action = probs.argmax().item()

                # Take action
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if render:
                    eval_env.render()

            rewards.append(episode_reward)
            if episode_reward > 0:
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
    if len(rewards) > 100:
        ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), color='red', linewidth=2)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)

    # Plot lengths
    ax2.plot(lengths, alpha=0.3, color='green')
    if len(lengths) > 100:
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
    # Create environment with ActionMasker wrapper
    env = ActionMasker(SpiderSolitaireEnv(max_steps=500))

    # Create agent with simplified network
    agent = SimpleA2CAgent(
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

    # Record training start time
    training_start_time = datetime.now()

    # Train
    total_timesteps = 1_000_000
    rewards, lengths = agent.train(total_timesteps, log_interval=100)

    # Calculate training duration
    training_duration = (datetime.now() - training_start_time).total_seconds()

    # Save final model
    agent.save_model("simple_a2c_faceup_final.pt")

    # Plot results
    plot_training_results(list(rewards), list(lengths), "simple_a2c_faceup_training_results.png")

    # Print training summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Training duration: {training_duration/60:.1f} minutes")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Timesteps per second: {total_timesteps/training_duration:.1f}")
    print(f"Episodes per minute: {agent.total_episodes/(training_duration/60):.1f}")

    # Evaluate
    print("\nEvaluating final model...")
    agent.evaluate(n_episodes=10, render=False)

    env.close()


if __name__ == "__main__":
    main()
