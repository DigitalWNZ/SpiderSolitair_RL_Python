##########################################################################################
# 网络结构：
# 	- tableau_cnn: 与 A2C 相同的卷积结构
# 	- 3层卷积处理 10×19 牌局
# 	- 32→64→128 通道
# - 全连接层 (fc)：
# 	- 合并 CNN 特征、库存牌数、基础牌数
# 	- 三层全连接：512→256→动作数
# - 输出：每个动作的 Q 值

# 2. ReplayBuffer 经验回放缓冲区 (第75-91行)
# 	DQN 的核心组件之一：
# 	- 存储过往经验 (state, action, reward, next_state, done)
# 	- 支持随机采样，打破数据相关性
# 	- 使用 deque 实现固定容量的循环缓冲区

# 3. DQNAgent 主要类 (第93-310行)

# 初始化参数：
# 	- learning_rate: 学习率（1e-4）
# 	- gamma: 折扣因子（0.99）
# 	- epsilon_start/end: ε-贪婪策略参数（1.0→0.01）
# 	- epsilon_decay: ε衰减率（0.995）
# 	- buffer_size: 经验缓冲区大小（100000）
# 	- batch_size: 批次大小（32）
# 	- target_update_freq: 目标网络更新频率（1000步）

# 双网络架构：
# 	- q_network: 主网络，用于选择动作和训练
# 	- target_network: 目标网络，用于计算目标 Q 值，定期同步
# 	- 这两个网络初始是一样的结构

# 核心方法：

# select_action (第139-164行)：
# 	ε-贪婪策略选择动作：
# 		if random < ε:
# 		  选择随机动作（考虑动作掩码）
# 		else:
# 		  调用q_network计算（s,a）的Q值，返回Q值最大的动作

# train_step (第166-208行)：
# 	1. 从replayBuffer区采样一批数据,放到state_dict和next_state_dict中
# 	2. 使用q_network计算当前状态state的Q值：Q(s,a)
# 	3. 使用target_network计算目标下一状态next_state的最大Q值：max Q'(s',a')
# 	4. 目标Q值 target_q_value = r + γ * max Q'(s',a')
# 	5. 计算 MSE 损失并更新网络
# 	6. 使用梯度裁剪防止梯度爆炸

# train (第209-279行)
# 	单环境顺序训练：
# 	1. 每个 episode：
# 		- 使用 ε-贪婪策略选择动作
# 		- 执行动作，存储经验到replayBuffer中
# 		- 每步都进行训练更新
# 		- 定期更新目标网络
# 	2. 逐渐衰减 ε 值
# 	3. 记录和保存模型
##########################################################################################
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

from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
from spider_solitaire_masked_env_fixed import MaskedSpiderSolitaireEnvFixed, ActionMasker


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Spider Solitaire.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Space):
        super(DQNNetwork, self).__init__()
        
        # Tableau CNN
        self.tableau_cnn = nn.Sequential(
            nn.Unflatten(1, (1, 10)),
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_tableau = torch.zeros(1, 10, 19)
            cnn_output_dim = self.tableau_cnn(sample_tableau).shape[1]
        
        # Combine all features
        total_features = cnn_output_dim + 2  # +2 for stock_count and foundation_count
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n if hasattr(action_space, 'n') else action_space.nvec.prod()),
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


class DQNAgent:
    """
    Deep Q-Learning agent for Spider Solitaire.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(env.observation_space, env.action_space).to(self.device)
        self.target_network = DQNNetwork(env.observation_space, env.action_space).to(self.device)
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
        print(f"Training DQN on {self.device}")
        steps = 0
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                mask = state.get('action_mask', None)
                action = self.select_action(state, mask)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
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
            
            if episode_reward > 0:  # Win condition
                self.wins += 1
            
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
                
                # Print detailed episode info
                if 'game_result' in info:
                    print(f"  Game Result: {info.get('game_result', 'unknown')}")
                    print(f"  Valid Moves: {info.get('valid_moves', 0)}")
                    print(f"  Invalid Moves: {info.get('invalid_moves', 0)}")
                    print(f"  Foundation Count: {info.get('foundation_count', 0)}")
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f"dqn_spider_{episode}.pt")
        
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
    # Create environment
    use_masked = True
    if use_masked:
        env = ActionMasker(MaskedSpiderSolitaireEnvFixed())
    else:
        env = SpiderSolitaireEnvFixed()
    
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
    
    # Train
    num_episodes = 10000
    rewards, lengths = agent.train(num_episodes, save_freq=1000)
    
    # Save final model
    agent.save_model("dqn_spider_final.pt")
    
    # Plot results
    plot_training_results(rewards, lengths, "dqn_training_results.png")
    
    # Print final training statistics
    print("\n=== Final Training Statistics ===")
    print(f"Total episodes: {agent.total_episodes}")
    print(f"Total wins: {agent.wins}")
    print(f"Final win rate: {agent.wins / agent.total_episodes:.2%}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Average length (last 100 episodes): {np.mean(lengths[-100:]):.1f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    env.close()


if __name__ == "__main__":
    main()