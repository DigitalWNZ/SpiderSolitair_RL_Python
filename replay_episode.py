"""
Episode Replay System for Spider Solitaire RL
Logs and replays episodes step-by-step for analysis.
"""
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any


class EpisodeRecorder:
    """Records episode steps for later replay and analysis."""

    def __init__(self, save_dir="replays"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_episode = []
        self.episode_metadata = {}

    def start_episode(self, algorithm: str, episode_num: int):
        """Start recording a new episode."""
        self.current_episode = []
        self.episode_metadata = {
            'algorithm': algorithm,
            'episode_num': episode_num,
            'start_time': datetime.now().isoformat(),
            'total_reward': 0,
            'steps': 0,
            'game_result': 'UNKNOWN'
        }

    def record_step(self, step_num: int, state: Dict, action: int,
                   reward: float, next_state: Dict, done: bool, info: Dict):
        """Record a single step."""
        # Decode action
        action_type = action // (10 * 10 * 13)
        remainder = action % (10 * 10 * 13)
        from_col = remainder // (10 * 13)
        remainder = remainder % (10 * 13)
        to_col = remainder // 13
        num_cards = remainder % 13

        step_data = {
            'step': step_num,
            'action': int(action),
            'action_decoded': {
                'type': 'move' if action_type == 0 else 'deal',
                'from_col': int(from_col) if action_type == 0 else None,
                'to_col': int(to_col) if action_type == 0 else None,
                'num_cards': int(num_cards) if action_type == 0 else None
            },
            'reward': float(reward),
            'cumulative_reward': self.episode_metadata['total_reward'] + float(reward),
            'foundation_count': int(info.get('foundation_count', 0)),
            'valid_moves': int(info.get('valid_moves', 0)),
            'done': bool(done)
        }

        self.current_episode.append(step_data)
        self.episode_metadata['total_reward'] += float(reward)
        self.episode_metadata['steps'] = step_num + 1

    def end_episode(self, game_result: str):
        """Finish recording and save episode."""
        self.episode_metadata['game_result'] = game_result
        self.episode_metadata['end_time'] = datetime.now().isoformat()

        # Save to file
        filename = f"{self.episode_metadata['algorithm']}_ep{self.episode_metadata['episode_num']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.save_dir, filename)

        data = {
            'metadata': self.episode_metadata,
            'steps': self.current_episode
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Episode saved to: {filepath}")
        return filepath


class EpisodePlayer:
    """Plays back recorded episodes."""

    def __init__(self, replay_file: str):
        with open(replay_file, 'r') as f:
            self.data = json.load(f)
        self.metadata = self.data['metadata']
        self.steps = self.data['steps']

    def print_summary(self):
        """Print episode summary."""
        print("=" * 80)
        print("EPISODE SUMMARY")
        print("=" * 80)
        print(f"Algorithm: {self.metadata['algorithm']}")
        print(f"Episode: {self.metadata['episode_num']}")
        print(f"Result: {self.metadata['game_result']}")
        print(f"Total Steps: {self.metadata['steps']}")
        print(f"Total Reward: {self.metadata['total_reward']:.2f}")
        print(f"Avg Reward/Step: {self.metadata['total_reward'] / max(1, self.metadata['steps']):.2f}")
        print("=" * 80)

    def print_step(self, step_num: int, verbose=True):
        """Print details of a specific step."""
        if step_num >= len(self.steps):
            print(f"Step {step_num} does not exist (max: {len(self.steps) - 1})")
            return

        step = self.steps[step_num]

        print(f"\n{'='*60}")
        print(f"STEP {step['step'] + 1}")
        print(f"{'='*60}")

        action = step['action_decoded']
        if action['type'] == 'move':
            print(f"Action: Move {action['num_cards']} cards from column {action['from_col']} to column {action['to_col']}")
        else:
            print(f"Action: Deal from stock")

        print(f"Reward: {step['reward']:+.2f}")
        print(f"Cumulative Reward: {step['cumulative_reward']:.2f}")
        print(f"Foundation Count: {step['foundation_count']}/1")
        print(f"Valid Moves Available: {step['valid_moves']}")
        print(f"Done: {step['done']}")

    def play_all(self, start=0, end=None, verbose=True):
        """Play through all steps."""
        if end is None:
            end = len(self.steps)

        self.print_summary()

        for i in range(start, min(end, len(self.steps))):
            self.print_step(i, verbose)
            if not verbose and (i + 1) % 50 == 0:
                print(f"\n... Processed {i + 1} steps ...")

    def analyze_rewards(self):
        """Analyze reward distribution."""
        rewards = [s['reward'] for s in self.steps]

        print("\n" + "=" * 80)
        print("REWARD ANALYSIS")
        print("=" * 80)
        print(f"Total Rewards: {sum(rewards):.2f}")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {max(rewards):.2f}")
        print(f"Min Reward: {min(rewards):.2f}")
        print(f"Std Deviation: {np.std(rewards):.2f}")

        # Count reward ranges
        positive = sum(1 for r in rewards if r > 0)
        negative = sum(1 for r in rewards if r < 0)
        zero = sum(1 for r in rewards if r == 0)

        print(f"\nReward Distribution:")
        print(f"  Positive: {positive} ({positive/len(rewards)*100:.1f}%)")
        print(f"  Negative: {negative} ({negative/len(rewards)*100:.1f}%)")
        print(f"  Zero: {zero} ({zero/len(rewards)*100:.1f}%)")

        # Top rewarding steps
        top_steps = sorted(enumerate(rewards), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 Rewarding Steps:")
        for idx, reward in top_steps:
            action = self.steps[idx]['action_decoded']
            if action['type'] == 'move':
                print(f"  Step {idx+1}: {reward:+.2f} - Move {action['num_cards']} cards (col{action['from_col']}→col{action['to_col']})")
            else:
                print(f"  Step {idx+1}: {reward:+.2f} - Deal from stock")


def record_agent_episodes(algorithm='DQN', num_episodes=5, max_steps=500):
    """Record episodes from a trained agent."""

    # Import here to avoid circular dependency
    import torch
    from spider_solitaire_masked_env_faceup import create_masked_faceup_env
    from train_dqn_simple_faceup import SimpleDQNAgent

    recorder = EpisodeRecorder()
    env = create_masked_faceup_env(max_steps=max_steps)

    # Load agent
    if algorithm == 'DQN':
        agent = SimpleDQNAgent(env, device='cpu')
        model_path = 'models/simple_dqn_faceup_best.pt'
        if os.path.exists(model_path):
            agent.q_network.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded DQN model from {model_path}")
        agent.epsilon = 0.1  # Low epsilon for mostly greedy actions

    elif algorithm == 'A2C':
        from train_a2c_simple_faceup import train_simple_a2c
        # For A2C, we'd need to instantiate differently
        print("A2C recording not yet implemented - use DQN for now")
        return

    print(f"\nRecording {num_episodes} episodes for {algorithm}...")
    print("=" * 80)

    for episode in range(num_episodes):
        recorder.start_episode(algorithm, episode)

        state, info = env.reset()
        done = False
        step = 0

        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")

        while not done and step < max_steps:
            # Select action
            mask = state.get('action_mask', None)
            action = agent.select_action(state, mask)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record
            recorder.record_step(step, state, action, reward, next_state, done, info)

            state = next_state
            step += 1

        # Determine result
        foundation_count = info.get('foundation_count', 0)
        if foundation_count >= 1:
            game_result = 'WON'
        elif truncated:
            game_result = 'TRUNCATED'
        else:
            game_result = 'LOST'

        filepath = recorder.end_episode(game_result)
        print(f"[Episode {episode + 1}/{num_episodes}] {game_result} - Steps: {step}, Reward: {recorder.episode_metadata['total_reward']:.2f}")

    print("\n" + "=" * 80)
    print(f"All episodes recorded to: {recorder.save_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Episode Replay System')
    parser.add_argument('--mode', choices=['record', 'play'], default='record',
                       help='Record new episodes or play existing replay')
    parser.add_argument('--algorithm', choices=['DQN', 'A2C', 'PPO'], default='DQN',
                       help='Algorithm to use for recording')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to record')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--replay-file', type=str,
                       help='Path to replay file (for play mode)')
    parser.add_argument('--analyze', action='store_true',
                       help='Show reward analysis')

    args = parser.parse_args()

    if args.mode == 'record':
        record_agent_episodes(
            algorithm=args.algorithm,
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )

    elif args.mode == 'play':
        if not args.replay_file:
            print("Error: --replay-file required for play mode")
            print("\nAvailable replays:")
            if os.path.exists('replays'):
                for f in sorted(os.listdir('replays')):
                    if f.endswith('.json'):
                        print(f"  replays/{f}")
        else:
            player = EpisodePlayer(args.replay_file)
            player.play_all()
            if args.analyze:
                player.analyze_rewards()
