"""
Quick demo to record and replay a single episode.
"""
from replay_episode import EpisodeRecorder, EpisodePlayer, record_agent_episodes
import os

def demo():
    print("="*80)
    print("SPIDER SOLITAIRE EPISODE REPLAY DEMO")
    print("="*80)

    # Record 3 episodes
    print("\n1. Recording 3 episodes...")
    record_agent_episodes(algorithm='DQN', num_episodes=3, max_steps=500)

    # Find the most recent replay
    replay_files = sorted([f for f in os.listdir('replays') if f.endswith('.json')])
    if replay_files:
        latest_replay = os.path.join('replays', replay_files[-1])

        print(f"\n2. Playing back most recent episode: {latest_replay}")
        print("="*80)

        player = EpisodePlayer(latest_replay)

        # Show summary
        player.print_summary()

        # Show first 10 steps
        print("\n3. First 10 steps:")
        for i in range(min(10, len(player.steps))):
            player.print_step(i, verbose=True)

        # Show reward analysis
        print("\n4. Reward Analysis:")
        player.analyze_rewards()

    else:
        print("No replay files found!")

if __name__ == "__main__":
    demo()
