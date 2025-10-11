"""
Diagnose why agents aren't completing sequences.
"""
import numpy as np
from spider_solitaire_masked_env_faceup import create_masked_faceup_env

def diagnose_random_play():
    """Test with random valid actions to see baseline."""
    env = create_masked_faceup_env(max_steps=1000)

    print("="*60)
    print("BASELINE TEST: Random Valid Actions (10 episodes)")
    print("="*60)

    sequences_per_episode = []
    max_seq_lengths = []

    for episode in range(10):
        state, info = env.reset()
        done = False
        step = 0

        while not done and step < 1000:
            mask = state['action_mask']
            valid_actions = np.where(mask > 0)[0]

            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        foundation_count = int(state.get('foundation_count', [0])[0] if isinstance(state.get('foundation_count'), np.ndarray) else state.get('foundation_count', 0))
        sequences_per_episode.append(foundation_count)

        # Check max sequence length in tableau
        max_seq = 0
        for col in range(10):
            seq_len = 1
            for i in range(len(env.env.tableau[col]) - 1, 0, -1):
                curr = env.env.tableau[col][i]
                prev = env.env.tableau[col][i-1]
                if (curr['face_up'] and prev['face_up'] and
                    prev['rank'] == curr['rank'] + 1):
                    seq_len += 1
                else:
                    break
            max_seq = max(max_seq, seq_len)
        max_seq_lengths.append(max_seq)

        print(f"Episode {episode + 1}: Steps: {step}, "
              f"Sequences: {foundation_count}/4, "
              f"Max sequence length: {max_seq}")

    print(f"\n{'='*60}")
    print(f"Average sequences completed: {np.mean(sequences_per_episode):.2f}/4")
    print(f"Max sequences in any episode: {np.max(sequences_per_episode)}/4")
    print(f"Average max sequence length: {np.mean(max_seq_lengths):.1f}")
    print(f"Best sequence length achieved: {np.max(max_seq_lengths)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    diagnose_random_play()
