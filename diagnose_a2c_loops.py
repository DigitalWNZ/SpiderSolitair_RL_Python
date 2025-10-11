"""
Diagnose why A2C gets stuck in back-and-forth loops.
Track detailed move patterns and penalties.
"""
import numpy as np
from spider_solitaire_masked_env_faceup import MaskedSpiderSolitaireEnvFaceup
from train_a2c_simple_faceup import SimpleA2CAgent
import torch

def analyze_a2c_behavior(episodes=5):
    """Run A2C and track move patterns."""

    # Create environment
    env = MaskedSpiderSolitaireEnvFaceup(max_steps=200)

    # Calculate action space size (masked env converts to discrete)
    action_space_size = env.action_space.nvec.prod()

    # Create agent
    agent = SimpleA2CAgent(
        observation_space=env.observation_space,
        action_space=action_space_size,
        device='cpu'
    )

    print("="*80)
    print("A2C LOOP DIAGNOSIS")
    print("="*80)

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        step = 0

        episode_reward = 0
        reverse_penalties = 0
        move_sequence = []

        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*80}")

        while not done and step < 200:
            # Get action from A2C
            state_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(agent.device)
                for k, v in state.items() if k != 'action_mask'
            }

            with torch.no_grad():
                policy_logits, _ = agent.network(state_tensor)

                # Apply mask
                if 'action_mask' in state:
                    mask = torch.FloatTensor(state['action_mask']).to(agent.device)
                    policy_logits = policy_logits.masked_fill(mask == 0, float('-inf'))

                # Sample action
                probs = torch.softmax(policy_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Decode action to see what move was made
            action_type = action // (10 * 10 * 13)
            remainder = action % (10 * 10 * 13)
            from_col = remainder // (10 * 13)
            remainder = remainder % (10 * 13)
            to_col = remainder // 13
            num_cards = remainder % 13

            # Track move
            if action_type == 0:  # Card move
                move_str = f"Move {num_cards} cards: col{from_col}→col{to_col}"
                move_sequence.append((from_col, to_col, num_cards))

                # Check for reverse
                if len(move_sequence) >= 2:
                    prev_from, prev_to, prev_num = move_sequence[-2]
                    curr_from, curr_to, curr_num = move_sequence[-1]

                    if prev_from == curr_to and prev_to == curr_from and prev_num == curr_num:
                        reverse_penalties += 1
                        print(f"  Step {step:3d}: {move_str} | Reward: {reward:6.1f} | ⚠️  REVERSE MOVE!")
                    else:
                        print(f"  Step {step:3d}: {move_str} | Reward: {reward:6.1f}")
                else:
                    print(f"  Step {step:3d}: {move_str} | Reward: {reward:6.1f}")
            else:
                print(f"  Step {step:3d}: Deal from stock | Reward: {reward:6.1f}")
                move_sequence = []  # Reset after dealing

            episode_reward += reward
            state = next_state
            step += 1

        print(f"\n{'-'*80}")
        print(f"Episode Summary:")
        print(f"  Total Reward: {episode_reward:.1f}")
        print(f"  Steps: {step}")
        print(f"  Reverse Penalties: {reverse_penalties}")
        print(f"  Estimated Penalty Loss: {reverse_penalties * -5:.1f}")
        print(f"  Game Result: {'WON' if info.get('foundation_count', 0) >= 1 else 'LOST/TRUNCATED'}")
        print(f"{'-'*80}")

        # Analyze last 10 moves for patterns
        if len(move_sequence) >= 5:
            print(f"\nLast 5 moves:")
            for i, (f, t, n) in enumerate(move_sequence[-5:]):
                print(f"  {i+1}. {n} cards: col{f}→col{t}")

if __name__ == "__main__":
    analyze_a2c_behavior(episodes=3)
