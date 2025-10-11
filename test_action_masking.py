"""
Test script to verify action masking is working correctly.
"""
import numpy as np
from spider_solitaire_masked_env_faceup import create_masked_faceup_env

def test_action_masking():
    """Test that action masking properly restricts actions."""
    env = create_masked_faceup_env(max_steps=500)

    state, info = env.reset()

    print("="*60)
    print("ACTION MASKING TEST")
    print("="*60)

    # Check if action_mask is in observation
    if 'action_mask' not in state:
        print("ERROR: action_mask not in observation!")
        return False

    mask = state['action_mask']
    print(f"\nAction mask shape: {mask.shape}")
    print(f"Total actions: {len(mask)}")
    print(f"Valid actions: {mask.sum()}")
    print(f"Invalid actions: {(mask == 0).sum()}")
    print(f"Valid action percentage: {(mask.sum() / len(mask)) * 100:.4f}%")

    # Test taking only valid actions
    print("\n" + "="*60)
    print("Testing 50 steps with ONLY valid actions:")
    print("="*60)

    valid_action_count = 0
    invalid_action_count = 0
    rewards = []

    for step in range(50):
        mask = state['action_mask']
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) == 0:
            print(f"Step {step}: No valid actions available!")
            break

        # Take a random VALID action
        action = np.random.choice(valid_actions)

        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # Check if we got an invalid move penalty
        if reward == -10:
            invalid_action_count += 1
            print(f"Step {step}: Got -10 penalty despite using mask! Action: {action}")
        else:
            valid_action_count += 1

        state = next_state

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nResults:")
    print(f"  Valid moves taken: {valid_action_count}")
    print(f"  Invalid moves (shouldn't happen): {invalid_action_count}")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")

    if invalid_action_count > 0:
        print(f"\nWARNING: Action masking is NOT working correctly!")
        print(f"  {invalid_action_count} invalid actions were taken despite using masks")
        return False
    else:
        print(f"\nSUCCESS: Action masking is working correctly!")
        print(f"  All {valid_action_count} actions were valid")
        return True

if __name__ == "__main__":
    success = test_action_masking()
    exit(0 if success else 1)
