#!/usr/bin/env python3
"""
Test script to verify the fixed environment works correctly.
"""

from spider_solitaire_masked_env_fixed import create_masked_env, MaskedSpiderSolitaireEnvFixed, ActionMasker
import numpy as np

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Fixed Spider Solitaire Environment")
    print("="*50)
    
    # Test 1: Create environment
    print("\n1. Creating masked environment...")
    try:
        env = create_masked_env(max_steps=100)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test 2: Reset environment
    print("\n2. Resetting environment...")
    try:
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Action mask shape: {obs['action_mask'].shape}")
        print(f"  - Valid actions: {obs['action_mask'].sum()}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return False
    
    # Test 3: Take a few steps
    print("\n3. Taking random valid actions...")
    try:
        for i in range(5):
            # Get valid actions
            valid_actions = np.where(obs['action_mask'] > 0)[0]
            if len(valid_actions) == 0:
                print(f"  Step {i+1}: No valid actions available")
                break
            
            # Take random valid action
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {i+1}: Action {action}, Reward {reward:.1f}, "
                  f"Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print(f"  Episode ended: {'Won!' if terminated else 'Truncated'}")
                break
                
    except Exception as e:
        print(f"✗ Failed during stepping: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed!")
    return True


def test_truncation():
    """Test that truncation works correctly."""
    print("\n\nTesting Truncation Behavior")
    print("="*50)
    
    # Create environment with very low max steps
    env = create_masked_env(max_steps=10)
    obs, info = env.reset()
    
    print("Running episode with max_steps=10...")
    steps = 0
    done = False
    
    while not done:
        valid_actions = np.where(obs['action_mask'] > 0)[0]
        if len(valid_actions) == 0:
            print(f"No valid actions at step {steps}")
            break
            
        action = valid_actions[0]  # Take first valid action
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        
        print(f"Step {steps}: {'Truncated' if truncated else 'Continuing'}")
        
        if done:
            print(f"\n✓ Episode ended after {steps} steps")
            print(f"  Reason: {'Won' if terminated else 'Truncated (max steps reached)'}")
            break
    
    env.close()


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        test_truncation()
    
    print("\nTest complete!")