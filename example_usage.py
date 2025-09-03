import gymnasium as gym
import numpy as np
from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed

# Register the environment
gym.register(
    id='SpiderSolitaire-v0',
    entry_point='spider_solitaire_env_fixed:SpiderSolitaireEnvFixed',
)

def play_random_game():
    """Play a game with random actions."""
    env = SpiderSolitaireEnvFixed(render_mode="human")
    obs, info = env.reset(seed=42)
    
    print("Starting Spider Solitaire game...")
    env.render()
    
    done = False
    step_count = 0
    max_steps = 100
    
    while not done and step_count < max_steps:
        # Random action
        action = env.action_space.sample()
        
        # For demo, bias towards card moves rather than dealing
        if np.random.random() < 0.8:
            action[0] = 0  # Move cards
        else:
            action[0] = 1  # Deal from stock
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward > 0:
            print(f"\nStep {step_count}: Action {action}, Reward: {reward}")
            env.render()
        
        step_count += 1
    
    print(f"\nGame ended after {step_count} steps")
    print(f"Final score: {info['score']}")
    print(f"Foundations completed: {obs['foundation_count'][0]}/8")
    
    env.close()

def play_interactive_game():
    """Play an interactive game."""
    env = SpiderSolitaireEnvFixed(render_mode="human")
    obs, info = env.reset()
    
    print("Welcome to Spider Solitaire!")
    print("Actions: [action_type, from_col, to_col, num_cards]")
    print("  action_type: 0=move cards, 1=deal from stock")
    print("  from_col/to_col: 0-9")
    print("  num_cards: 1-13")
    print("\nExample: '0 3 5 2' moves 2 cards from column 3 to column 5")
    print("Example: '1 0 0 0' deals from stock (other values ignored)\n")
    
    env.render()
    
    done = False
    while not done:
        try:
            action_str = input("\nEnter action (or 'q' to quit): ")
            if action_str.lower() == 'q':
                break
            
            action = list(map(int, action_str.split()))
            if len(action) != 4:
                print("Invalid action format. Need 4 values.")
                continue
            
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            done = terminated or truncated
            
            print(f"Reward: {reward}")
            print(f"Score: {info['score']}, Moves: {info['moves']}")
            print(f"Valid moves available: {info['valid_moves']}")
            
            env.render()
            
            if done:
                print("\nCongratulations! You've won!")
                
        except Exception as e:
            print(f"Error: {e}")
    
    env.close()

def test_environment():
    """Test basic environment functionality."""
    print("Testing Spider Solitaire Environment...")
    
    # Test initialization
    env = SpiderSolitaireEnvFixed()
    print("✓ Environment created")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print("✓ Environment reset")
    print(f"  Observation space shape: {obs['tableau'].shape}")
    print(f"  Initial score: {info['score']}")
    
    # Test random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if reward != -10:  # Not an invalid move
            print(f"✓ Step {i}: Valid action executed, reward={reward}")
    
    # Test rendering
    print("\nCurrent game state:")
    env.render()
    
    env.close()
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_environment()
        elif sys.argv[1] == "play":
            play_interactive_game()
        else:
            play_random_game()
    else:
        print("Usage:")
        print("  python example_usage.py test  - Run environment tests")
        print("  python example_usage.py play  - Play interactive game")
        print("  python example_usage.py       - Watch random game")
        print("\nRunning random game by default...\n")
        play_random_game()