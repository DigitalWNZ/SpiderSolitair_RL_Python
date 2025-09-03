"""
Summary of environment usage across training scripts:

All training scripts have been updated to use the fixed environments:

1. train_dqn_simple.py:
   - Uses: ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - Fixed: Now uses fixed environment
   - Action masking: YES

2. train_a2c_simple.py:
   - Uses: ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - Fixed: Now uses fixed environment
   - Action masking: YES

3. train_ppo_simple.py:
   - Uses: SpiderSolitaireEnvFixed()
   - Fixed: Now uses fixed environment
   - Action masking: NO (PPO handles invalid actions internally)

4. train_dqn.py:
   - Uses: ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - Fixed: Now uses fixed environment
   - Action masking: YES

5. train_a2c.py:
   - Uses: ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - Fixed: Now uses fixed environment
   - Action masking: YES

6. train_ppo.py:
   - Uses: SpiderSolitaireEnvFixed()
   - Fixed: Now uses fixed environment
   - Action masking: NO (PPO handles invalid actions internally)

7. quick_compare_simple_fixed.py:
   - DQN: Uses create_masked_env() -> ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - A2C: Uses create_masked_env() -> ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - PPO: Uses SpiderSolitaireEnvFixed() directly via train_ppo_simple.py
   
8. quick_compare_fixed.py:
   - DQN: Uses ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - A2C: Uses ActionMasker(MaskedSpiderSolitaireEnvFixed())
   - PPO: Uses SpiderSolitaireEnvFixed() directly via train_ppo.py

The ActionMasker wrapper bug has been fixed in spider_solitaire_masked_env_fixed.py
and all training scripts now use the fixed environments.
"""

def check_imports():
    """Check which environments are imported in each file."""
    import os
    
    files = [
        'train_dqn_simple.py',
        'train_a2c_simple.py', 
        'train_ppo_simple.py',
        'train_dqn.py',
        'train_a2c.py',
        'train_ppo.py',
        'quick_compare_simple_fixed.py',
        'quick_compare_fixed.py',
        'example_usage.py',
        'visualize_training.py'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"\n{file}:")
            with open(file, 'r') as f:
                for line in f:
                    if 'from spider_solitaire' in line and 'import' in line:
                        print(f"  {line.strip()}")

if __name__ == "__main__":
    check_imports()