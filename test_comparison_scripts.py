#!/usr/bin/env python3
"""
Test script to verify the fixed comparison scripts work correctly.
"""

import subprocess
import sys
import time

def test_quick_compare_fixed():
    """Test quick_compare_fixed.py with minimal episodes."""
    print("="*60)
    print("Testing quick_compare_fixed.py with 2 episodes...")
    print("="*60)
    
    try:
        # Run with just 2 episodes for quick test
        result = subprocess.run(
            [sys.executable, "quick_compare_fixed.py", "--episodes", "2", "--max-steps", "100"],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print("✓ quick_compare_fixed.py completed successfully!")
            print("\nOutput preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("✗ quick_compare_fixed.py failed!")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ quick_compare_fixed.py timed out!")
        return False
    except Exception as e:
        print(f"✗ Error running quick_compare_fixed.py: {e}")
        return False
    
    return True


def test_compare_algorithms_fixed():
    """Test compare_algorithms_fixed.py with minimal timesteps."""
    print("\n" + "="*60)
    print("Testing compare_algorithms_fixed.py with 1000 timesteps...")
    print("="*60)
    
    try:
        # Run with just 1000 timesteps and only DQN for quick test
        result = subprocess.run(
            [sys.executable, "compare_algorithms_fixed.py", 
             "--timesteps", "1000", 
             "--max-steps", "100",
             "--algorithms", "dqn"],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print("✓ compare_algorithms_fixed.py completed successfully!")
            print("\nOutput preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("✗ compare_algorithms_fixed.py failed!")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ compare_algorithms_fixed.py timed out!")
        return False
    except Exception as e:
        print(f"✗ Error running compare_algorithms_fixed.py: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Testing Fixed Comparison Scripts")
    print("="*60)
    
    # Test environment import first
    print("Testing environment imports...")
    try:
        from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
        from spider_solitaire_masked_env_fixed import MaskedSpiderSolitaireEnvFixed, ActionMasker
        print("✓ Fixed environments imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import fixed environments: {e}")
        return
    
    # Test quick_compare_fixed.py
    quick_compare_success = test_quick_compare_fixed()
    
    # Test compare_algorithms_fixed.py
    compare_algorithms_success = test_compare_algorithms_fixed()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"quick_compare_fixed.py: {'✓ PASSED' if quick_compare_success else '✗ FAILED'}")
    print(f"compare_algorithms_fixed.py: {'✓ PASSED' if compare_algorithms_success else '✗ FAILED'}")
    
    if quick_compare_success and compare_algorithms_success:
        print("\nAll tests passed! The fixed comparison scripts are working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()