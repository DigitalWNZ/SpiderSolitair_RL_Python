#!/usr/bin/env python3
"""
Quick test script to verify quick_compare_simple.py with 2 episodes
"""

# Import the main function from quick_compare_simple
from quick_compare_simple import main

if __name__ == "__main__":
    print("Testing quick_compare_simple.py with 2 episodes...")
    print("="*60)
    
    # Run with just 2 episodes for quick verification
    main(episodes=2)
    
    print("\n" + "="*60)
    print("Test completed!")