import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any
from spider_solitaire_env import SpiderSolitaireEnv


class MaskedSpiderSolitaireEnv(SpiderSolitaireEnv):
    """
    Spider Solitaire environment with action masking to prevent invalid moves.
    This improves training efficiency by preventing the agent from trying invalid actions.
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        
        # Add action mask to observation space
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(low=0, high=26, shape=(10, 19), dtype=np.int8),
            'stock_count': spaces.Box(low=0, high=5, shape=(1,), dtype=np.int8),
            'foundation_count': spaces.Box(low=0, high=8, shape=(1,), dtype=np.int8),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space.nvec.prod(),), dtype=np.int8),
        })
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed, options)
        obs['action_mask'] = self._get_action_mask()
        return obs, info
    
    def step(self, action: np.ndarray):
        # Check if action is masked (invalid)
        action_idx = self._action_to_index(action)
        action_mask = self._get_action_mask()
        
        if action_mask[action_idx] == 0:
            # Invalid action - return penalty without changing state
            obs = self._get_obs()
            obs['action_mask'] = action_mask
            return obs, -10, False, False, self._get_info()
        
        # Execute valid action
        obs, reward, terminated, truncated, info = super().step(action)
        obs['action_mask'] = self._get_action_mask()
        return obs, reward, terminated, truncated, info
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Generate action mask where 1 = valid action, 0 = invalid action.
        """
        # Total possible actions
        n_actions = self.action_space.nvec.prod()
        mask = np.zeros(n_actions, dtype=np.int8)
        
        # Check all possible actions
        for action_type in range(2):
            if action_type == 0:  # Move cards
                for from_col in range(10):
                    if not self.tableau[from_col]:
                        continue
                    
                    # Find all valid sequences starting from this column
                    for num_cards in range(1, min(14, len(self.tableau[from_col]) + 1)):
                        # Check if these cards form a valid sequence
                        if not self._is_valid_sequence(self.tableau[from_col][-num_cards:]):
                            continue
                        
                        for to_col in range(10):
                            if from_col == to_col:
                                continue
                            
                            if self._is_valid_move(from_col, to_col, num_cards):
                                action = np.array([action_type, from_col, to_col, num_cards])
                                idx = self._action_to_index(action)
                                mask[idx] = 1
            
            elif action_type == 1:  # Deal from stock
                if len(self.stock) > 0:
                    # For dealing, only the action_type matters
                    action = np.array([1, 0, 0, 0])
                    idx = self._action_to_index(action)
                    mask[idx] = 1
        
        # Ensure at least one action is available (even if it's bad)
        if mask.sum() == 0:
            # Allow dealing if stock available, otherwise allow any move
            if len(self.stock) > 0:
                action = np.array([1, 0, 0, 0])
                mask[self._action_to_index(action)] = 1
            else:
                # Game might be stuck - allow first valid action found
                mask[0] = 1
        
        return mask
    
    def _action_to_index(self, action: np.ndarray) -> int:
        """Convert multi-discrete action to single index."""
        action_type, from_col, to_col, num_cards = action
        return (action_type * 10 * 10 * 13 + 
                from_col * 10 * 13 + 
                to_col * 13 + 
                num_cards)
    
    def _index_to_action(self, index: int) -> np.ndarray:
        """Convert single index to multi-discrete action."""
        num_cards = index % 13
        index //= 13
        to_col = index % 10
        index //= 10
        from_col = index % 10
        index //= 10
        action_type = index
        return np.array([action_type, from_col, to_col, num_cards])


# Wrapper for stable-baselines3 that handles action masking
class ActionMasker(gym.ActionWrapper):
    """
    Wrapper that applies action masks for discrete action spaces.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._action_space_shape = env.action_space.nvec.prod()
        self.action_space = spaces.Discrete(self._action_space_shape)
        
    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to multi-discrete."""
        return self.env._index_to_action(action)
    
    def reverse_action(self, action: np.ndarray) -> int:
        """Convert multi-discrete action to discrete."""
        return self.env._action_to_index(action)