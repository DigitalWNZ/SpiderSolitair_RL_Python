import numpy as np
import gymnasium as gym
from gymnasium import spaces
from spider_solitaire_env_fixed import SpiderSolitaireEnvFixed
from typing import Any, Dict, List, Optional, Tuple


class MaskedSpiderSolitaireEnvFixed(SpiderSolitaireEnvFixed):
    """
    Fixed Spider Solitaire with action masking to ensure only valid actions are taken.
    """
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500):
        super().__init__(render_mode, max_steps)
        self._action_masks = None
        
        # Update observation space to include action mask
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(0, 52, shape=(10, 19), dtype=np.int32),
            'stock_count': spaces.Box(0, 5, shape=(1,), dtype=np.int32),
            'foundation_count': spaces.Box(0, 8, shape=(1,), dtype=np.int32),
            'action_mask': spaces.Box(0, 1, shape=(self.action_space.nvec.prod(),), dtype=np.int8),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed, options)
        self._action_masks = self._compute_action_masks()
        obs['action_mask'] = self._action_masks
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Convert flat action to multi-discrete
        action_array = self._index_to_action(action)
        obs, reward, terminated, truncated, info = super().step(action_array)
        
        # Update action masks
        self._action_masks = self._compute_action_masks()
        obs['action_mask'] = self._action_masks
        
        # If no valid actions and game not over, truncate
        if not terminated and not truncated and self._action_masks.sum() == 0:
            truncated = True
            reward -= 100  # Penalty for getting completely stuck
        
        return obs, reward, terminated, truncated, info
    
    def _compute_action_masks(self) -> np.ndarray:
        """Compute which actions are valid in current state."""
        total_actions = 2 * 10 * 10 * 13  # deal * from * to * num_cards
        mask = np.zeros(total_actions, dtype=bool)
        
        # Check deal actions
        if len(self.stock) > 0 and self._is_valid_deal():
            # Deal action is [1, 0, 0, 0]
            deal_idx = self._action_to_index(np.array([1, 0, 0, 0]))
            mask[deal_idx] = True
        
        # Check move actions
        for from_col in range(10):
            if not self.tableau[from_col]:
                continue
            
            # Try different numbers of cards
            max_movable = 0
            for num_cards in range(1, len(self.tableau[from_col]) + 1):
                if self._is_movable_sequence(from_col, num_cards):
                    max_movable = num_cards
                else:
                    break
            
            # For each movable sequence length
            for num_cards in range(1, max_movable + 1):
                for to_col in range(10):
                    if from_col != to_col and self._is_valid_move(from_col, to_col, num_cards):
                        action = np.array([0, from_col, to_col, num_cards])
                        idx = self._action_to_index(action)
                        if 0 <= idx < total_actions:
                            mask[idx] = True
        
        return mask
    
    def _action_to_index(self, action: np.ndarray) -> int:
        """Convert multi-discrete action to flat index."""
        deal, from_col, to_col, num_cards = action
        return deal * (10 * 10 * 13) + from_col * (10 * 13) + to_col * 13 + num_cards
    
    def _index_to_action(self, index: int) -> np.ndarray:
        """Convert flat index to multi-discrete action."""
        deal = index // (10 * 10 * 13)
        remainder = index % (10 * 10 * 13)
        
        from_col = remainder // (10 * 13)
        remainder = remainder % (10 * 13)
        
        to_col = remainder // 13
        num_cards = remainder % 13
        
        return np.array([deal, from_col, to_col, num_cards])
    
    def action_masks(self) -> np.ndarray:
        """Return current action masks for gymnasium ActionMasker."""
        return self._action_masks


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


# Create the ActionMasker wrapper function
def create_masked_env(render_mode: Optional[str] = None, max_steps: int = 500):
    """Create a masked Spider Solitaire environment."""
    env = MaskedSpiderSolitaireEnvFixed(render_mode, max_steps)
    return ActionMasker(env)