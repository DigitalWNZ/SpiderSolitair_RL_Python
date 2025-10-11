##########################################################################################
# Masked environment for Faceup Spider Solitaire
# Finds all potential actions and validates them using _is_valid_move
# Valid actions are marked as 1, invalid as 0
##########################################################################################
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from spide_solitaire_env_faceup import SpiderSolitaireEnv
from typing import Any, Dict, List, Optional, Tuple


class MaskedSpiderSolitaireEnvFaceup(SpiderSolitaireEnv):
    """
    Faceup Spider Solitaire with action masking to ensure only valid actions are taken.
    """

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500):
        super().__init__(render_mode, max_steps)
        self._action_masks = None

        # Update observation space to include action mask
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(0, 26, shape=(10, 19), dtype=np.int8),
            'stock_count': spaces.Box(0, 5, shape=(1,), dtype=np.int8),
            'foundation_count': spaces.Box(0, 1, shape=(1,), dtype=np.int8),
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
        mask = np.zeros(total_actions, dtype=np.int8)

        # Check deal actions
        if len(self.stock) > 0:
            # Deal action is [1, 0, 0, 0]
            deal_idx = self._action_to_index(np.array([1, 0, 0, 0]))
            mask[deal_idx] = 1

        # Check move actions
        for from_col in range(10):
            if not self.tableau[from_col]:
                continue

            # Try different numbers of cards to move
            for num_cards in range(1, len(self.tableau[from_col]) + 1):
                # Check if this sequence is valid to move
                cards_to_move = self.tableau[from_col][-num_cards:]
                if not self._is_valid_sequence(cards_to_move):
                    continue

                # Try moving to each column
                for to_col in range(10):
                    if from_col != to_col and self._is_valid_move(from_col, to_col, num_cards):
                        action = np.array([0, from_col, to_col, num_cards])
                        idx = self._action_to_index(action)
                        if 0 <= idx < total_actions:
                            mask[idx] = 1

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
class ActionMasker(gym.Wrapper):
    """
    Wrapper that converts the multi-discrete action space to discrete.
    Note: Using gym.Wrapper instead of gym.ActionWrapper to avoid automatic action conversion.
    """

    def __init__(self, env):
        super().__init__(env)
        self._action_space_shape = env.action_space.nvec.prod()
        self.action_space = spaces.Discrete(self._action_space_shape)

    def step(self, action: int):
        """Pass the action directly to the wrapped environment."""
        # MaskedSpiderSolitaireEnvFaceup already expects integer actions
        return self.env.step(action)


# Create the ActionMasker wrapper function
def create_masked_faceup_env(render_mode: Optional[str] = None, max_steps: int = 500):
    """Create a masked Faceup Spider Solitaire environment."""
    env = MaskedSpiderSolitaireEnvFaceup(render_mode, max_steps)
    return ActionMasker(env)
