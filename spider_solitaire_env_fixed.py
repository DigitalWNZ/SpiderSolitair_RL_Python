import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import random

class SpiderSolitaireEnvFixed(gym.Env):
    """
    Fixed Spider Solitaire environment with proper truncation.
    """
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_steps = 0
        
        # Action space: deal or move card
        # [deal, from_col, to_col, num_cards]
        self.action_space = spaces.MultiDiscrete([2, 10, 10, 13])
        
        # Observation space
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(0, 52, shape=(10, 19), dtype=np.int32),
            'stock_count': spaces.Box(0, 5, shape=(1,), dtype=np.int32),
            'foundation_count': spaces.Box(0, 8, shape=(1,), dtype=np.int32),
        })
        
        self.render_mode = render_mode
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_steps = 0
        self._episode_reward = 0
        
        # Initialize deck (104 cards for 2-suit spider)
        self.deck = list(range(52)) * 2
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.deck)
        
        # Setup tableau (10 columns)
        self.tableau = [[] for _ in range(10)]
        card_idx = 0
        
        # First 4 columns get 6 cards, last 6 get 5 cards
        for col in range(10):
            num_cards = 6 if col < 4 else 5
            for _ in range(num_cards):
                self.tableau[col].append(self.deck[card_idx])
                card_idx += 1
        
        # Remaining cards go to stock (5 deals of 10 cards each)
        self.stock = [self.deck[i:i+10] for i in range(card_idx, len(self.deck), 10)]
        
        # Foundation piles (8 complete sequences can be removed)
        self.foundation = []
        
        # Game state
        self.moves = 0
        self.score = 500  # Starting score
        self.valid_moves = 0
        self.invalid_moves = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_steps += 1
        # Ensure action is properly formatted
        if isinstance(action, (list, tuple, np.ndarray)):
            action = np.array(action).flatten()
            deal = int(action[0])
            from_col = int(action[1])
            to_col = int(action[2])
            num_cards = int(action[3])
        else:
            # Single integer action - should not happen in this environment
            raise ValueError(f"Expected array-like action, got {type(action)}")
        
        reward = -1  # Default penalty for each move (encourages efficiency)
        terminated = False
        truncated = False
        
        if deal == 1:
            # Deal action
            if len(self.stock) > 0:
                if self._is_valid_deal():
                    deal_cards = self.stock.pop(0)
                    for i, card in enumerate(deal_cards):
                        self.tableau[i].append(card)
                    self.moves += 1
                    self.valid_moves += 1
                else:
                    reward = -10  # Invalid deal penalty
                    self.invalid_moves += 1
            else:
                reward = -10  # No stock penalty
                self.invalid_moves += 1
        else:
            # Move action
            if self._is_valid_move(from_col, to_col, num_cards):
                # Perform move
                cards_to_move = self.tableau[from_col][-num_cards:]
                self.tableau[from_col] = self.tableau[from_col][:-num_cards]
                self.tableau[to_col].extend(cards_to_move)
                self.moves += 1
                self.valid_moves += 1
                
                # Check for completed sequences
                sequences_completed = self._check_completed_sequences()
                if sequences_completed > 0:
                    reward += sequences_completed * 100  # +100 for each completed sequence
                
                # Valid move still costs -1 (already set above)
                # This encourages finding the shortest path to win
            else:
                reward = -10  # Invalid move penalty
                self.invalid_moves += 1
        
        # Update score
        self.score = max(0, self.score + reward)
        
        # Check win condition
        if len(self.foundation) == 8:
            terminated = True
            reward += 1000  # Win bonus
        
        # Check truncation conditions
        if self.current_steps >= self.max_steps:
            truncated = True
            reward -= 50  # Penalty for taking too long
        
        # Check if game is stuck (no valid moves and no stock)
        if len(self.stock) == 0 and not self._has_valid_moves():
            truncated = True
            reward -= 100  # Penalty for getting stuck
        
        # Track episode reward
        if not hasattr(self, '_episode_reward'):
            self._episode_reward = 0
        self._episode_reward += reward
        
        info = self._get_info()
        
        # Add game result info when episode ends
        if terminated or truncated:
            if terminated:
                info['game_result'] = 'WON' if len(self.foundation) == 8 else 'LOST'
            else:
                info['game_result'] = 'TRUNCATED'
            info['episode_reward'] = self._episode_reward
            # Reset episode reward for next episode
            self._episode_reward = 0
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _has_valid_moves(self) -> bool:
        """Check if any valid moves exist."""
        for from_col in range(10):
            if not self.tableau[from_col]:
                continue
            
            # Try moving different numbers of cards
            for num_cards in range(1, len(self.tableau[from_col]) + 1):
                if self._is_movable_sequence(from_col, num_cards):
                    for to_col in range(10):
                        if from_col != to_col and self._is_valid_move(from_col, to_col, num_cards):
                            return True
        
        # Check if we can deal
        if len(self.stock) > 0 and self._is_valid_deal():
            return True
        
        return False
    
    def _is_valid_deal(self) -> bool:
        """Check if dealing is valid (no empty columns)."""
        return all(len(col) > 0 for col in self.tableau)
    
    def _is_valid_move(self, from_col: int, to_col: int, num_cards: int) -> bool:
        """Check if a move is valid."""
        if from_col < 0 or from_col >= 10 or to_col < 0 or to_col >= 10:
            return False
        
        if from_col == to_col:
            return False
        
        if len(self.tableau[from_col]) < num_cards:
            return False
        
        # Check if cards form a valid sequence
        if not self._is_movable_sequence(from_col, num_cards):
            return False
        
        # Check destination
        if len(self.tableau[to_col]) == 0:
            return True  # Can move to empty column
        
        # Check if move is valid (descending rank)
        moving_card = self.tableau[from_col][-num_cards]
        dest_card = self.tableau[to_col][-1]
        
        return self._get_rank(moving_card) == self._get_rank(dest_card) - 1
    
    def _is_movable_sequence(self, col: int, num_cards: int) -> bool:
        """Check if cards form a movable sequence."""
        if num_cards == 1:
            return True
        
        cards = self.tableau[col][-num_cards:]
        
        # Check if same suit and descending
        for i in range(len(cards) - 1):
            if self._get_suit(cards[i]) != self._get_suit(cards[i + 1]):
                return False
            if self._get_rank(cards[i]) != self._get_rank(cards[i + 1]) + 1:
                return False
        
        return True
    
    def _check_completed_sequences(self) -> int:
        """Check and remove completed sequences (K to A of same suit).
        Returns the number of sequences completed."""
        completed_count = 0
        for col in range(10):
            if len(self.tableau[col]) >= 13:
                # Check if last 13 cards form a complete sequence
                cards = self.tableau[col][-13:]
                
                # All same suit?
                suits = [self._get_suit(card) for card in cards]
                if len(set(suits)) != 1:
                    continue
                
                # Ranks from K to A?
                ranks = [self._get_rank(card) for card in cards]
                if ranks == list(range(12, -1, -1)):
                    # Remove completed sequence
                    self.tableau[col] = self.tableau[col][:-13]
                    self.foundation.append(cards)
                    self.score += 100  # Bonus for completing sequence
                    completed_count += 1
        return completed_count
    
    def _get_suit(self, card: int) -> int:
        """Get suit of card (0-3)."""
        return (card % 52) // 13
    
    def _get_rank(self, card: int) -> int:
        """Get rank of card (0-12, where 0 is Ace)."""
        return (card % 52) % 13
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Pad tableau to fixed size
        tableau_obs = np.zeros((10, 19), dtype=np.int32)
        for i, col in enumerate(self.tableau):
            for j, card in enumerate(col[:19]):  # Max 19 cards shown
                tableau_obs[i, j] = card + 1  # +1 so 0 means empty
        
        return {
            'tableau': tableau_obs,
            'stock_count': np.array([len(self.stock)], dtype=np.int32),
            'foundation_count': np.array([len(self.foundation)], dtype=np.int32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info."""
        return {
            'moves': self.moves,
            'score': self.score,
            'current_steps': self.current_steps,
            'max_steps': self.max_steps,
            'valid_moves': self.valid_moves,
            'invalid_moves': self.invalid_moves,
            'foundation_count': len(self.foundation),
            'episode_reward': getattr(self, '_episode_reward', 0),
            'episode_length': self.current_steps,
        }
    
    def render(self):
        """Render the game state."""
        if self.render_mode == "human":
            print("\n" + "="*50)
            print(f"Spider Solitaire - Moves: {self.moves}, Score: {self.score}")
            print(f"Steps: {self.current_steps}/{self.max_steps}")
            print(f"Stock: {len(self.stock)} deals, Foundation: {len(self.foundation)} completed")
            print("="*50)
            
            # Show tableau
            max_height = max(len(col) for col in self.tableau)
            for row in range(max_height):
                row_str = ""
                for col in range(10):
                    if row < len(self.tableau[col]):
                        card = self.tableau[col][row]
                        rank = self._get_rank(card)
                        suit = self._get_suit(card)
                        rank_str = "A23456789TJQK"[rank]
                        suit_str = "♠♥♦♣"[suit]
                        row_str += f"{rank_str}{suit_str} "
                    else:
                        row_str += "   "
                print(row_str)
            print("-" * 30)