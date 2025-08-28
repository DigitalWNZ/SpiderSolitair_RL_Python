import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, List, Dict, Any


class SpiderSolitaireEnv(gym.Env):
    """
    Spider Solitaire environment for reinforcement learning.
    
    Game Rules:
    - 104 cards (2 decks), single suit variant
    - 10 tableau columns
    - Initial deal: 54 cards (6,6,6,6,5,5,5,5,5,5 per column)
    - Face-down cards except last card in each column
    - 5 stock piles with 10 cards each (50 cards total)
    - Goal: Create 8 complete sequences (K to A) of the same suit
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    # Card constants
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        
        # Game state
        self.tableau = [[] for _ in range(10)]  # 10 columns
        self.stock = []  # Remaining cards to deal
        self.foundation = []  # Completed sequences
        self.score = 0
        self.moves = 0
        
        # Define observation space
        # Maximum cards in a column: 19 (theoretical max)
        # Card representation: 0=empty, 1-13=face up (A-K), 14-26=face down (A-K)
        self.observation_space = spaces.Dict({
            'tableau': spaces.Box(low=0, high=26, shape=(10, 19), dtype=np.int8),
            'stock_count': spaces.Box(low=0, high=5, shape=(1,), dtype=np.int8),
            'foundation_count': spaces.Box(low=0, high=8, shape=(1,), dtype=np.int8),
        })
        
        # Define action space
        # Actions: (action_type, from_col, to_col, num_cards)
        # action_type: 0=move cards, 1=deal from stock
        self.action_space = spaces.MultiDiscrete([2, 10, 10, 13])
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # options parameter kept for gymnasium compatibility
        
        # Initialize deck (single suit - simplified version)
        deck = []
        for _ in range(8):  # 8 sets of 13 cards
            for rank in range(13):
                deck.append({'rank': rank, 'face_up': False})
        
        # Shuffle deck
        if seed is not None:
            random.seed(seed)
        random.shuffle(deck)
        
        # Deal initial tableau
        self.tableau = [[] for _ in range(10)]
        cards_per_column = [6, 6, 6, 6, 5, 5, 5, 5, 5, 5]
        
        for col, num_cards in enumerate(cards_per_column):
            for i in range(num_cards):
                card = deck.pop()
                # Last card in each column is face up
                if i == num_cards - 1:
                    card['face_up'] = True
                self.tableau[col].append(card)
        
        # Remaining cards go to stock (5 piles of 10 cards each)
        self.stock = [deck[i:i+10] for i in range(0, 50, 10)]
        self.foundation = []
        self.score = 500  # Starting score
        self.moves = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        action_type, from_col, to_col, num_cards = action
        
        terminated = False
        reward = 0
        
        if action_type == 0:  # Move cards
            if self._is_valid_move(from_col, to_col, num_cards):
                reward = self._move_cards(from_col, to_col, num_cards)
                self.moves += 1
                self.score -= 1  # Penalty for each move
                
                # Check for completed sequences
                self._check_completed_sequences()
                
                # Check if game is won
                if len(self.foundation) == 8:
                    terminated = True
                    reward += 1000  # Win bonus
            else:
                reward = -10  # Invalid move penalty
                
        elif action_type == 1:  # Deal from stock
            if len(self.stock) > 0:
                self._deal_from_stock()
                self.moves += 1
            else:
                reward = -10  # No stock available
        
        truncated = False  # Could add move limit
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _is_valid_move(self, from_col: int, to_col: int, num_cards: int) -> bool:
        """Check if a move is valid."""
        if from_col < 0 or from_col >= 10 or to_col < 0 or to_col >= 10:
            return False
        
        if from_col == to_col:
            return False
        
        from_pile = self.tableau[from_col]
        to_pile = self.tableau[to_col]
        
        if len(from_pile) < num_cards:
            return False
        
        # Check if cards to move form a valid sequence
        cards_to_move = from_pile[-num_cards:]
        if not self._is_valid_sequence(cards_to_move):
            return False
        
        # Check if move to destination is valid
        if len(to_pile) == 0:
            return True  # Can move any sequence to empty column
        
        # Top card of sequence must be one rank lower than destination
        top_card = to_pile[-1]
        moving_card = cards_to_move[0]
        
        if not top_card['face_up']:
            return False
        
        return top_card['rank'] == moving_card['rank'] + 1
    
    def _is_valid_sequence(self, cards: List[Dict]) -> bool:
        """Check if cards form a valid descending sequence."""
        if not cards:
            return False
        
        for i in range(len(cards) - 1):
            if not cards[i]['face_up'] or not cards[i+1]['face_up']:
                return False
            if cards[i]['rank'] != cards[i+1]['rank'] + 1:
                return False
        
        return True
    
    def _move_cards(self, from_col: int, to_col: int, num_cards: int) -> float:
        """Execute a card move and return reward."""
        cards_to_move = self.tableau[from_col][-num_cards:]
        self.tableau[from_col] = self.tableau[from_col][:-num_cards]
        self.tableau[to_col].extend(cards_to_move)
        
        # Flip face-down card if exposed
        if self.tableau[from_col] and not self.tableau[from_col][-1]['face_up']:
            self.tableau[from_col][-1]['face_up'] = True
            return 5  # Reward for revealing a card
        
        return 0
    
    def _deal_from_stock(self):
        """Deal one card from stock to each column."""
        if not self.stock:
            return
        
        stock_pile = self.stock.pop()
        for col in range(10):
            if stock_pile:
                card = stock_pile.pop()
                card['face_up'] = True
                self.tableau[col].append(card)
    
    def _check_completed_sequences(self):
        """Check for and remove completed K-A sequences."""
        for col in range(10):
            if len(self.tableau[col]) >= 13:
                # Check if last 13 cards form a complete sequence
                last_13 = self.tableau[col][-13:]
                if self._is_complete_sequence(last_13):
                    self.tableau[col] = self.tableau[col][:-13]
                    self.foundation.append(last_13)
                    self.score += 100  # Bonus for completing sequence
                    
                    # Flip face-down card if exposed
                    if self.tableau[col] and not self.tableau[col][-1]['face_up']:
                        self.tableau[col][-1]['face_up'] = True
    
    def _is_complete_sequence(self, cards: List[Dict]) -> bool:
        """Check if cards form a complete K to A sequence."""
        if len(cards) != 13:
            return False
        
        for i, card in enumerate(cards):
            if not card['face_up']:
                return False
            if card['rank'] != 12 - i:  # K=12, Q=11, ..., A=0
                return False
        
        return True
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        tableau_obs = np.zeros((10, 19), dtype=np.int8)
        
        for col in range(10):
            for row, card in enumerate(self.tableau[col]):
                if row < 19:  # Max observable cards
                    if card['face_up']:
                        tableau_obs[col, row] = card['rank'] + 1
                    else:
                        tableau_obs[col, row] = card['rank'] + 14
        
        return {
            'tableau': tableau_obs,
            'stock_count': np.array([len(self.stock)], dtype=np.int8),
            'foundation_count': np.array([len(self.foundation)], dtype=np.int8),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            'score': self.score,
            'moves': self.moves,
            'valid_moves': self._count_valid_moves(),
        }
    
    def _count_valid_moves(self) -> int:
        """Count number of valid moves available."""
        count = 0
        
        # Count card moves
        for from_col in range(10):
            if not self.tableau[from_col]:
                continue
                
            # Find all movable sequences
            for num_cards in range(1, len(self.tableau[from_col]) + 1):
                if self._is_valid_sequence(self.tableau[from_col][-num_cards:]):
                    for to_col in range(10):
                        if from_col != to_col and self._is_valid_move(from_col, to_col, num_cards):
                            count += 1
        
        # Add stock deal if available
        if self.stock:
            count += 1
        
        return count
    
    def render(self):
        """Render the game state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Render game state as ASCII text."""
        lines = []
        lines.append(f"Spider Solitaire - Score: {self.score}, Moves: {self.moves}")
        lines.append(f"Stock: {len(self.stock)} piles, Foundation: {len(self.foundation)}/8")
        lines.append("-" * 50)
        
        # Find max height
        max_height = max(len(col) for col in self.tableau)
        
        # Render tableau
        for row in range(max_height):
            line = ""
            for col in range(10):
                if row < len(self.tableau[col]):
                    card = self.tableau[col][row]
                    if card['face_up']:
                        line += f"{self.RANKS[card['rank']]:>3} "
                    else:
                        line += " ## "
                else:
                    line += "    "
            lines.append(line)
        
        lines.append("-" * 50)
        lines.append("Col: 0   1   2   3   4   5   6   7   8   9")
        
        return "\n".join(lines)