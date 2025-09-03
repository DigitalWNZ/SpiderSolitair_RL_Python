##########################################################################################
# 游戏规则： 
# 	两副牌
# 	显示的牌面部分是10组牌，分别有[6, 6, 6, 6, 5, 5, 5, 5, 5, 5]张牌
# 	还有5组牌（stock）是隐藏的， 每组10张；每次可以选取一组，这组牌的每张牌会附加到显示的10组牌的末尾。 
# 	当凑足一个A-K，这些牌就会消除，然后foundation + 1， 当foundation = 8，游戏胜利。 

# Reward：
# 	初始reward ==》500
# 	移动一张 ==》-1 分， 
# 	开一张牌 ==》+5 分， 
# 	无效的移动 ==》-10分， 
# 	用光了stock ==》-10分， 
# 	凑足一组A-K ==》+100分， 
# 	凑足8组 ==》+1000分。 

# reset:
# 	两副牌，放到8个数组中，然后放到deck数组中，这是最简洁的实现方式； 
# 	将deck数组打散，用deck.pop方法填满用于显示的牌面[6, 6, 6, 6, 5, 5, 5, 5, 5, 5]。 
# 	剩下的分成5个stock，foundation置为0

# _is_valid_sequence:
# 	花色一致且连续

# _is_valid_move:
# 	每个move的表现形式（from_col， to_col，num_cards）
# 	待移动序列满足_is_valid_sequence
# 	from_col的顶牌 = to_col的底牌

# _move_cards
# 	将待移动序列移动到to_col
# 	看from_col的底牌是否可开，如果开 +5分


# _deal_from_stock
# 	开一个stock
# 	stock中的每张牌会附加到显示的10组牌的末尾

# _is_complete_sequence
# 	判断是否是同花色A-K

# _check_completed_sequences
# 	找出所有A-K序列然后移除， 分数+100

# _count_valid_moves
# 	找出所有潜在的有效的move

# step
# 	如果是移动牌
# 		调用_is_valid_move校验合法性
# 		如果合法==》 调用_move_cards移动，步数 + 1， reward -1； 如果结束 reward + 1000
# 		如果非法==〉reward - 10
# 	如果是开stock
# 		步数 + 1
# 		如果stock开没了 reward - 10
##########################################################################################

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
        
        reward = -1  # Default penalty for each move
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
                else:
                    reward = -10  # Invalid deal penalty
            else:
                reward = -10  # No stock penalty
        else:
            # Move action
            if self._is_valid_move(from_col, to_col, num_cards):
                # Perform move
                cards_to_move = self.tableau[from_col][-num_cards:]
                self.tableau[from_col] = self.tableau[from_col][:-num_cards]
                self.tableau[to_col].extend(cards_to_move)
                self.moves += 1
                
                # Check for completed sequences
                self._check_completed_sequences()
                
                # Small reward for valid move
                reward = 1
            else:
                reward = -10  # Invalid move penalty
        
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
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
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
    
    def _check_completed_sequences(self):
        """Check and remove completed sequences (K to A of same suit)."""
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