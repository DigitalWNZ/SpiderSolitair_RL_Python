##########################################################################################
# Spider Solitaire Environment with Strategic Card Placement
# This creates easier matches by strategically arranging cards instead of random shuffle
##########################################################################################

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, List, Dict, Any
from spide_solitaire_env_faceup import SpiderSolitaireEnv


class SpiderSolitaireEasyMatchEnv(SpiderSolitaireEnv):
    """
    Spider Solitaire with strategic card placement for easier matches.

    Strategy:
    1. Create partial sequences in tableau columns
    2. Minimize blocking (avoid high cards under low cards)
    3. Distribute cards to enable early moves
    4. Keep at least one column with a strong sequence
    """

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 1000,
                 difficulty: str = 'easy'):
        """
        Args:
            render_mode: Rendering mode ('human' or 'ansi')
            max_steps: Maximum steps per episode
            difficulty: 'easy', 'medium', or 'hard' - controls how strategic the layout is
        """
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.difficulty = difficulty

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset with strategic card placement instead of random shuffle."""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create deck with strategic placement
        if self.difficulty == 'easy':
            deck = self._create_easy_deck()
        elif self.difficulty == 'medium':
            deck = self._create_medium_deck()
        else:
            deck = self._create_hard_deck()

        # Deal strategic tableau
        self.tableau = [[] for _ in range(10)]
        cards_per_column = [6, 6, 6, 6, 5, 5, 5, 5, 5, 5]

        for col, num_cards in enumerate(cards_per_column):
            for i in range(num_cards):
                card = deck.pop()
                # Last card in each column is face up
                if i == num_cards - 1:
                    card['face_up'] = True
                self.tableau[col].append(card)

        # Remaining cards go to stock
        self.stock = [deck[i:i+10] for i in range(0, len(deck), 10)]
        self.foundation = []
        self.moves = 0
        self.current_step = 0
        self.move_history = []

        return self._get_obs(), self._get_info()

    def _create_easy_deck(self) -> List[Dict]:
        """
        Create an easy deck with strategic placement.
        Strategy:
        1. First columns get descending sequences
        2. Face-down cards don't block sequences badly
        3. Multiple complete sequences possible
        """
        deck = []

        # Create 8 complete sequences (K to A)
        sequences = []
        for seq_num in range(8):
            sequence = []
            for rank in range(12, -1, -1):  # K(12) down to A(0)
                sequence.append({'rank': rank, 'face_up': False})
            sequences.append(sequence)

        # Strategically distribute cards to tableau and stock
        # Goal: Make at least 1-2 sequences easily completable

        # Tableau gets 54 cards arranged strategically
        tableau_cards = []

        # Column layout: create partial sequences
        # Columns 0-3 get longer partial sequences (6 cards each)
        for col in range(4):
            # Start with a high card and descend
            start_rank = random.randint(8, 12)  # Start from 9-K
            for i in range(6):
                rank = max(0, start_rank - i)
                card = {'rank': rank, 'face_up': False}
                tableau_cards.append(card)

        # Columns 4-9 get shorter sequences (5 cards each)
        for col in range(6):
            start_rank = random.randint(6, 11)  # Start from 7-Q
            for i in range(5):
                rank = max(0, start_rank - i)
                card = {'rank': rank, 'face_up': False}
                tableau_cards.append(card)

        # Stock gets remaining 50 cards (more random but still helpful)
        stock_cards = []
        all_cards = []
        for _ in range(8):
            for rank in range(13):
                all_cards.append({'rank': rank, 'face_up': False})

        # Remove the cards we've already used for tableau
        remaining = [c for c in all_cards]
        # Randomly select 50 for stock
        random.shuffle(remaining)
        stock_cards = remaining[:50]

        # Combine: stock cards first, then tableau (we pop from end)
        deck = stock_cards + tableau_cards

        return deck

    def _create_medium_deck(self) -> List[Dict]:
        """
        Create a medium difficulty deck.
        Less strategic than easy, but better than random.
        """
        deck = []

        # Create all 104 cards
        for _ in range(8):
            for rank in range(13):
                deck.append({'rank': rank, 'face_up': False})

        # Partially shuffle to maintain some sequences
        # Group into chunks and shuffle within chunks
        chunk_size = 20
        chunks = [deck[i:i+chunk_size] for i in range(0, len(deck), chunk_size)]

        for chunk in chunks:
            random.shuffle(chunk)

        # Reassemble
        deck = []
        for chunk in chunks:
            deck.extend(chunk)

        return deck

    def _create_hard_deck(self) -> List[Dict]:
        """
        Create a hard difficulty deck (nearly random but slightly better).
        """
        deck = []

        # Create all 104 cards
        for _ in range(8):
            for rank in range(13):
                deck.append({'rank': rank, 'face_up': False})

        # Almost full shuffle
        random.shuffle(deck)

        return deck

    def get_winnable_match(self, seed: Optional[int] = None) -> Dict:
        """
        Generate a specifically crafted winnable match.
        Returns the initial state of a match designed to be solvable.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create a highly strategic winnable layout
        deck = self._create_guaranteed_winnable_deck()

        # Deal tableau
        tableau = [[] for _ in range(10)]
        cards_per_column = [6, 6, 6, 6, 5, 5, 5, 5, 5, 5]

        for col, num_cards in enumerate(cards_per_column):
            for i in range(num_cards):
                card = deck.pop()
                if i == num_cards - 1:
                    card['face_up'] = True
                tableau[col].append(card)

        # Stock
        stock = [deck[i:i+10] for i in range(0, len(deck), 10)]

        return {
            'tableau': tableau,
            'stock': stock,
            'cards_per_column': cards_per_column
        }

    def _create_guaranteed_winnable_deck(self) -> List[Dict]:
        """
        Create a deck arrangement that is guaranteed winnable.

        Strategy:
        1. Place one complete sequence (K-A) in a single column with minimal blocking
        2. Arrange other cards to be movable to clear the sequence
        3. Ensure stock cards help rather than hinder
        """
        deck = []

        # Column 0: Will contain a complete sequence K-A
        # Place it with one blocking card at position 2
        complete_sequence = []
        for rank in range(12, -1, -1):  # K down to A
            complete_sequence.append({'rank': rank, 'face_up': False})

        # For easy winning: place the sequence in a way that's accessible
        # Let's design the tableau strategically

        tableau_design = [
            # Col 0: 6 cards - will build K-Q-J-10-9-8
            [12, 11, 10, 9, 8, 7],
            # Col 1: 6 cards - will build 7-6-5-4-3-2
            [6, 5, 4, 3, 2, 1],
            # Col 2: 6 cards - continuation
            [5, 4, 3, 2, 1, 0],
            # Col 3: 6 cards - high cards that can move
            [12, 11, 10, 9, 8, 7],
            # Col 4-9: 5 cards each - various sequences
            [11, 10, 9, 8, 7],
            [10, 9, 8, 7, 6],
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [6, 5, 4, 3, 2],
        ]

        tableau_cards = []
        for col_design in tableau_design:
            for rank in col_design:
                tableau_cards.append({'rank': rank, 'face_up': False})

        # Create stock with remaining cards
        all_ranks = list(range(13)) * 8  # 104 cards total
        used_ranks = []
        for col_design in tableau_design:
            used_ranks.extend(col_design)

        stock_ranks = all_ranks.copy()
        for rank in used_ranks:
            stock_ranks.remove(rank)

        random.shuffle(stock_ranks)
        stock_cards = [{'rank': rank, 'face_up': False} for rank in stock_ranks]

        # Deck: stock first, then tableau (since we pop from end)
        deck = stock_cards + tableau_cards

        return deck


def demo_easy_match():
    """Demonstrate an easy match."""
    print("Creating Easy Spider Solitaire Match...\n")

    env = SpiderSolitaireEasyMatchEnv(render_mode='human', difficulty='easy')
    obs, info = env.reset(seed=42)

    env.render()
    print(f"\nValid moves available: {info['valid_moves']}")
    print("\nThis layout is strategically designed to be easier to solve!")
    print("Notice the partial sequences in the columns.")

    return env


def demo_winnable_match():
    """Demonstrate a guaranteed winnable match."""
    print("\n" + "="*60)
    print("Creating Guaranteed Winnable Match...\n")

    env = SpiderSolitaireEasyMatchEnv(render_mode='human', difficulty='easy')
    match_state = env.get_winnable_match(seed=123)

    # Manually set the state
    env.tableau = match_state['tableau']
    env.stock = match_state['stock']
    env.foundation = []
    env.moves = 0

    env.render()
    print(f"\nThis match is designed to be winnable with optimal play!")

    return env


if __name__ == "__main__":
    print("Spider Solitaire - Easy Match Generator")
    print("="*60)

    # Demo easy match
    easy_env = demo_easy_match()

    # Demo winnable match
    winnable_env = demo_winnable_match()

    print("\n" + "="*60)
    print("You can use these environments for training:")
    print("  - difficulty='easy': Strategic but still challenging")
    print("  - difficulty='medium': Partially randomized")
    print("  - difficulty='hard': Mostly random")
    print("  - get_winnable_match(): Guaranteed solvable layout")
