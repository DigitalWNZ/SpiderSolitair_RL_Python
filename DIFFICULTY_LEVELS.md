# Spider Solitaire Difficulty Levels

## Overview

The Spider Solitaire environment supports multiple difficulty levels when using strategic dealing (`use_strategic_deal=True`).

## Difficulty Levels

### 1. **HAPPY** (Guaranteed Winnable)
- **Purpose**: Create layouts that are guaranteed to be solvable with optimal play
- **Strategy**:
  - Tableau cards arranged in descending sequences
  - Designed to allow combining sequences to form complete K→A runs
  - Stock cards are strategically placed to help rather than hinder
- **Initial Valid Moves**: ~5 (fewer initial moves but better long-term winnability)
- **Best For**: Testing winning strategies, demonstration purposes, curriculum learning (final stage)

**Tableau Design:**
```
Col 0: K-Q-J-10-9-8 (6 cards)
Col 1: 7-6-5-4-3-2 (6 cards)
Col 2: 6-5-4-3-2-A (6 cards)
Col 3: K-Q-J-10-9-8 (6 cards)
Col 4: Q-J-10-9-8 (5 cards)
Col 5: J-10-9-8-7 (5 cards)
Col 6: 10-9-8-7-6 (5 cards)
Col 7: 9-8-7-6-5 (5 cards)
Col 8: 8-7-6-5-4 (5 cards)
Col 9: 7-6-5-4-3 (5 cards)
```

### 2. **EASY** (Strategic Random)
- **Purpose**: Create easier-than-random layouts with good initial moves
- **Strategy**:
  - Partial descending sequences in columns
  - High cards start sequences, descending downward
  - Minimizes blocking patterns
- **Initial Valid Moves**: ~13 (47% more than random)
- **Best For**: Initial RL training, faster learning, curriculum learning (early stage)

### 3. **MEDIUM** (Partially Randomized)
- **Purpose**: Balanced difficulty between easy and hard
- **Strategy**:
  - Cards grouped into chunks
  - Shuffle within chunks to maintain some structure
  - Moderate blocking patterns
- **Initial Valid Moves**: ~10
- **Best For**: Intermediate training, testing robustness

### 4. **HARD** (Nearly Random)
- **Purpose**: Challenge agents with realistic random deals
- **Strategy**:
  - Almost full shuffle of all cards
  - Minimal strategic placement
- **Initial Valid Moves**: ~8 (similar to pure random)
- **Best For**: Final evaluation, testing generalization

---

## Usage Examples

### Basic Usage
```python
from spide_solitaire_env_faceup import SpiderSolitaireEnv

# Happy difficulty (guaranteed winnable)
env_happy = SpiderSolitaireEnv(use_strategic_deal=True, difficulty='happy')

# Easy difficulty (most initial moves)
env_easy = SpiderSolitaireEnv(use_strategic_deal=True, difficulty='easy')

# Medium difficulty
env_medium = SpiderSolitaireEnv(use_strategic_deal=True, difficulty='medium')

# Hard difficulty
env_hard = SpiderSolitaireEnv(use_strategic_deal=True, difficulty='hard')
```

### With Masked Environment
```python
from spider_solitaire_masked_env_faceup import create_masked_faceup_env

env = create_masked_faceup_env(
    max_steps=500,
    use_strategic_deal=True,
    difficulty='happy'  # or 'easy', 'medium', 'hard'
)
```

---

## Performance Comparison (Random Actions)

| Difficulty | Initial Moves | Avg Reward | Win Rate | Best For |
|------------|---------------|------------|----------|----------|
| **Happy** | 5 | ~1,200 | 0%* | Winning with trained agents |
| **Easy** | 13 | ~1,200 | 0%* | Early RL training |
| **Medium** | 10 | ~1,100 | 0%* | Intermediate training |
| **Hard** | 8 | ~1,100 | 0%* | Evaluation |
| **Random** | 8 | -4,980 | 0%* | Baseline |

*With random actions. Trained agents show much better performance.

---

## Curriculum Learning Recommendation

For best RL training results, use a curriculum approach:

1. **Phase 1 (Episodes 0-200)**: `difficulty='easy'`
   - Maximum initial valid moves
   - Fastest initial learning
   - High exploration opportunities

2. **Phase 2 (Episodes 200-500)**: `difficulty='medium'`
   - Balanced challenge
   - Tests generalization
   - Reduces overfitting to easy layouts

3. **Phase 3 (Episodes 500+)**: `difficulty='happy'`
   - Practice winning strategies
   - Learn to complete full sequences
   - Maximize win rate

4. **Evaluation**: `difficulty='hard'` or random
   - Test on realistic random deals
   - Measure true performance

---

## Implementation Details

The difficulty levels are implemented in `spide_solitaire_env_faceup.py`:

- `_create_happy_deck()`: Lines 397-451 - Guaranteed winnable layout
- `_create_easy_deck()`: Lines 453-508 - Strategic random with partial sequences
- `_create_medium_deck()`: Lines 510-535 - Chunk-based partial shuffle
- `_create_hard_deck()`: Lines 537-550 - Nearly full random shuffle

All difficulty levels maintain the same game rules and total card count (104 cards).
