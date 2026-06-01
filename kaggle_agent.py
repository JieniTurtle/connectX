#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Kaggle ConnectX submission agent powered by AlphaZero.

Loads a trained AlphaZero model checkpoint and uses MCTS to select moves.

Usage (local test):
    from kaggle_environments import make
    env = make("connectx", debug=True)
    env.run([kaggle_agent.agent, "random"])
    env.render(mode="human")

Kaggle submission:
    Submit this file (and the model checkpoint as a data file) to Kaggle.
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make the alphazero-board-games package importable
# ---------------------------------------------------------------------------
_ALPHAZERO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphazero-board-games")
if _ALPHAZERO_DIR not in sys.path:
    sys.path.insert(0, _ALPHAZERO_DIR)

from connect4.config import Connect4Config
from connect4.game import Connect4Game, ChessType
from alphazero.nnet import AlphaZeroNNet
from alphazero.mcts import MCTS


# ---------------------------------------------------------------------------
# Global caches — model and MCTS are loaded once then reused across moves
# ---------------------------------------------------------------------------
_cached_config = None
_cached_config_key = None
_cached_game = None
_cached_nnet = None
_cached_mcts = None


def _config_key(configuration):
    """Return a hashable key for the board configuration."""
    return (configuration.rows, configuration.columns, configuration.inarow)


def _get_or_load_model(observation, configuration):
    """Load (or reload) the AlphaZero model for the given board configuration.

    Since the neural network architecture depends on board dimensions,
    we must recreate it when the configuration changes.  In practice
    Kaggle uses a single configuration per episode, so this is called
    at most once per game.
    """
    global _cached_config, _cached_config_key
    global _cached_game, _cached_nnet, _cached_mcts

    key = _config_key(configuration)
    if key == _cached_config_key and _cached_config is not None:
        return _cached_game, _cached_nnet, _cached_mcts, _cached_config

    # Build a fresh config matching the Kaggle-provided board dimensions
    config = Connect4Config(
        rows=configuration.rows,
        columns=configuration.columns,
        n_in_row=configuration.inarow,
        simulation_num=200,  # 200 sims balances speed vs strength
    )

    game = Connect4Game(config)
    nnet = AlphaZeroNNet(game, config)

    # Try to load the bundled checkpoint
    import glob as _glob

    candidates = [
        os.path.join(os.path.dirname(__file__), "model"),
        os.path.join(_ALPHAZERO_DIR, "connect4", "data", "model"),
    ]
    for prefix in candidates:
        if _glob.glob(prefix + "*.pt"):
            nnet.load_checkpoint(prefix)
            break
    else:
        print(
            "[kaggle_agent] WARNING: no checkpoint found — "
            "network has random weights; train first."
        )

    mcts = MCTS(nnet, game, config)

    _cached_config = config
    _cached_config_key = key
    _cached_game = game
    _cached_nnet = nnet
    _cached_mcts = mcts

    return game, nnet, mcts, config


# ---------------------------------------------------------------------------
# Observation → board-string conversion
# ---------------------------------------------------------------------------
def _obs_to_board_string(observation, configuration):
    """Convert Kaggle's flat board array into the repo's SGF-style string.

    Kaggle board layout (row-major, 0-indexed)::

        board[row * cols + col]  ∈ {0 (empty), 1 (player-1), 2 (player-2)}
        row 0 = top, row (rows-1) = bottom

    Repo board format::

        "B[rx][cx];W[rx][cx];…"   (hex-encoded row/col, semicolon-separated)
        B = first player (BLACK), W = second player (WHITE)

    """
    rows = configuration.rows
    cols = configuration.columns
    flat = observation.board

    stones = []
    for r in range(rows):
        for c in range(cols):
            val = flat[r * cols + c]
            if val == 1:
                stones.append(f"B[{r:x}{c:x}]")
            elif val == 2:
                stones.append(f"W[{r:x}{c:x}]")

    return ";".join(stones)


# ---------------------------------------------------------------------------
# Kaggle agent entry-point
# ---------------------------------------------------------------------------
def agent(observation, configuration):
    """Kaggle ConnectX agent — returns the 0-indexed column to drop into.

    Called once per turn by the Kaggle environment.
    """
    game, nnet, mcts, config = _get_or_load_model(observation, configuration)

    # Convert observation to the format expected by the game engine
    board = _obs_to_board_string(observation, configuration)
    player = ChessType.BLACK if observation.mark == 1 else ChessType.WHITE

    # Run MCTS guided by the neural network
    actions, counts = mcts.simulate(board, player)

    if len(actions) == 0:
        # Shouldn't happen, but pick the first non-full column as fallback
        for c in range(configuration.columns):
            if observation.board[c] == 0:
                return c
        return 0

    # Greedy: pick the action with the most MCTS visits
    best = np.argmax(counts)
    best_action = actions[best]
    col = int(best_action % configuration.columns)
    return col


# ---------------------------------------------------------------------------
# Local smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from kaggle_environments import make, evaluate

    print("=" * 60)
    print("AlphaZero ConnectX Agent — local evaluation")
    print("=" * 60)

    # --- Single game vs random (verbose) ---
    print("\n▶ Playing one game vs 'random' agent…")
    env = make("connectx", debug=True)
    env.run([agent, "random"])
    result = env.render(mode="human", width=500, height=450)
    print(result)

    # --- Batch evaluation vs random ---
    print("\n▶ Evaluating vs 'random' (20 games)…")
    outcomes = evaluate("connectx", [agent, "random"], num_episodes=20)
    print(outcomes)

    # --- Batch evaluation vs negamax ---
    print("\n▶ Evaluating vs 'negamax' (20 games)…")
    outcomes = evaluate("connectx", [agent, "negamax"], num_episodes=20)
    print(outcomes)
