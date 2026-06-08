#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Version 1 ConnectX agent: direct residual policy-value network inference.

This file reuses the embedded AlphaZero-style network weights and game helpers
from submission.py, but it does not run MCTS. It selects the legal action with
the highest policy probability from a single network forward pass.
"""

import argparse

try:
    import submission as final_submission
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        final_submission = None
        _IMPORT_ERROR = exc
    else:
        raise
else:
    _IMPORT_ERROR = None


_STATE = None


def _init_state(rows, cols, inarow):
    game = final_submission._ConnectGame(rows=rows, columns=cols, n_in_row=inarow)
    nnet = final_submission._NNet(rows, cols)
    final_submission._load_embedded_weights(nnet)
    return {
        "game": game,
        "nnet": nnet,
        "config_key": (rows, cols, inarow),
    }


def agent(observation, configuration):
    """Kaggle entry point: choose the highest-probability legal action."""
    if final_submission is None:
        raise RuntimeError("torch is required because this agent reuses submission.py")

    global _STATE

    rows = getattr(configuration, "rows", 6)
    cols = getattr(configuration, "columns", 7)
    inarow = getattr(configuration, "inarow", 4)
    key = (rows, cols, inarow)

    if _STATE is None or _STATE["config_key"] != key:
        _STATE = _init_state(rows, cols, inarow)

    game = _STATE["game"]
    nnet = _STATE["nnet"]

    board = final_submission._obs_to_board(observation, configuration)
    player = (
        final_submission._ChessType.BLACK
        if observation.mark == 1
        else final_submission._ChessType.WHITE
    )

    actions = game.available_actions(board)
    if not actions:
        return 0

    canonical = game.get_canonical_form(board, player)
    probs, _ = nnet.predict(canonical)
    best_action = max(actions, key=lambda action: probs[action])
    return int(best_action % cols)


def _summarize(results, agent_index):
    wins = sum(1 for row in results if row[agent_index] == 1)
    losses = sum(1 for row in results if row[agent_index] == -1)
    draws = len(results) - wins - losses
    return wins, losses, draws, wins / max(1, len(results))


def _run_eval(episodes, opponent):
    from kaggle_environments import evaluate

    as_p1 = evaluate("connectx", [agent, opponent], num_episodes=episodes)
    p1 = _summarize(as_p1, 0)

    as_p2 = evaluate("connectx", [opponent, agent], num_episodes=episodes)
    p2 = _summarize(as_p2, 1)

    print(f"Version 1: policy-value network only")
    print(f"Opponent : {opponent}")
    print(f"Episodes : {episodes} as P1 + {episodes} as P2")
    print(
        f"As P1   : wins={p1[0]} losses={p1[1]} draws={p1[2]} "
        f"win_rate={p1[3] * 100:.1f}%"
    )
    print(
        f"As P2   : wins={p2[0]} losses={p2[1]} draws={p2[2]} "
        f"win_rate={p2[3] * 100:.1f}%"
    )


if __name__ == "__main__":
    if _IMPORT_ERROR is not None:
        print("torch is not installed.")
        print("Install with: pip install torch numpy kaggle-environments")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--opponent", default="random", choices=["random", "negamax"])
    args = parser.parse_args()

    try:
        _run_eval(args.episodes, args.opponent)
    except ImportError:
        print("kaggle_environments is not installed.")
        print("Install with: pip install kaggle-environments")
