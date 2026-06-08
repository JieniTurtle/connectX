#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Version 2 ConnectX agent: policy-value network with one-step lookahead.

This file reuses the embedded network weights and game helpers from
submission.py. It is a lightweight bridge between direct network inference and
full AlphaZero MCTS: the policy head proposes candidates, then the value head
scores the board after each candidate move.
"""

import argparse
import math

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
_TOP_K = 7
_POLICY_WEIGHT = 0.35
_VALUE_WEIGHT = 1.0


def _init_state(rows, cols, inarow):
    game = final_submission._ConnectGame(rows=rows, columns=cols, n_in_row=inarow)
    nnet = final_submission._NNet(rows, cols)
    final_submission._load_embedded_weights(nnet)
    return {
        "game": game,
        "nnet": nnet,
        "config_key": (rows, cols, inarow),
    }


def _score_action(game, nnet, board, player, action, prior):
    next_board, next_player = game.next_state(board, action, player)
    terminal = game.is_terminal_state(next_board, action, player)

    if terminal is not None:
        value_for_current = game.compute_reward(terminal, player)
    else:
        next_canonical = game.get_canonical_form(next_board, next_player)
        _, value_for_next_player = nnet.predict(next_canonical)
        value_for_current = -float(value_for_next_player)

    policy_score = math.log(max(float(prior), 1e-12))
    return _POLICY_WEIGHT * policy_score + _VALUE_WEIGHT * value_for_current


def agent(observation, configuration):
    """Kaggle entry point: choose by policy prior plus one-step value."""
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
    ranked_actions = sorted(actions, key=lambda action: probs[action], reverse=True)
    candidates = ranked_actions[: min(_TOP_K, len(ranked_actions))]

    best_action = max(
        candidates,
        key=lambda action: _score_action(game, nnet, board, player, action, probs[action]),
    )
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

    print("Version 2: policy-value network with one-step lookahead")
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
