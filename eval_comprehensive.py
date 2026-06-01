#!/usr/bin/python3
"""Comprehensive evaluation of the AlphaZero agent for Kaggle ConnectX."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "alphazero-board-games"))

from kaggle_environments import make, evaluate
import kaggle_agent

def eval_matchups(agent, opponents, num_games=40):
    """Evaluate agent against each opponent, both as P1 and P2."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {num_games} games per matchup")
    print(f"{'='*60}")

    for opp_name in opponents:
        # Agent as Player 1
        t0 = time.time()
        outcomes_p1 = evaluate("connectx", [agent, opp_name], num_episodes=num_games)
        t1 = time.time()
        wins_p1 = sum(1 for r in outcomes_p1 if r[0] == 1)
        losses_p1 = sum(1 for r in outcomes_p1 if r[1] == 1)
        draws_p1 = num_games - wins_p1 - losses_p1

        print(f"\nvs {opp_name:12s} (as P1):  "
              f"W={wins_p1:2d}/{num_games}  L={losses_p1:2d}/{num_games}  "
              f"D={draws_p1:2d}/{num_games}  "
              f"WR={wins_p1/num_games*100:.0f}%  "
              f"avg={ (t1-t0)/num_games*1000:.0f}ms/game")

        # Agent as Player 2
        t0 = time.time()
        outcomes_p2 = evaluate("connectx", [opp_name, agent], num_episodes=num_games)
        t1 = time.time()
        wins_p2 = sum(1 for r in outcomes_p2 if r[1] == 1)
        losses_p2 = sum(1 for r in outcomes_p2 if r[0] == 1)
        draws_p2 = num_games - wins_p2 - losses_p2

        print(f"vs {opp_name:12s} (as P2):  "
              f"W={wins_p2:2d}/{num_games}  L={losses_p2:2d}/{num_games}  "
              f"D={draws_p2:2d}/{num_games}  "
              f"WR={wins_p2/num_games*100:.0f}%  "
              f"avg={ (t1-t0)/num_games*1000:.0f}ms/game")


def main():
    print("AlphaZero ConnectX Agent — Comprehensive Evaluation")
    print(f"Model: {kaggle_agent._cached_config or 'will load on first call'}")

    # Evaluate against random and negamax
    eval_matchups(kaggle_agent.agent, ["random", "negamax"], num_games=40)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
