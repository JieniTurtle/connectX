#!/usr/bin/python3
"""Quick evaluation of kaggle_agent against built-in opponents."""
import sys
import os
import time

# Ensure the alphazero package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "alphazero-board-games"))

from kaggle_environments import make, evaluate

# Import the agent so the module-level model cache kicks in
import kaggle_agent

def main():
    print("=" * 60)
    print("AlphaZero Kaggle Agent — Quick Evaluation")
    print("=" * 60)

    # --- 1. Single game vs random (with render) ---
    print("\n[1/3] Single game vs 'random'…")
    start = time.time()
    env = make("connectx", debug=True)
    env.run([kaggle_agent.agent, "random"])
    elapsed = time.time() - start
    print(env.render(mode="human", width=500, height=450))
    print(f"  Time: {elapsed:.1f}s")

    # --- 2. 10 games vs random ---
    print("\n[2/3] 10 games vs 'random'…")
    start = time.time()
    outcomes = evaluate("connectx", [kaggle_agent.agent, "random"], num_episodes=10)
    elapsed = time.time() - start
    print(f"  Results: {outcomes}")
    print(f"  Time: {elapsed:.1f}s (avg {elapsed/10:.1f}s/game)")

    # --- 3. 10 games vs negamax ---
    print("\n[3/3] 10 games vs 'negamax'…")
    start = time.time()
    outcomes = evaluate("connectx", [kaggle_agent.agent, "negamax"], num_episodes=10)
    elapsed = time.time() - start
    print(f"  Results: {outcomes}")
    print(f"  Time: {elapsed:.1f}s (avg {elapsed/10:.1f}s/game)")

    print("\nDone!")

if __name__ == "__main__":
    main()
