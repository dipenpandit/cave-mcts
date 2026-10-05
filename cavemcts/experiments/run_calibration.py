"""Fit the per-role logistic that maps a raw evaluation to a win probability.

Positions are sampled from self-play games, each labelled with the eventual
outcome of the game it came from, and a sigmoid is fitted per role:

    q(E) = 1 / (1 + exp(-beta * (E - mu)))

This is the same procedure chess engines use to map centipawns to an expected
score.  The fitted parameters are written to results/calibration.json and can
be loaded with `LogisticCalibrator.load`.

    python -m experiments.run_calibration --games 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
from cavemcts.arena import MAX_PLIES
from cavemcts.game import (
    GOAT,
    ONGOING,
    State,
    TIGER,
    apply_move,
    legal_moves,
    outcome,
    outcome_to_unit,
)
from cavemcts.heuristics import evaluate_raw, fit_logistic
from cavemcts.mcts import MCTS, RandomAgent
from cavemcts.policies import CAVEPolicy
from cavemcts.heuristics import OnlineCalibrator

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def collect(games: int, sims: int, sample_every: int, seed: int):
    """Play games with mixed-strength agents and record (raw eval, outcome)."""
    data = {TIGER: ([], []), GOAT: ([], [])}
    rng = random.Random(seed)

    for g in range(games):
        # Mixing strengths keeps the sampled positions from collapsing onto a
        # single style of play, which would bias the fit.
        strong = g % 2 == 0
        goat_agent = (
            MCTS(CAVEPolicy(OnlineCalibrator()), sims, 10, seed=seed + g)
            if strong else RandomAgent(seed + g)
        )
        tiger_agent = (
            RandomAgent(seed + 5000 + g)
            if strong else MCTS(CAVEPolicy(OnlineCalibrator()), sims, 10, seed=seed + g)
        )

        state = State.initial()
        sampled = []
        plies = 0

        while plies < MAX_PLIES:
            result = outcome(state)
            if result is not ONGOING:
                break
            # Sample stochastically: a fixed stride would lock onto one
            # parity of the ply counter and only ever see a single role.
            if rng.random() < 1.0 / sample_every:
                sampled.append((state, state.turn))
            agent = goat_agent if state.turn == GOAT else tiger_agent
            move = agent.search(state)
            if move is None:
                break
            state = apply_move(state, move)
            plies += 1

        final = outcome(state)
        if final is ONGOING:
            final = 0  # unresolved games are scored as draws

        for snapshot, player in sampled:
            raws, ys = data[player]
            raws.append(evaluate_raw(snapshot, player))
            ys.append(outcome_to_unit(final, player))

        print(f"  game {g + 1}/{games}: {len(sampled)} samples, outcome {final}", flush=True)

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    data = collect(args.games, args.sims, args.sample_every, args.seed)

    params = {}
    for role, label in ((TIGER, "tiger"), (GOAT, "goat")):
        raws, ys = data[role]
        beta, mu = fit_logistic(raws, ys)
        params[role] = (beta, mu)
        print(f"{label:>6}: n={len(raws):5d}  beta={beta:.6f}  mu={mu:9.2f}  "
              f"base rate={sum(ys) / max(len(ys), 1):.3f}")

    path = os.path.join(RESULTS, "calibration.json")
    with open(path, "w") as fh:
        json.dump({str(k): list(v) for k, v in params.items()}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
