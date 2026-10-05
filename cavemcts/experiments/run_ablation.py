"""Ablation grid: every variant plays both roles against a fixed reference.

Because Baaghchaal is asymmetric, a single aggregate win rate hides the thing
we most want to see.  Every row is therefore reported twice, once as tiger and
once as goat, each with a Wilson interval.

The reference opponent is the prior work's rule (raw heuristic in the UCT
exploitation term), so a score above 0.5 is a gain over the published method
at an identical simulation budget.

python -m experiments.run_ablation --games 20 --sims 400
"""

from __future__ import annotations

import argparse
import json
import os
from cavemcts.arena import play_match
from cavemcts.game import GOAT, TIGER
from cavemcts.heuristics import LogisticCalibrator, OnlineCalibrator
from cavemcts.mcts import MCTS, RandomAgent
from cavemcts.policies import (
    CAVEPolicy,
    CalibratedUCT,
    MinMaxNormalizedUCT,
    RandomPlayoutUCT,
    RawHeuristicUCT,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def build_variants(sims, depth, calibrator_factory):
    """Each entry adds exactly one ingredient to the one above it."""

    def engine(policy_factory):
        return lambda seed: MCTS(policy_factory(), sims, depth, seed=seed)

    return [
        ("random",
         lambda seed: RandomAgent(seed)),

        ("baseline-uct (no heuristic)",
         lambda seed: MCTS(RandomPlayoutUCT(c=1.41), sims, None, seed=seed)),

        ("raw-heuristic-uct (prior work)",
         engine(lambda: RawHeuristicUCT(c=1.2))),

        ("+ minmax normalisation",
         engine(lambda: MinMaxNormalizedUCT(c=1.2))),

        ("+ calibration",
         engine(lambda: CalibratedUCT(calibrator_factory(), c=1.41))),

        ("+ variance-aware bonus",
         engine(lambda: CAVEPolicy(calibrator_factory(), alpha=0.0,
                                   c_tiger=1.0, c_goat=1.0))),

        ("+ implicit minimax (alpha=0.3)",
         engine(lambda: CAVEPolicy(calibrator_factory(), alpha=0.3,
                                   c_tiger=1.0, c_goat=1.0))),

        ("CAVE (full, asymmetric C)",
         engine(lambda: CAVEPolicy(calibrator_factory(), alpha=0.3,
                                   c_tiger=0.9, c_goat=1.3))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games per role")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--calibration", default=os.path.join(RESULTS, "calibration.json"))
    args = ap.parse_args()

    if os.path.exists(args.calibration):
        params = LogisticCalibrator.load(args.calibration).params
        calibrator_factory = lambda: LogisticCalibrator(dict(params))
        print(f"using fitted calibration from {args.calibration}\n")
    else:
        calibrator_factory = lambda: OnlineCalibrator()
        print("no fitted calibration found, falling back to OnlineCalibrator\n")

    reference = lambda seed: MCTS(RawHeuristicUCT(c=1.2), args.sims, args.depth, seed=seed)
    variants = build_variants(args.sims, args.depth, calibrator_factory)

    print(f"{args.games} games per role, {args.sims} simulations per move, "
          f"d_max={args.depth}")
    print("opponent: raw-heuristic-uct (prior work). Score counts a draw as 0.5.\n")

    table = {}
    for label, factory in variants:
        row = {}
        for role, role_name in ((TIGER, "as tiger"), (GOAT, "as goat ")):
            result = play_match(factory, reference, role, args.games, seed=args.seed)
            print(result.summary(f"{label} {role_name}"), flush=True)
            lo, hi = result.wilson()
            row[role_name.strip()] = {
                "wins": result.wins, "draws": result.draws, "losses": result.losses,
                "score": result.score, "ci": [lo, hi],
            }
        table[label] = row
        print()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "ablation.json")
    with open(path, "w") as fh:
        json.dump({"config": vars(args), "results": table}, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
