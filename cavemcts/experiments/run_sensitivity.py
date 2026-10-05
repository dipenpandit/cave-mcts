"""Sweep the exploration constant and watch which rules actually respond.

The claim under test is not "CAVE wins more" but something sharper: under the
prior rule the exploration constant barely matters, because the term it scales
is numerically swamped.  A rule whose performance is flat in C is a rule whose
exploration is not doing anything.

Once the value is bounded, C becomes a real knob: performance should vary with
it and peak somewhere sensible.

    python -m experiments.run_sensitivity --games 12 --sims 300
"""

from __future__ import annotations

import argparse
import json
import os
from cavemcts.arena import play_match
from cavemcts.game import GOAT, TIGER
from cavemcts.heuristics import LogisticCalibrator, OnlineCalibrator
from cavemcts.mcts import MCTS
from cavemcts.policies import CAVEPolicy, RawHeuristicUCT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--calibration", default=os.path.join(RESULTS, "calibration.json"))
    args = ap.parse_args()

    if os.path.exists(args.calibration):
        params = LogisticCalibrator.load(args.calibration).params
        calibrator_factory = lambda: LogisticCalibrator(dict(params))
    else:
        calibrator_factory = lambda: OnlineCalibrator()

    # A fixed, C-independent yardstick, so the only thing moving is C.
    reference = lambda seed: MCTS(RawHeuristicUCT(c=1.2), args.sims, args.depth, seed=seed)

    families = {
        "raw-heuristic-uct": lambda c: (
            lambda seed: MCTS(RawHeuristicUCT(c=c), args.sims, args.depth, seed=seed)
        ),
        "cave": lambda c: (
            lambda seed: MCTS(CAVEPolicy(calibrator_factory(), alpha=0.3,
                                         c_tiger=c, c_goat=c),
                              args.sims, args.depth, seed=seed)
        ),
    }

    out = {}
    for name, make in families.items():
        print(f"\n{name}")
        out[name] = {}
        for c in C_VALUES:
            factory = make(c)
            scores = {}
            for role, role_name in ((TIGER, "tiger"), (GOAT, "goat")):
                result = play_match(factory, reference, role, args.games, seed=args.seed)
                scores[role_name] = result.score
            spread_note = f"C={c:<5} tiger={scores['tiger']:.3f}  goat={scores['goat']:.3f}"
            print("  " + spread_note, flush=True)
            out[name][c] = scores

        for role in ("tiger", "goat"):
            vals = [out[name][c][role] for c in C_VALUES]
            print(f"  range over C ({role}): {max(vals) - min(vals):.3f}")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "sensitivity.json")
    with open(path, "w") as fh:
        json.dump({"config": vars(args), "results": out}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
