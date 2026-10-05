"""Measure the defect directly, before measuring any fix.

For each selection rule we run one search and record, at the root, the spread
of the exploitation term across children and the size of the exploration
bonus.  The ratio

    exploration influence = spread(bonus) / spread(exploitation)

says how much say the exploration term actually has in the argmax.  When that
ratio is far below 1, the exploration term is decorative: the rule is pure
greedy exploitation no matter what C is set to.

    python -m experiments.run_diagnostics
"""

from __future__ import annotations

import argparse
import random
from cavemcts.game import ONGOING, State, apply_move, legal_moves, outcome
from cavemcts.heuristics import OnlineCalibrator
from cavemcts.mcts import MCTS
from cavemcts.policies import (
    CAVEPolicy,
    CalibratedUCT,
    MinMaxNormalizedUCT,
    RawHeuristicUCT,
)


def sample_positions(n: int, seed: int):
    """A handful of mid-game positions reached by random play."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < n:
        state = State.initial()
        target = rng.randint(12, 30)
        for _ in range(target):
            if outcome(state) is not ONGOING:
                break
            moves = legal_moves(state)
            if not moves:
                break
            state = apply_move(state, rng.choice(moves))
        if outcome(state) is ONGOING:
            positions.append(state)
    return positions


def measure(policy, state, sims, seed):
    engine = MCTS(policy, simulations=sims, max_depth=10, seed=seed)
    engine.search(state)
    root = engine.last_root

    exploit, bonus = [], []
    for child in root.children:
        if child.visits == 0:
            continue
        exploit.append(policy.exploitation(root, child))
        bonus.append(policy.bonus(root, child))

    def spread(xs):
        return max(xs) - min(xs) if len(xs) > 1 else 0.0

    return spread(exploit), spread(bonus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    positions = sample_positions(args.positions, args.seed)

    policies = [
        ("raw-heuristic-uct (prior work)",  lambda: RawHeuristicUCT(c=1.2)),
        ("minmax-uct",                      lambda: MinMaxNormalizedUCT(c=1.2)),
        ("calibrated-uct",                  lambda: CalibratedUCT(OnlineCalibrator())),
        ("cave",                            lambda: CAVEPolicy(OnlineCalibrator())),
    ]

    print(f"{'policy':<34}{'spread(Q)':>12}{'spread(bonus)':>15}{'ratio':>10}")
    print("-" * 71)

    for label, factory in policies:
        qs, bs = [], []
        for i, state in enumerate(positions):
            q, b = measure(factory(), state, args.sims, args.seed + i)
            qs.append(q)
            bs.append(b)
        mq = sum(qs) / len(qs)
        mb = sum(bs) / len(bs)
        ratio = mb / mq if mq > 1e-12 else float("inf")
        print(f"{label:<34}{mq:12.4f}{mb:15.4f}{ratio:10.4f}")

    print(
        "\nA ratio far below 1 means the exploration term cannot change the "
        "argmax:\nthe rule is effectively greedy whatever C is set to."
    )


if __name__ == "__main__":
    main()
