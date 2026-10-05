"""Raw evaluation functions and the calibration layer that bounds them.

Two things live here, and keeping them apart is the point of the module.

`evaluate_raw(state, player)` returns an unbounded score on an arbitrary
scale, exactly like the handcrafted evaluators in the original MCTS-EPT
implementation.  This is the quantity that, when fed straight into the UCT
exploitation term, destroys the [0, 1] assumption behind UCB1's regret bound.

`Calibrator` maps that raw score onto [0, 1] so it is commensurate with a
win rate.  Two mappings are provided:
1. `LogisticCalibrator` - a fitted sigmoid, q = 1 / (1 + exp(-b (E - m))),
    with (b, m) estimated per role by logistic regression against observed
    game outcomes.  This is the chess centipawn -> win-probability recipe.
2. `OnlineCalibrator` - the same sigmoid, but with (m, s) tracked online as
    a running mean and standard deviation of the raw scores actually seen
    during the current search.  Needs no offline fitting and adapts when the
    scale of the evaluation drifts between game phases.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .game import (
    ADJACENCY,
    EMPTY,
    GOAT,
    JUMPS,
    N,
    NUM_POINTS,
    TIGER,
    TOTAL_GOATS,
    blocked_tigers,
)

CENTRE = 12
STRONG_POINTS = (6, 8, 12, 16, 18)


def _threat_count(state) -> int:
    """Number of immediate capture jumps available to the tigers."""
    board = state.board
    threats = 0
    for src in state.tigers:
        for mid, dest in JUMPS[src]:
            if board[mid] == GOAT and board[dest] == EMPTY:
                threats += 1
    return threats


def _goat_support(state) -> int:
    """Goats that have at least one adjacent friendly goat."""
    board = state.board
    supported = 0
    for p in range(NUM_POINTS):
        if board[p] == GOAT and any(board[q] == GOAT for q in ADJACENCY[p]):
            supported += 1
    return supported


def _tiger_mobility(state) -> int:
    board = state.board
    return sum(
        1
        for src in state.tigers
        for dest in ADJACENCY[src]
        if board[dest] == EMPTY
    )


def evaluate_tiger_raw(state) -> float:
    """Unbounded score, larger is better for the tigers."""
    score = 0.0
    score += 100.0 * state.goats_captured
    score += 50.0 * _threat_count(state)
    score += 30.0 * sum(1 for t in state.tigers if t in STRONG_POINTS)
    score += 8.0 * _tiger_mobility(state)
    score -= 45.0 * blocked_tigers(state)
    score -= 12.0 * _goat_support(state)
    return score


def evaluate_goat_raw(state) -> float:
    """Unbounded score, larger is better for the goats."""
    score = 0.0
    score += 50.0 * blocked_tigers(state)
    score -= 60.0 * _threat_count(state)
    score -= 80.0 * state.goats_captured
    score += 25.0 * _goat_support(state)
    score -= 40.0 * sum(1 for t in state.tigers if t in STRONG_POINTS)
    score -= 6.0 * _tiger_mobility(state)
    score -= 3.0 * state.goats_in_hand
    return score


def evaluate_raw(state, player: int) -> float:
    """Raw evaluation from `player`'s point of view."""
    return evaluate_tiger_raw(state) if player == TIGER else evaluate_goat_raw(state)


def phase(state) -> float:
    """0.0 at the start of placement, 1.0 once every goat is on the board."""
    return 1.0 - state.goats_in_hand / TOTAL_GOATS


# Calibration

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Calibrator:
    """Base class: map a raw evaluation to a bounded value in [0, 1]."""

    def __call__(self, state, player: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


class IdentityCalibrator(Calibrator):
    """No calibration at all - reproduces the unbounded baseline."""

    def __call__(self, state, player: int) -> float:
        return evaluate_raw(state, player)

    def __repr__(self) -> str:
        return "IdentityCalibrator()"


class LogisticCalibrator(Calibrator):
    """Fitted per-role sigmoid, q = sigmoid(beta * (E - mu))."""

    def __init__(self, params=None):
        # role -> (beta, mu); defaults are sane priors before any fitting.
        self.params = params or {TIGER: (0.006, 0.0), GOAT: (0.006, 0.0)}

    def __call__(self, state, player: int) -> float:
        beta, mu = self.params[player]
        return _sigmoid(beta * (evaluate_raw(state, player) - mu))

    @classmethod
    def load(cls, path: str) -> "LogisticCalibrator":
        with open(path) as fh:
            blob = json.load(fh)
        return cls({int(k): tuple(v) for k, v in blob.items()})

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({str(k): list(v) for k, v in self.params.items()}, fh, indent=2)

    def __repr__(self) -> str:
        return f"LogisticCalibrator({self.params})"


class OnlineCalibrator(Calibrator):
    """Sigmoid over a z-score computed from statistics gathered online.

    Keeps a running mean and variance of the raw scores seen for each role
    during the current search (Welford's algorithm) and maps

        q = sigmoid(k * (E - mean) / std).

    Because the location and scale are re-estimated continuously, the mapped
    value stays spread across [0, 1] whatever the magnitude of the raw
    evaluation, which is what makes a single exploration constant meaningful
    across game phases.
    """

    def __init__(self, k: float = 1.0, warmup: int = 30):
        self.k = k
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self._n = {TIGER: 0, GOAT: 0}
        self._mean = {TIGER: 0.0, GOAT: 0.0}
        self._m2 = {TIGER: 0.0, GOAT: 0.0}

    def observe(self, raw: float, player: int) -> None:
        self._n[player] += 1
        delta = raw - self._mean[player]
        self._mean[player] += delta / self._n[player]
        self._m2[player] += delta * (raw - self._mean[player])

    def std(self, player: int) -> float:
        if self._n[player] < 2:
            return 1.0
        return max(math.sqrt(self._m2[player] / (self._n[player] - 1)), 1e-6)

    def __call__(self, state, player: int) -> float:
        raw = evaluate_raw(state, player)
        self.observe(raw, player)
        if self._n[player] < self.warmup:
            return 0.5
        return _sigmoid(self.k * (raw - self._mean[player]) / self.std(player))

    def __repr__(self) -> str:
        return f"OnlineCalibrator(k={self.k})"


def fit_logistic(raw_scores, outcomes, iters: int = 400, lr: float = 0.5):
    """Fit q = sigmoid(beta * (E - mu)) by gradient ascent on log-likelihood.

    `outcomes` are in [0, 1] (1 win, 0.5 draw, 0 loss) from the same point of
    view as `raw_scores`.  Standardising first keeps the optimisation stable
    and lets us report (beta, mu) on the original scale.
    """
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if raw.size == 0:
        return 0.006, 0.0

    mean, std = raw.mean(), max(raw.std(), 1e-6)
    z = (raw - mean) / std
    w, b = 1.0, 0.0

    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(w * z + b)))
        err = y - p
        w += lr * float((err * z).mean())
        b += lr * float(err.mean())

    beta = w / std
    mu = mean - b / w if abs(w) > 1e-9 else mean
    return float(beta), float(mu)
