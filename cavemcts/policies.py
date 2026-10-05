"""Selection policies.

Every policy answers three questions for the search driver:

1. `leaf_value(state, player)` - what a depth-limited playout is worth.
2. `flip(v)` - how a value is re-expressed in the opponent's frame.
3. `score(parent, child)` - the selection index, in the parent's frame.

Separating these lets one MCTS driver run every variant, so an ablation
changes exactly one object and nothing else.
"""

from __future__ import annotations

import math

from .game import GOAT, TIGER
from .heuristics import evaluate_raw, phase


class MinMaxStats:
    """Running min/max of the values seen inside one search tree."""

    def __init__(self):
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: float) -> None:
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.5

    def reset(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf


class Policy:
    """Base class.  Defaults describe a bounded [0, 1] value space."""

    name = "policy"
    bounded = True
    uses_implicit_minimax = False

    def flip(self, v: float) -> float:
        return 1.0 - v if self.bounded else -v

    def reset(self) -> None:
        pass

    def leaf_value(self, state, player: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def exploitation(self, parent, child) -> float:  # pragma: no cover
        """Value of `child` expressed in `parent`'s frame."""
        raise NotImplementedError

    def bonus(self, parent, child) -> float:  # pragma: no cover
        """Exploration bonus for `child`."""
        raise NotImplementedError

    def score(self, parent, child) -> float:
        return self.exploitation(parent, child) + self.bonus(parent, child)

    def _uct_radius(self, parent, child) -> float:
        return math.sqrt(2.0 * math.log(parent.visits) / child.visits)

    def __repr__(self) -> str:
        return self.name


class RandomPlayoutUCT(Policy):
    """Textbook MCTS: random playouts to a terminal state, UCT selection.

    This is the Baseline-MCTS of the original paper.  Values are win rates,
    so they genuinely lie in [0, 1] and UCB1's assumptions hold.
    """

    name = "baseline-uct"

    def __init__(self, c: float = 1.41):
        self.c = c

    def leaf_value(self, state, player: int) -> float:
        # Never called: the driver runs playouts to termination for this
        # policy.  Returning a draw keeps depth-capped playouts neutral.
        return 0.5

    def exploitation(self, parent, child) -> float:
        return self.flip(child.value_sum / child.visits)

    def bonus(self, parent, child) -> float:
        return self.c * self._uct_radius(parent, child)


class RawHeuristicUCT(Policy):
    """The prior paper's MCTS-EPT rule, reproduced faithfully.

        UCT_enhanced = E_avg(j) + C_p * sqrt(2 ln n / n_j)

    `E_avg` is an average of raw, unnormalised heuristic scores that routinely
    run into the hundreds, while the exploration bonus is O(1).  The
    exploration term is therefore numerically irrelevant almost everywhere,
    which is the defect the rest of this package is about.
    """

    name = "raw-heuristic-uct"
    bounded = False

    def __init__(self, c: float = 1.2):
        self.c = c

    def leaf_value(self, state, player: int) -> float:
        return evaluate_raw(state, player)

    def exploitation(self, parent, child) -> float:
        return self.flip(child.value_sum / child.visits)

    def bonus(self, parent, child) -> float:
        return self.c * self._uct_radius(parent, child)


class MinMaxNormalizedUCT(Policy):
    """Raw heuristics, rescued by MuZero-style min-max normalisation.

    Q is mapped into [0, 1] using the smallest and largest value seen so far
    in this tree, so the exploration bonus becomes commensurate again without
    any calibration data.  A strong, cheap ablation control.
    """

    name = "minmax-uct"
    bounded = False

    def __init__(self, c: float = 1.2):
        self.c = c
        self.stats = MinMaxStats()

    def reset(self) -> None:
        self.stats.reset()

    def leaf_value(self, state, player: int) -> float:
        return evaluate_raw(state, player)

    def exploitation(self, parent, child) -> float:
        q = self.flip(child.value_sum / child.visits)
        self.stats.update(q)
        return self.stats.normalize(q)

    def bonus(self, parent, child) -> float:
        return self.c * self._uct_radius(parent, child)


class CalibratedUCT(Policy):
    """Calibrated heuristics in [0, 1], otherwise plain UCT.

    Isolates the contribution of calibration alone, before any variance
    awareness or minimax backup is added.
    """

    name = "calibrated-uct"

    def __init__(self, calibrator, c: float = 1.41):
        self.calibrator = calibrator
        self.c = c

    def reset(self) -> None:
        self.calibrator.reset()

    def leaf_value(self, state, player: int) -> float:
        return self.calibrator(state, player)

    def exploitation(self, parent, child) -> float:
        return self.flip(child.value_sum / child.visits)

    def bonus(self, parent, child) -> float:
        return self.c * self._uct_radius(parent, child)


class CAVEPolicy(Policy):
    """Calibrated, Asymmetric, Variance-aware Exploration.

    Three changes to the selection rule, each answering one failure of the
    raw-heuristic UCT above.

    1. Calibration.  The exploitation channel is a calibrated value in
       [0, 1], so the exploration bonus is on the same scale as the value it
       is competing with, and C recovers a meaning.

    2. Dual-channel exploitation (implicit minimax, Lanctot et al. 2014).
       The sampled average and a minimax backup of the calibrated heuristic
       are kept as separate channels and blended,

           Q_hat = (1 - alpha) * Q_sampled + alpha * V_minimax,

       so tactical information propagates without contaminating the
       statistical estimate.

    3. Asymmetric, variance-aware exploration.  The confidence radius is
       Bernstein-style (UCB-V), driven by the observed variance of the child
       rather than a constant:

           U = c_role * [ sqrt(2 V_j ln n / n_j) + 3 ln n / n_j ]

       A node whose playouts agree stops being explored quickly; a node whose
       playouts disagree keeps its budget.  Because tigers and goats face
       different branching factors and different horizons, c_role is set per
       role, and optionally annealed with game phase.
    """

    name = "cave"
    uses_implicit_minimax = True

    def __init__(
        self,
        calibrator,
        alpha: float = 0.3,
        c_tiger: float = 0.9,
        c_goat: float = 1.3,
        phase_anneal: float = 0.0,
        use_variance: bool = True,
    ):
        self.calibrator = calibrator
        self.alpha = alpha
        self.c = {TIGER: c_tiger, GOAT: c_goat}
        self.phase_anneal = phase_anneal
        self.use_variance = use_variance

    def reset(self) -> None:
        self.calibrator.reset()

    def leaf_value(self, state, player: int) -> float:
        return self.calibrator(state, player)

    def _c_for(self, parent) -> float:
        c = self.c[parent.player]
        if self.phase_anneal:
            c *= 1.0 - self.phase_anneal * phase(parent.state)
        return c

    def exploitation(self, parent, child) -> float:
        q = self.flip(child.value_sum / child.visits)
        if self.alpha > 0.0 and child.minimax_value is not None:
            q = (1.0 - self.alpha) * q + self.alpha * self.flip(child.minimax_value)
        return q

    def bonus(self, parent, child) -> float:
        log_n = math.log(parent.visits)
        if self.use_variance:
            mean = child.value_sum / child.visits
            var = max(child.value_sq_sum / child.visits - mean * mean, 0.0)
            u = math.sqrt(2.0 * var * log_n / child.visits) + 3.0 * log_n / child.visits
        else:
            u = math.sqrt(2.0 * log_n / child.visits)
        return self._c_for(parent) * u


