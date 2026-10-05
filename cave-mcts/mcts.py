from __future__ import annotations
import random
from .game import ONGOING, apply_move, legal_moves, outcome, outcome_to_unit



class Node:
    __slots__ = (
        "state",
        "player",
        "parent",
        "move",
        "children",
        "untried",
        "visits",
        "value_sum",
        "value_sq_sum",
        "minimax_value",
    )

    def __init__(self, state, parent=None, move=None, rng=None):
        self.state = state
        self.player = state.turn
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = legal_moves(state)
        if rng is not None:
            rng.shuffle(self.untried)
        self.visits = 0
        self.value_sum = 0.0
        self.value_sq_sum = 0.0
        self.minimax_value = None

    @property
    def fully_expanded(self) -> bool:
        return not self.untried and bool(self.children)

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    """Monte Carlo Tree Search with a pluggable selection policy.

    Args:
        policy: a `cavemcts.policies.Policy`.
        simulations: playouts per move.
        max_depth: Early Playout Termination depth.  `None` means playouts
            run to a terminal state (the pure-MCTS baseline).
        seed: RNG seed for reproducible runs.
    """

    def __init__(self, policy, simulations: int = 1000, max_depth=10, seed=None):
        self.policy = policy
        self.simulations = simulations
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.last_root = None


    def search(self, state):
        moves = legal_moves(state)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        self.policy.reset()
        root = Node(state, rng=self.rng)

        for _ in range(self.simulations):
            node = self._select(root)
            result = outcome(node.state)

            if result is not ONGOING:
                value = outcome_to_unit(result, node.player)
                self._backpropagate(node, value, terminal=True)
            else:
                node = self._expand(node)
                value = self._simulate(node)
                self._backpropagate(node, value)

        self.last_root = root
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _select(self, node):
        while node.fully_expanded:
            node = max(node.children, key=lambda c: self.policy.score(node, c))
        return node

    def _expand(self, node):
        if not node.untried:
            return node
        move = node.untried.pop()
        child = Node(apply_move(node.state, move), parent=node, move=move, rng=self.rng)
        node.children.append(child)
        return child

    def _simulate(self, node) -> float:
        """Playout from `node`, returning a value in `node.player`'s frame."""
        state = node.state
        depth = 0

        while True:
            result = outcome(state)
            if result is not ONGOING:
                return outcome_to_unit(result, node.player)
            if self.max_depth is not None and depth >= self.max_depth:
                value = self.policy.leaf_value(state, state.turn)
                return value if state.turn == node.player else self.policy.flip(value)

            moves = legal_moves(state)
            if not moves:
                return 0.5
            state = apply_move(state, self.rng.choice(moves))
            depth += 1

    def _backpropagate(self, node, value: float, terminal: bool = False) -> None:
        use_minimax = self.policy.uses_implicit_minimax

        if use_minimax and node.minimax_value is None:
            node.minimax_value = (
                value if terminal else self.policy.leaf_value(node.state, node.player)
            )

        while node is not None:
            node.visits += 1
            node.value_sum += value
            node.value_sq_sum += value * value

            if use_minimax and node.children:
                node.minimax_value = max(
                    self.policy.flip(c.minimax_value)
                    for c in node.children
                    if c.minimax_value is not None
                )

            node = node.parent
            value = self.policy.flip(value)


class RandomAgent:
    """Uniform random legal move.  A floor to measure everything against."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def search(self, state):
        moves = legal_moves(state)
        return self.rng.choice(moves) if moves else None
