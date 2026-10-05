"""Match play, result bookkeeping and Wilson confidence intervals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .game import DRAW, GOAT, GOAT_WIN, ONGOING, TIGER, TIGER_WIN, State, apply_move, outcome

MAX_PLIES = 400


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    plies: list = field(default_factory=list)
    seconds_per_move: list = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Win rate counting a draw as a half point."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def wilson(self, z: float = 1.96):
        """Wilson score interval for `score`, treating draws as half wins."""
        n, p = self.games, self.score
        if n == 0:
            return (0.0, 1.0)
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def summary(self, label: str = "") -> str:
        lo, hi = self.wilson()
        avg = sum(self.seconds_per_move) / len(self.seconds_per_move) if self.seconds_per_move else 0.0
        return (
            f"{label:<34} {self.wins:>3}W {self.draws:>3}D {self.losses:>3}L  "
            f"score={self.score:5.3f}  95% CI [{lo:5.3f}, {hi:5.3f}]  "
            f"{avg * 1000:6.1f} ms/move"
        )


def play_game(goat_agent, tiger_agent, max_plies: int = MAX_PLIES, timings=None):
    """Play one game.  Returns GOAT_WIN, TIGER_WIN or DRAW."""
    state = State.initial()
    plies = 0

    while plies < max_plies:
        result = outcome(state)
        if result is not ONGOING:
            return result, plies

        agent = goat_agent if state.turn == GOAT else tiger_agent
        start = time.perf_counter()
        move = agent.search(state)
        if timings is not None:
            timings.append(time.perf_counter() - start)
        if move is None:
            return (TIGER_WIN if state.turn == GOAT else GOAT_WIN), plies

        state = apply_move(state, move)
        plies += 1

    return DRAW, plies


def play_match(agent_factory, opponent_factory, role: int, games: int,
               seed: int = 0, verbose: bool = False) -> MatchResult:
    """Play `games` games with the agent under test fixed to `role`.

    Each game gets its own seed so that the two sides are never handed the
    same random stream, and so a run is reproducible from `seed` alone.
    """
    result = MatchResult()

    for g in range(games):
        agent = agent_factory(seed + g)
        opponent = opponent_factory(10_000 + seed + g)
        timings = []

        if role == GOAT:
            code, plies = play_game(agent, opponent, timings=timings)
            won = code == GOAT_WIN
        else:
            code, plies = play_game(opponent, agent, timings=timings)
            won = code == TIGER_WIN

        if code == DRAW:
            result.draws += 1
        elif won:
            result.wins += 1
        else:
            result.losses += 1

        result.plies.append(plies)
        result.seconds_per_move.extend(timings)

        if verbose:
            side = "goat" if role == GOAT else "tiger"
            print(f"  game {g + 1}/{games} as {side}: "
                  f"{'win' if won else 'draw' if code == DRAW else 'loss'} "
                  f"({plies} plies)", flush=True)

    return result
