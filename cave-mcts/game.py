"""Baaghchaal (Bagh Chal) game engine.

Board is a flat length-25 int array, index = 5*row + col.

    +1  goat
    -1  tiger
     0  empty

Rules implemented (Lim & Nievergelt rule set):
1.  Goats move first and must place all 20 goats before any goat may slide.
2.  Tigers win on 5 captures; goats win when no tiger has a legal move.
3.  Draw on three-fold repetition or 50 plies without progress
    (progress = a capture or a placement).
"""

from __future__ import annotations

import numpy as np

N = 5
NUM_POINTS = N * N
EMPTY, GOAT, TIGER = 0, 1, -1

TOTAL_GOATS = 20
CAPTURES_TO_WIN = 5
CORNERS = (0, 4, 20, 24)

NO_PROGRESS_LIMIT = 50
REPETITION_LIMIT = 3

# Outcome codes, always from the goat's point of view.
GOAT_WIN, TIGER_WIN, DRAW, ONGOING = 1, -1, 0, None


def _build_topology():
    """Precompute adjacency and jump tables once at import time."""
    ortho = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    adjacency = [[] for _ in range(NUM_POINTS)]
    jumps = [[] for _ in range(NUM_POINTS)]

    for r in range(N):
        for c in range(N):
            src = r * N + c
            directions = ortho + diag if (r + c) % 2 == 0 else ortho
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    adjacency[src].append(nr * N + nc)
                jr, jc = r + 2 * dr, c + 2 * dc
                if 0 <= jr < N and 0 <= jc < N:
                    mid = (r + dr) * N + (c + dc)
                    jumps[src].append((mid, jr * N + jc))

    return (
        tuple(tuple(a) for a in adjacency),
        tuple(tuple(j) for j in jumps),
    )


ADJACENCY, JUMPS = _build_topology()


class State:
    """A complete Baaghchaal position."""

    __slots__ = (
        "board",
        "turn",
        "goats_in_hand",
        "goats_captured",
        "tigers",
        "plies_without_progress",
        "history",
    )

    def __init__(self, board, turn, goats_in_hand, goats_captured, tigers,
                 plies_without_progress=0, history=None):
        self.board = board
        self.turn = turn
        self.goats_in_hand = goats_in_hand
        self.goats_captured = goats_captured
        self.tigers = tigers
        self.plies_without_progress = plies_without_progress
        self.history = history if history is not None else {}

    @staticmethod
    def initial() -> "State":
        board = np.zeros(NUM_POINTS, dtype=np.int8)
        for corner in CORNERS:
            board[corner] = TIGER
        state = State(board, GOAT, TOTAL_GOATS, 0, CORNERS)
        state.history = {state.key(): 1}
        return state

    def key(self):
        return (self.board.tobytes(), self.turn, self.goats_in_hand)

    def clone(self) -> "State":
        return State(
            self.board.copy(),
            self.turn,
            self.goats_in_hand,
            self.goats_captured,
            self.tigers,
            self.plies_without_progress,
            dict(self.history),
        )

    @property
    def goats_on_board(self) -> int:
        return int(np.count_nonzero(self.board == GOAT))

    @property
    def in_placement_phase(self) -> bool:
        return self.goats_in_hand > 0

    def __str__(self) -> str:
        glyph = {EMPTY: ".", GOAT: "G", TIGER: "T"}
        rows = [
            " ".join(glyph[int(v)] for v in self.board[r * N:(r + 1) * N])
            for r in range(N)
        ]
        return "\n".join(rows)


# Move generation.  A move is the tuple (src, dest); src == -1 is a placement.

def tiger_moves(state: State):
    board = state.board
    moves = []
    for src in state.tigers:
        for dest in ADJACENCY[src]:
            if board[dest] == EMPTY:
                moves.append((src, dest))
        for mid, dest in JUMPS[src]:
            if board[mid] == GOAT and board[dest] == EMPTY:
                moves.append((src, dest))
    return moves


def goat_moves(state: State):
    board = state.board
    if state.goats_in_hand > 0:
        return [(-1, d) for d in range(NUM_POINTS) if board[d] == EMPTY]
    moves = []
    for src in range(NUM_POINTS):
        if board[src] != GOAT:
            continue
        for dest in ADJACENCY[src]:
            if board[dest] == EMPTY:
                moves.append((src, dest))
    return moves


def legal_moves(state: State):
    return goat_moves(state) if state.turn == GOAT else tiger_moves(state)


def blocked_tigers(state: State) -> int:
    """Number of tigers with no legal move."""
    board = state.board
    count = 0
    for src in state.tigers:
        mobile = any(board[d] == EMPTY for d in ADJACENCY[src]) or any(
            board[m] == GOAT and board[d] == EMPTY for m, d in JUMPS[src]
        )
        if not mobile:
            count += 1
    return count


def apply_move(state: State, move) -> State:
    """Return the successor state.  The input state is never mutated."""
    src, dest = move
    nxt = state.clone()
    board = nxt.board
    progress = False

    if state.turn == GOAT:
        if src == -1:
            nxt.goats_in_hand -= 1
            progress = True
        else:
            board[src] = EMPTY
        board[dest] = GOAT
    else:
        board[src] = EMPTY
        for mid, jump_dest in JUMPS[src]:
            if jump_dest == dest:
                board[mid] = EMPTY
                nxt.goats_captured += 1
                progress = True
                break
        board[dest] = TIGER
        nxt.tigers = tuple(dest if t == src else t for t in state.tigers)

    nxt.turn = -state.turn
    nxt.plies_without_progress = 0 if progress else state.plies_without_progress + 1
    k = nxt.key()
    nxt.history[k] = nxt.history.get(k, 0) + 1
    return nxt


def outcome(state: State):
    """Return GOAT_WIN / TIGER_WIN / DRAW, or ONGOING (None)."""
    if state.goats_captured >= CAPTURES_TO_WIN:
        return TIGER_WIN
    if blocked_tigers(state) == len(state.tigers):
        return GOAT_WIN
    if state.history.get(state.key(), 0) >= REPETITION_LIMIT:
        return DRAW
    if state.plies_without_progress >= NO_PROGRESS_LIMIT:
        return DRAW
    if state.turn == GOAT and not state.in_placement_phase and not goat_moves(state):
        # Every goat is walled in and none can slide: treated as a tiger win.
        return TIGER_WIN
    return ONGOING


def outcome_to_unit(result: int, player: int) -> float:
    """Map an outcome to [0, 1] from `player`'s point of view."""
    if result == DRAW:
        return 0.5
    return 1.0 if result == player else 0.0
