"""
heuristic.py - heuristic evaluation for the Connect 4 agent.

the search is truncated after K levels and every
leaf is scored by a heuristic that "returns bigger number when computer is
nearer to win and smaller number when human is nearer to win".

Key idea: score the WINDOWS, not the win
----------------------------------------
In this variant the game always runs until the board is full and the winner
has MORE connected-fours (overlapping fours count separately). A connected-
four is exactly a 4-cell window filled by one player, so the final score is

    final = sum over all 4-cell windows of  [all 4 are AI] - [all 4 are HUMAN]

The heuristic therefore estimates the EXPECTED FINAL SCORE DIFFERENCE in
points (one completed four = 1.0):

    h = (AI fours - human fours)                       <- points already banked
      + credit for every window that can still be completed (a fraction of a
        point that grows with the number of discs in it, and for 3-disc
        windows depends on playability, side to move and odd/even parity)

Consequences:
  * at a full board h == ai_score - human_score exactly;
  * blocking needs no extra term: a disc in an opponent window kills that
    window's credit;
  * values are real "expected points", which Expectiminimax needs because it
    AVERAGES leaf values (a 10^6 "win" constant would dominate every average).

Public functions
----------------
  evaluate_v1(board)                  simple baseline
  evaluate_v2(board, weights)         full heuristic (the one we submit)
  heuristic(board, algorithm)         v2 with the right weight set
  explain(board, weights)             v2 broken down per feature (report/GUI)
  evaluate_score_diff / evaluate_galli / evaluate_big_constants
                                      reference heuristics for the experiments

Board conventions (board.py): grid[r][c], row 0 is the TOP, EMPTY=0,
HUMAN=1, AI=2. evaluate_v2 needs to know who moved first (for parity):
board.first_player. If the board has no such attribute HUMAN is assumed.
"""

from functools import lru_cache

from board import EMPTY, HUMAN, AI

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------------------------------------------------------------------
# Board geometry - computed once per board size and cached
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def geometry(rows, cols):
    """
    Returns (windows, through):
      windows : tuple of all 4-cell windows, each a tuple of 4 (row, col)
                (69 windows on a 6x7 board; any size works)
      through : through[r][c] = number of windows passing through (r, c);
                generalises the classic 3-4-5-7-5-4-3 centre table.
    """
    windows = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in DIRECTIONS:
                cells = tuple((r + i * dr, c + i * dc) for i in range(4))
                if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in cells):
                    windows.append(cells)
    through = [[0] * cols for _ in range(rows)]
    for w in windows:
        for r, c in w:
            through[r][c] += 1
    return tuple(windows), tuple(tuple(row) for row in through)


def _landing_rows(board):
    """landing[c] = row where a disc dropped in column c lands, -1 if full."""
    R = board.rows
    return [R - 1 - h for h in board.heights]       # -1 when the column is full


def first_player(board):
    """Who moved first (needed for odd/even parity). Defaults to HUMAN."""
    return getattr(board, "first_player", HUMAN)


def side_to_move(board):
    """Moves always alternate (even with random slips), so parity of the
    number of discs tells whose turn it is."""
    first = first_player(board)
    return first if board.move_count % 2 == 0 else (AI + HUMAN - first)


def _exact(board):
    h, a = board.scores()
    return float(a - h)


# ---------------------------------------------------------------------------
# v1 - simple baseline
# ---------------------------------------------------------------------------
V1 = dict(W1=0.02, W2=0.10, W3_PLAY=0.50, W3_FLOAT=0.30, W_CELL=0.002)


def evaluate_v1(board, w=V1):
    """Banked fours + live-window credit + playable/floating threes + centre."""
    if board.is_full():
        return _exact(board)
    windows, through = geometry(board.rows, board.cols)
    g = board.grid
    land = _landing_rows(board)
    val = 0.0
    for win in windows:
        na = nh = 0
        empty = None
        for r, c in win:
            v = g[r][c]
            if v == AI:
                na += 1
            elif v == HUMAN:
                nh += 1
            else:
                empty = (r, c)
        if (na and nh) or not (na or nh):
            continue                          # dead (both colours) or untouched
        n, s = (na, 1.0) if na else (nh, -1.0)
        if n == 4:
            val += s
        elif n == 3:
            val += s * (w["W3_PLAY"] if land[empty[1]] == empty[0] else w["W3_FLOAT"])
        elif n == 2:
            val += s * w["W2"]
        else:
            val += s * w["W1"]
    for r in range(board.rows):
        row, trow = g[r], through[r]
        for c in range(board.cols):
            if row[c] == AI:
                val += w["W_CELL"] * trow[c]
            elif row[c] == HUMAN:
                val -= w["W_CELL"] * trow[c]
    return val


# ---------------------------------------------------------------------------
# v2 - full heuristic
# ---------------------------------------------------------------------------
# All values are in points (a completed four = 1.0).
#   W1, W2      live window holding 1 / 2 discs of one player
#   CASH        best playable threat when its OWNER is to move (cashes it next)
#   PLAY_EXTRA  further playable threats when the owner is to move
#   BLOCKED     best playable threat when the OPPONENT is to move (gets blocked)
#   FORK        further playable threats when the opponent is to move
#               (he can block only one per turn)
#   GOOD / BAD  floating threat on the owner's / the wrong parity row
#   STACK       floating threat with another own threat directly above/below
#   MIXED       floating threat next to an opponent threat in the same column
#   W_CELL      centre tie-breaker: windows through each own disc
# Flags: MULT counts a cell that completes m windows m times; STACKS enables
# the STACK/MIXED rules.
#
# STACKS is OFF in the final heuristic. The first tournament (results/
# round1_with_stacks) showed the stack rule makes the agent WEAKER: v2 without
# it beat v2 with it 61% (K=4) and 69% (K=5) of the points. Stacked threats
# are then valued by the parity rule like any other floating threat. The
# stack rule is kept only as an ablation ("v2+stacks (rejected)").
V2_MINIMAX = dict(W1=0.02, W2=0.10, W_CELL=0.002,
                  CASH=0.90, PLAY_EXTRA=0.60, BLOCKED=0.20, FORK=0.70,
                  GOOD=0.60, BAD=0.25, STACK=0.60, MIXED=0.40,
                  MULT=True, STACKS=False)

# With random slips (0.6 / 0.2 / 0.2) an aimed threat is less certain and the
# "answer in the same column" parity plan breaks 40% of the time, so the
# side-to-move and parity gaps shrink.
V2_EXPECTI = dict(V2_MINIMAX, CASH=0.60, PLAY_EXTRA=0.50, BLOCKED=0.35,
                  FORK=0.60, GOOD=0.45, BAD=0.30, STACK=0.55)


def _ablate(w, **changes):
    out = dict(w)
    out.update(changes)
    return out


# Ablation variants: each removes ONE idea so the experiment can measure it.
ABLATIONS = {
    "v2-no-parity": _ablate(V2_MINIMAX, GOOD=0.425, BAD=0.425),
    "v2-no-side-to-move": _ablate(V2_MINIMAX, CASH=0.55, BLOCKED=0.55,
                                  PLAY_EXTRA=0.65, FORK=0.65),
    "v2-no-multiplicity": _ablate(V2_MINIMAX, MULT=False),
    "v2+stacks (rejected)": _ablate(V2_MINIMAX, STACKS=True),
}


def evaluate_v2(board, w=V2_MINIMAX, parts=None):
    """
    Full heuristic, value in points from the computer's point of view.
    If `parts` is a dict it is filled with the contribution of each feature
    (used by explain()); the returned total is the same either way.
    """
    if board.is_full():
        v = _exact(board)
        if parts is not None:
            parts["banked fours"] = v
        return v

    R, C, g = board.rows, board.cols, board.grid
    windows, through = geometry(R, C)
    land = _landing_rows(board)
    first = first_player(board)
    to_move = side_to_move(board)
    phase = min(1.0, 2.0 * (R * C - board.move_count) / (R * C))
    use_mult, use_stacks = w.get("MULT", True), w.get("STACKS", True)

    banked = small = 0.0
    threats = {AI: {}, HUMAN: {}}            # empty cell -> windows it completes
    for win in windows:
        na = nh = 0
        empty = None
        for r, c in win:
            v = g[r][c]
            if v == AI:
                na += 1
            elif v == HUMAN:
                nh += 1
            else:
                empty = (r, c)
        if (na and nh) or not (na or nh):
            continue
        if na:
            p, n, s = AI, na, 1.0
        else:
            p, n, s = HUMAN, nh, -1.0
        if n == 4:
            banked += s
        elif n == 3:
            d = threats[p]
            d[empty] = d.get(empty, 0) + 1
        elif n == 2:
            small += s * w["W2"] * phase
        else:
            small += s * w["W1"] * phase

    playable_v = floating_v = 0.0
    for p, s in ((AI, 1.0), (HUMAN, -1.0)):
        opp = HUMAN if p == AI else AI
        mine, theirs = threats[p], threats[opp]
        if not use_mult:
            mine = {cell: 1 for cell in mine}
        playable = sorted((m for (r, c), m in mine.items() if land[c] == r),
                          reverse=True)
        if playable:
            if p == to_move:
                playable_v += s * (w["CASH"] * playable[0]
                                   + w["PLAY_EXTRA"] * sum(playable[1:]))
            else:
                playable_v += s * (w["BLOCKED"] * playable[0]
                                   + w["FORK"] * sum(playable[1:]))
        for (r, c), m in mine.items():
            if land[c] == r:
                continue                      # already counted as playable
            if use_stacks and ((r - 1, c) in mine or (r + 1, c) in mine):
                t = w["STACK"]
            elif use_stacks and ((r - 1, c) in theirs or (r + 1, c) in theirs):
                t = w["MIXED"]
            else:
                from_bottom = R - r           # row 0 is the TOP of the grid
                good = (from_bottom % 2 == 1) == (p == first)
                t = w["GOOD"] if good else w["BAD"]
            floating_v += s * t * m

    centre = 0.0
    wc = w["W_CELL"] * phase
    for r in range(R):
        row, trow = g[r], through[r]
        for c in range(C):
            if row[c] == AI:
                centre += wc * trow[c]
            elif row[c] == HUMAN:
                centre -= wc * trow[c]

    if parts is not None:
        parts["banked fours"] = banked
        parts["1-2 disc windows"] = small
        parts["playable threats"] = playable_v
        parts["floating threats"] = floating_v
        parts["centre"] = centre
    return banked + small + playable_v + floating_v + centre


def heuristic(board, algorithm="minimax"):
    """The heuristic the agent uses: v2 with the weight set for `algorithm`."""
    return evaluate_v2(board, V2_EXPECTI if algorithm == "expectiminimax"
                       else V2_MINIMAX)


def explain(board, w=V2_MINIMAX):
    """Dict feature -> contribution (+ 'total'); handy for the report/debug."""
    parts = {}
    parts["total"] = evaluate_v2(board, w, parts)
    return parts


# ---------------------------------------------------------------------------
# Reference heuristics (opponents in heuristic_experiments.py)
# ---------------------------------------------------------------------------
def evaluate_score_diff(board):
    """Baseline: computer fours minus human fours (the score difference)."""
    return _exact(board)


def evaluate_galli(board):
    """
    The widely copied Keith Galli heuristic (first-to-four Connect 4):
    own 4 = 100, own 3 + empty = 5, own 2 + 2 empty = 2, opponent 3 + empty
    = -4, +3 per disc in the centre column. Made symmetric (AI - human).
    """
    windows, _ = geometry(board.rows, board.cols)
    g = board.grid

    def side(me, opp):
        score = 0
        for win in windows:
            vals = [g[r][c] for r, c in win]
            m, o, e = vals.count(me), vals.count(opp), vals.count(EMPTY)
            if m == 4:
                score += 100
            elif m == 3 and e == 1:
                score += 5
            elif m == 2 and e == 2:
                score += 2
            if o == 3 and e == 1:
                score -= 4
        centre = board.cols // 2
        score += 3 * sum(1 for r in range(board.rows) if g[r][centre] == me)
        return score

    return float(side(AI, HUMAN) - side(HUMAN, AI))


def evaluate_big_constants(board):
    """
    Typical student heuristic for this assignment: first-to-four style
    constants (four = 10^6, open three = 1000, open two = 10), no playability,
    no parity. Symmetric (AI - human).
    """
    windows, _ = geometry(board.rows, board.cols)
    g = board.grid
    val = 0.0
    for win in windows:
        na = nh = 0
        for r, c in win:
            v = g[r][c]
            if v == AI:
                na += 1
            elif v == HUMAN:
                nh += 1
        if na and nh:
            continue
        n, s = (na, 1.0) if na else (nh, -1.0)
        val += s * (0, 1, 10, 1000, 1_000_000)[n]
    return val
