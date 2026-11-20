import random

# Cell values
EMPTY = 0
HUMAN = 1
AI = 2

# Minimum board size required by the assignment (width >= 7, length >= 6)
MIN_COLS = 7
MIN_ROWS = 6

# The 4 line directions as (d_row, d_col):
# horizontal, vertical, diagonal "\" and anti-diagonal "/".
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Stochastic drop probabilities (Assignment Section 2.2)
P_CHOSEN = 0.6           # disc lands in the chosen column
P_SIDE_BOTH = 0.2        # each neighbour, when both neighbours are available
P_SIDE_ONE = 0.4         # the single neighbour, when only one is available


def opponent(piece):
    """Return the other player's piece value."""
    return HUMAN if piece == AI else AI


class Board:
    """A Connect 4 board of any size >= 6 rows x 7 columns."""

    def __init__(self, rows=6, cols=7):
        if rows < MIN_ROWS or cols < MIN_COLS:
            raise ValueError(
                f"Board must be at least {MIN_ROWS} rows x {MIN_COLS} columns "
                f"(got {rows} x {cols})")
        self.rows = rows
        self.cols = cols
        self.grid = [[EMPTY] * cols for _ in range(rows)]
        self.heights = [0] * cols          # discs currently in each column
        self.move_count = 0                # total discs on the board

    # ------------------------------------------------------------------
    # Gravity and valid moves
    # ------------------------------------------------------------------
    def is_valid_move(self, col):
        """A column is playable if it exists and is not full."""
        return 0 <= col < self.cols and self.heights[col] < self.rows

    def valid_moves(self):
        """List of all playable columns, left to right."""
        return [c for c in range(self.cols) if self.heights[c] < self.rows]

    def landing_row(self, col):
        """Row index where a disc dropped in `col` would land (no change)."""
        return self.rows - 1 - self.heights[col]

    def drop(self, col, piece):
        """
        Drop `piece` into `col`. The disc falls to the lowest empty cell.
        Returns the row it landed in. Raises ValueError for an illegal move.
        """
        if not self.is_valid_move(col):
            raise ValueError(f"Column {col} is full or out of range")
        row = self.landing_row(col)
        self.grid[row][col] = piece
        self.heights[col] += 1
        self.move_count += 1
        return row

    def undo(self, col):
        """Remove the top disc of `col` (lets the AI search without copying)."""
        if self.heights[col] == 0:
            raise ValueError(f"Column {col} is empty, nothing to undo")
        self.heights[col] -= 1
        row = self.rows - 1 - self.heights[col]
        self.grid[row][col] = EMPTY
        self.move_count -= 1

    # ------------------------------------------------------------------
    # End-of-game detection
    # ------------------------------------------------------------------
    def is_full(self):
        """The game ends only when every cell is filled."""
        return self.move_count == self.rows * self.cols

    # ------------------------------------------------------------------
    # Scoring: counting connected-fours
    # ------------------------------------------------------------------
    def count_fours(self, piece):
        """
        Count every window of 4 consecutive cells (any direction) that is
        completely filled with `piece`. Overlapping windows count separately.
        """
        g, R, C = self.grid, self.rows, self.cols
        total = 0
        for r in range(R):
            for c in range(C):
                if g[r][c] != piece:
                    continue
                # Only count a window from its starting cell so each window
                # is counted exactly once.
                for dr, dc in DIRECTIONS:
                    end_r, end_c = r + 3 * dr, c + 3 * dc
                    if not (0 <= end_r < R and 0 <= end_c < C):
                        continue
                    if (g[r + dr][c + dc] == piece and
                            g[r + 2 * dr][c + 2 * dc] == piece and
                            g[end_r][end_c] == piece):
                        total += 1
        return total

    def fours_through(self, row, col):
        """
        Number of connected-fours that include cell (row, col) for the piece
        in that cell. Called right after a drop, this tells how many points
        that single move scored - a fast incremental alternative to
        recounting the whole board.
        """
        piece = self.grid[row][col]
        if piece == EMPTY:
            return 0
        g, R, C = self.grid, self.rows, self.cols
        total = 0
        for dr, dc in DIRECTIONS:
            # Length of the run of `piece` through (row, col) in this direction
            back = 0
            r, c = row - dr, col - dc
            while 0 <= r < R and 0 <= c < C and g[r][c] == piece:
                back += 1
                r, c = r - dr, c - dc
            fwd = 0
            r, c = row + dr, col + dc
            while 0 <= r < R and 0 <= c < C and g[r][c] == piece:
                fwd += 1
                r, c = r + dr, c + dc
            # A window of 4 containing this cell starts at offset s (relative
            # to the cell) with -3 <= s <= 0, and must stay inside the run:
            # -back <= s and s + 3 <= fwd.
            lo = max(-back, -3)
            hi = min(0, fwd - 3)
            if hi >= lo:
                total += hi - lo + 1
        return total

    def scores(self):
        """Return (human_score, ai_score)."""
        return self.count_fours(HUMAN), self.count_fours(AI)

    def winner(self):
        """
        Only meaningful when the board is full.
        Returns HUMAN, AI, or None for a draw.
        """
        h, a = self.scores()
        if h > a:
            return HUMAN
        if a > h:
            return AI
        return None

    # ------------------------------------------------------------------
    # Randomness (used by Expectiminimax and by the real game)
    # ------------------------------------------------------------------
    def chance_outcomes(self, col):
        """
        Possible columns a disc aimed at `col` may actually land in, with
        their probabilities (Assignment Section 2.2):
          * both neighbours available : chosen 0.6, left 0.2, right 0.2
          * only one neighbour        : chosen 0.6, that neighbour 0.4
          * no neighbour available    : chosen 1.0
        A neighbour is "available" if it is inside the board and not full.
        Returns a list of (column, probability).
        """
        if not self.is_valid_move(col):
            raise ValueError(f"Column {col} is full or out of range")
        sides = [c for c in (col - 1, col + 1) if self.is_valid_move(c)]
        if len(sides) == 2:
            return [(col, P_CHOSEN), (sides[0], P_SIDE_BOTH),
                    (sides[1], P_SIDE_BOTH)]
        if len(sides) == 1:
            return [(col, P_CHOSEN), (sides[0], P_SIDE_ONE)]
        return [(col, 1.0)]

    def resolve_stochastic(self, col, rng=random):
        """
        Sample the column the disc really lands in, according to
        chance_outcomes(col). Does NOT modify the board.
        """
        outcomes = self.chance_outcomes(col)
        cols = [c for c, _ in outcomes]
        probs = [p for _, p in outcomes]
        return rng.choices(cols, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def copy(self):
        """Independent deep copy of the board."""
        b = Board.__new__(Board)
        b.rows, b.cols = self.rows, self.cols
        b.grid = [row[:] for row in self.grid]
        b.heights = self.heights[:]
        b.move_count = self.move_count
        return b

    def __str__(self):
        """Text view for the console:  . empty,  X human,  O AI."""
        symbols = {EMPTY: ".", HUMAN: "X", AI: "O"}
        lines = [" ".join(symbols[v] for v in row) for row in self.grid]
        lines.append(" ".join(str(c % 10) for c in range(self.cols)))
        return "\n".join(lines)
