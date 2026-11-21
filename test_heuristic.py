"""Unit tests for heuristic.py and tree_printer.py.
Run with:  python -m unittest test_heuristic -v"""

import io
import random
import unittest

from board import Board, EMPTY, HUMAN, AI
from heuristic import (geometry, evaluate_v1, evaluate_v2, explain, heuristic,
                       V2_MINIMAX, V2_EXPECTI, ABLATIONS, evaluate_galli,
                       evaluate_big_constants, side_to_move)
from tree import TreeNode, MAX, MIN, CHANCE, LEAF
from tree_printer import print_tree, tree_lines

ALL_V2 = [V2_MINIMAX, V2_EXPECTI] + list(ABLATIONS.values())


def random_board(rng, rows=6, cols=7, moves=None, first=HUMAN):
    b = Board(rows, cols)
    b.first_player = first
    n = rng.randint(0, rows * cols) if moves is None else moves
    piece = first
    for _ in range(n):
        if b.is_full():
            break
        b.drop(rng.choice(b.valid_moves()), piece)
        piece = AI if piece == HUMAN else HUMAN
    return b


def swapped(b):
    """Same position with the colours (and the first player) exchanged."""
    s = b.copy()
    for r in range(b.rows):
        for c in range(b.cols):
            v = b.grid[r][c]
            s.grid[r][c] = {HUMAN: AI, AI: HUMAN}.get(v, EMPTY)
    s.first_player = AI if b.first_player == HUMAN else HUMAN
    return s


def mirrored(b):
    m = b.copy()
    for r in range(b.rows):
        m.grid[r] = b.grid[r][::-1]
    m.heights = b.heights[::-1]
    return m


def board_from(rows_text, first=HUMAN):
    """Build a 6x7 board from text, top row first: . empty, X human, O AI."""
    b = Board(len(rows_text), len(rows_text[0]))
    b.first_player = first
    for r, line in enumerate(rows_text):
        for c, ch in enumerate(line):
            v = {".": EMPTY, "X": HUMAN, "O": AI}[ch]
            b.grid[r][c] = v
            if v != EMPTY:
                b.move_count += 1
    for c in range(b.cols):
        b.heights[c] = sum(1 for r in range(b.rows) if b.grid[r][c] != EMPTY)
    return b


class TestGeometry(unittest.TestCase):
    def test_window_counts(self):
        # R(C-3) + (R-3)C + 2(R-3)(C-3)
        for rows, cols in [(6, 7), (7, 8), (6, 9), (10, 12)]:
            windows, through = geometry(rows, cols)
            expected = rows * (cols - 3) + (rows - 3) * cols + 2 * (rows - 3) * (cols - 3)
            self.assertEqual(len(windows), expected)
            self.assertEqual(sum(map(sum, through)), 4 * expected)
        self.assertEqual(len(geometry(6, 7)[0]), 69)

    def test_classic_centre_table(self):
        _, through = geometry(6, 7)
        self.assertEqual(list(through[0]), [3, 4, 5, 7, 5, 4, 3])
        self.assertEqual(list(through[2]), [5, 8, 11, 13, 11, 8, 5])


class TestExactAtFullBoard(unittest.TestCase):
    def test_full_board_equals_score(self):
        rng = random.Random(1)
        for rows, cols in [(6, 7), (7, 8), (6, 9)]:
            for _ in range(60):
                b = random_board(rng, rows, cols, moves=rows * cols)
                h, a = b.scores()
                for w in ALL_V2:
                    self.assertEqual(evaluate_v2(b, w), a - h)
                self.assertEqual(evaluate_v1(b), a - h)

    def test_empty_board_is_zero(self):
        for rows, cols in [(6, 7), (7, 9)]:
            self.assertEqual(evaluate_v2(Board(rows, cols)), 0)
            self.assertEqual(evaluate_v1(Board(rows, cols)), 0)


class TestSymmetry(unittest.TestCase):
    def test_colour_swap_negates(self):
        rng = random.Random(2)
        for rows, cols in [(6, 7), (7, 8), (6, 9)]:
            for _ in range(150):
                b = random_board(rng, rows, cols,
                                 first=rng.choice([HUMAN, AI]))
                s = swapped(b)
                for w in ALL_V2:
                    self.assertAlmostEqual(evaluate_v2(b, w), -evaluate_v2(s, w), 9)
                self.assertAlmostEqual(evaluate_v1(b), -evaluate_v1(s), 9)
                self.assertAlmostEqual(evaluate_galli(b), -evaluate_galli(s), 9)
                self.assertAlmostEqual(evaluate_big_constants(b),
                                       -evaluate_big_constants(s), 6)

    def test_mirror_invariant(self):
        rng = random.Random(3)
        for _ in range(150):
            b = random_board(rng)
            m = mirrored(b)
            for w in ALL_V2:
                self.assertAlmostEqual(evaluate_v2(b, w), evaluate_v2(m, w), 9)

    def test_does_not_modify_board(self):
        rng = random.Random(4)
        b = random_board(rng, moves=20)
        before = ([row[:] for row in b.grid], b.heights[:], b.move_count)
        evaluate_v2(b)
        evaluate_v1(b)
        explain(b)
        self.assertEqual(before, ([row[:] for row in b.grid], b.heights[:], b.move_count))


class TestFeatures(unittest.TestCase):
    def test_banked_four_is_one_point(self):
        b = board_from(["......."] * 5 + ["OOOO..."])
        self.assertAlmostEqual(explain(b)["banked fours"], 1.0)

    def test_overlapping_fours(self):
        b = board_from(["......."] * 5 + ["OOOOO.."])
        self.assertAlmostEqual(explain(b)["banked fours"], 2.0)

    def test_playable_threat_owner_to_move(self):
        # AI has O O O _ on the bottom row; the gap (column 4) is playable.
        # 6 discs and the human moved first -> human to move -> gets blocked
        b = board_from(["......."] * 4 + ["......X", "XOOO..X"])
        self.assertEqual(side_to_move(b), HUMAN)
        self.assertAlmostEqual(explain(b)["playable threats"], V2_MINIMAX["BLOCKED"])
        # 5 discs -> AI to move -> it will cash the point
        b2 = board_from(["......."] * 5 + ["XOOO..X"])
        self.assertEqual(side_to_move(b2), AI)
        self.assertAlmostEqual(explain(b2)["playable threats"], V2_MINIMAX["CASH"])

    def test_fork_is_worth_about_one_point(self):
        # _ O O O _ : two playable threats, the human can block only one
        b = board_from(["......."] * 4 + ["X......", "X.OOO.X"])
        self.assertEqual(side_to_move(b), HUMAN)
        self.assertAlmostEqual(explain(b)["playable threats"],
                               V2_MINIMAX["BLOCKED"] + V2_MINIMAX["FORK"])

    def test_multiplicity(self):
        # O O O _ O O O : filling the gap makes 7 in a row = FOUR fours
        # (windows 0-3, 1-4, 2-5, 3-6 all hold 3 O's + the gap), so m = 4
        b = board_from(["......."] * 4 + ["XXX.XXX", "OOO.OOO"])
        self.assertEqual(side_to_move(b), HUMAN)
        self.assertAlmostEqual(explain(b)["playable threats"],
                               4 * V2_MINIMAX["BLOCKED"])
        no_mult = {}
        evaluate_v2(b, ABLATIONS["v2-no-multiplicity"], no_mult)
        self.assertAlmostEqual(no_mult["playable threats"], V2_MINIMAX["BLOCKED"])

    def test_parity_good_vs_bad(self):
        # AI floating threat O O O _ on the 3rd row from the bottom (odd row);
        # the gap (column 3) has two empty cells below it.
        rows = [".......", ".......", ".......",
                "OOO....", "OXO....", "XOX...."]
        good = explain(board_from(rows, first=AI))      # AI started: odd = good
        bad = explain(board_from(rows, first=HUMAN))    # human started: bad
        self.assertAlmostEqual(good["floating threats"], V2_MINIMAX["GOOD"])
        self.assertAlmostEqual(bad["floating threats"], V2_MINIMAX["BAD"])

    def test_own_stack(self):
        # two AI threats on top of each other in column 3 -> STACK
        # (only with the rejected stack rule switched on)
        rows = [".......", ".......",
                "OOO....", "OOO....", "XXX....", "XXX...."]
        b = board_from(rows)
        parts = {}
        evaluate_v2(b, ABLATIONS["v2+stacks (rejected)"], parts)
        f = parts["floating threats"]
        # AI stack (2,3)+(3,3): 2 x STACK; the human also has a stack,
        # (4,3) floating above its playable (5,3): -1 x STACK
        self.assertAlmostEqual(f, V2_MINIMAX["STACK"])
        # human to move with a playable threat -> cashes it
        self.assertAlmostEqual(explain(b)["playable threats"], -V2_MINIMAX["CASH"])

    def test_explain_sums_to_total(self):
        rng = random.Random(5)
        for _ in range(100):
            b = random_board(rng)
            parts = explain(b)
            total = parts.pop("total")
            self.assertAlmostEqual(sum(parts.values()), total, 9)
            self.assertAlmostEqual(total, evaluate_v2(b), 9)

    def test_heuristic_picks_weights(self):
        rng = random.Random(6)
        b = random_board(rng, moves=18)
        self.assertEqual(heuristic(b, "minimax"), evaluate_v2(b, V2_MINIMAX))
        self.assertEqual(heuristic(b, "alphabeta"), evaluate_v2(b, V2_MINIMAX))
        self.assertEqual(heuristic(b, "expectiminimax"), evaluate_v2(b, V2_EXPECTI))

    def test_missing_first_player_defaults(self):
        b = Board()
        del b.first_player
        b.drop(3, HUMAN)
        evaluate_v2(b)                       # must not crash

    def test_copy_keeps_first_player(self):
        b = Board()
        b.first_player = AI
        self.assertEqual(b.copy().first_player, AI)


class TestTreePrinter(unittest.TestCase):
    def _tiny_tree(self):
        root = TreeNode(MAX, value=0.5)
        a = TreeNode(MIN, move=3, player=AI, value=0.5)
        a.children = [TreeNode(LEAF, move=2, player=HUMAN, value=0.5),
                      TreeNode(LEAF, move=3, player=HUMAN, value=0.9)]
        b = TreeNode(MIN, move=4, player=AI, value=0.1)
        b.children = [TreeNode(LEAF, move=0, player=HUMAN, value=0.1),
                      TreeNode(MIN, move=1, player=HUMAN, pruned=True)]
        ch = TreeNode(CHANCE, aimed=6, player=AI, value=0.2)
        ch.children = [TreeNode(LEAF, move=6, player=AI, prob=0.6, value=0.3),
                       TreeNode(LEAF, move=5, player=AI, prob=0.4, value=0.05)]
        root.children = [a, b, ch]
        return root

    def test_lines(self):
        lines = tree_lines(self._tiny_tree(), max_depth=None)
        text = "\n".join(lines)
        self.assertIn("[MAX] value=0.50", lines[0])
        self.assertIn("col 3 -> [MIN] value=0.50  <== BEST MOVE", text)
        self.assertIn("[LEAF] h=0.90", text)
        self.assertIn("(pruned)", text)
        self.assertIn("aim 6 -> [CHANCE] E=0.20", text)
        self.assertIn("lands 5 (p=0.4)", text)
        self.assertEqual(len(lines), 10)            # one line per node
        text.encode("ascii")                         # prints in any console

    def test_depth_limit_summarises(self):
        lines = tree_lines(self._tiny_tree(), max_depth=1)
        text = "\n".join(lines)
        self.assertIn("more node", text)             # the non-best branch folded
        self.assertIn("h=0.50", text)                # best line still to leaf

    def test_print_tree_header(self):
        out = io.StringIO()
        print_tree(self._tiny_tree(), title="Alpha-Beta, K=2",
                   stats={"nodes": 7, "column": 3, "time": 0.01}, file=out)
        s = out.getvalue()
        self.assertIn("Alpha-Beta, K=2", s)
        self.assertIn("chosen column: 3", s)

    def test_real_agent_tree_prints(self):
        import ai_interface
        rng = random.Random(7)
        b = random_board(rng, moves=10)
        old = ai_interface.QUIET
        try:
            ai_interface.QUIET = True               # nothing printed when quiet
            for algo in ("minimax", "alphabeta", "expectiminimax"):
                col, info = ai_interface.get_move(b.copy(), algo, 2)
                self.assertIn(col, b.valid_moves())
                out = io.StringIO()
                print_tree(info["tree"], max_depth=2, file=out)
                self.assertIn("BEST MOVE", out.getvalue())
        finally:
            ai_interface.QUIET = old


if __name__ == "__main__":
    unittest.main()
