"""Unit tests for board.py. Run with:  python -m unittest test_board -v"""

import random
import unittest

from board import Board, EMPTY, HUMAN, AI


class TestGravity(unittest.TestCase):
    def test_disc_falls_to_bottom_and_stacks(self):
        b = Board()
        self.assertEqual(b.drop(3, HUMAN), 5)     # bottom row
        self.assertEqual(b.drop(3, AI), 4)        # stacks on top
        self.assertEqual(b.grid[5][3], HUMAN)
        self.assertEqual(b.grid[4][3], AI)

    def test_full_column_is_invalid(self):
        b = Board()
        for _ in range(6):
            b.drop(0, HUMAN)
        self.assertFalse(b.is_valid_move(0))
        self.assertNotIn(0, b.valid_moves())
        with self.assertRaises(ValueError):
            b.drop(0, AI)

    def test_out_of_range(self):
        b = Board()
        self.assertFalse(b.is_valid_move(-1))
        self.assertFalse(b.is_valid_move(7))

    def test_undo(self):
        b = Board()
        b.drop(2, HUMAN)
        b.drop(2, AI)
        b.undo(2)
        self.assertEqual(b.grid[4][2], EMPTY)
        self.assertEqual(b.heights[2], 1)
        self.assertEqual(b.move_count, 1)

    def test_min_size_enforced(self):
        with self.assertRaises(ValueError):
            Board(5, 7)
        with self.assertRaises(ValueError):
            Board(6, 6)
        Board(8, 9)   # bigger boards are fine


class TestEndOfGame(unittest.TestCase):
    def test_full_only_when_every_cell_filled(self):
        b = Board()
        for c in range(7):
            for _ in range(6):
                self.assertFalse(b.is_full())
                b.drop(c, HUMAN)
        self.assertTrue(b.is_full())
        self.assertEqual(b.valid_moves(), [])


class TestScoring(unittest.TestCase):
    def test_horizontal(self):
        b = Board()
        for c in range(4):
            b.drop(c, HUMAN)
        self.assertEqual(b.scores(), (1, 0))

    def test_vertical(self):
        b = Board()
        for _ in range(4):
            b.drop(6, AI)
        self.assertEqual(b.scores(), (0, 1))

    def test_diagonal_both_ways(self):
        b = Board()
        # "/" diagonal for HUMAN from (col0,bottom) up to (col3)
        for c in range(4):
            for _ in range(c):
                b.drop(c, AI)
            b.drop(c, HUMAN)
        self.assertEqual(b.count_fours(HUMAN), 1)

        b = Board()
        # "\" diagonal: higher on the left
        for c in range(4):
            for _ in range(3 - c):
                b.drop(c, AI)
            b.drop(c, HUMAN)
        self.assertEqual(b.count_fours(HUMAN), 1)

    def test_overlapping_fours(self):
        b = Board()
        for c in range(6):        # 6 in a row -> 3 fours
            b.drop(c, HUMAN)
        self.assertEqual(b.count_fours(HUMAN), 3)

    def test_three_is_not_a_four(self):
        b = Board()
        for c in range(3):
            b.drop(c, HUMAN)
        self.assertEqual(b.count_fours(HUMAN), 0)

    def test_winner(self):
        b = Board()
        for c in range(4):
            b.drop(c, AI)
        self.assertEqual(b.winner(), AI)
        self.assertIsNone(Board().winner())

    def test_incremental_matches_full_count(self):
        """fours_through() after each move == change in count_fours()."""
        rng = random.Random(42)
        for rows, cols in [(6, 7), (7, 9)]:
            for _ in range(200):
                b = Board(rows, cols)
                piece = HUMAN
                totals = {HUMAN: 0, AI: 0}
                while not b.is_full():
                    col = rng.choice(b.valid_moves())
                    row = b.drop(col, piece)
                    totals[piece] += b.fours_through(row, col)
                    piece = AI if piece == HUMAN else HUMAN
                self.assertEqual(totals[HUMAN], b.count_fours(HUMAN))
                self.assertEqual(totals[AI], b.count_fours(AI))


class TestRandomness(unittest.TestCase):
    def test_middle_column(self):
        self.assertEqual(Board().chance_outcomes(3),
                         [(3, 0.6), (2, 0.2), (4, 0.2)])

    def test_edge_column(self):
        self.assertEqual(Board().chance_outcomes(0), [(0, 0.6), (1, 0.4)])
        self.assertEqual(Board().chance_outcomes(6), [(6, 0.6), (5, 0.4)])

    def test_full_neighbour(self):
        b = Board()
        for _ in range(6):
            b.drop(2, HUMAN)
        self.assertEqual(b.chance_outcomes(3), [(3, 0.6), (4, 0.4)])
        for _ in range(6):
            b.drop(4, HUMAN)
        self.assertEqual(b.chance_outcomes(3), [(3, 1.0)])

    def test_sampling_frequencies(self):
        rng = random.Random(0)
        b = Board()
        n = 20000
        counts = {2: 0, 3: 0, 4: 0}
        for _ in range(n):
            counts[b.resolve_stochastic(3, rng)] += 1
        self.assertAlmostEqual(counts[3] / n, 0.6, delta=0.02)
        self.assertAlmostEqual(counts[2] / n, 0.2, delta=0.02)
        self.assertAlmostEqual(counts[4] / n, 0.2, delta=0.02)


class TestCopy(unittest.TestCase):
    def test_copy_is_independent(self):
        b = Board()
        b.drop(0, HUMAN)
        c = b.copy()
        c.drop(0, AI)
        self.assertEqual(b.heights[0], 1)
        self.assertEqual(c.heights[0], 2)


if __name__ == "__main__":
    unittest.main()
