"""Tests for the per-turn tree output of ai_interface.get_move:
whole tree on the console when small, summary when big, and a text file per turn."""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

import ai_interface as ai
from board import Board


class TreeOutputTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (ai.TREE_DIR, ai.FULL_PRINT_LIMIT, ai.SAVE_LIMIT, ai.SAVE_TREES, ai.QUIET, ai.PRINT_DEPTH)
        ai.TREE_DIR, ai.SAVE_TREES, ai.QUIET, ai.PRINT_DEPTH = self.dir, True, False, 2

    def tearDown(self):
        (ai.TREE_DIR, ai.FULL_PRINT_LIMIT, ai.SAVE_LIMIT, ai.SAVE_TREES, ai.QUIET,
         ai.PRINT_DEPTH) = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_move(self, algorithm, k, board=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ai.get_move(board or Board(), algorithm, k)
        return buf.getvalue()

    def read_turn(self, n=0):
        with open(os.path.join(self.dir, f"turn_{n:02d}.txt"), encoding="utf-8") as f:
            return f.read()

    def test_small_tree_printed_in_full(self):
        ai.FULL_PRINT_LIMIT = 3000
        out = self.run_move("minimax", 3)              # 400 nodes
        self.assertNotIn("depth limit", out)
        self.assertEqual(out.count("[LEAF]"), 343)     # every leaf is on the console
        self.assertIn("Full tree saved to", out)

    def test_big_tree_summarised_on_console_but_complete_in_file(self):
        ai.FULL_PRINT_LIMIT = 0
        out = self.run_move("minimax", 3)
        self.assertIn("depth limit", out)
        text = self.read_turn()
        self.assertNotIn("depth limit", text)
        self.assertEqual(text.count("[LEAF]"), 343)

    def test_file_per_turn_named_by_disc_count(self):
        b = Board()
        self.run_move("alphabeta", 2, b)
        b.drop(3, 1)
        b.drop(2, 2)
        b.drop(4, 1)                                   # 3 discs on the board
        self.run_move("alphabeta", 2, b)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "turn_00.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "turn_03.txt")))
        self.assertIn("Alpha-Beta, K=2", self.read_turn(3))

    def test_new_game_clears_old_files(self):
        stale = os.path.join(self.dir, "turn_99.txt")
        with open(stale, "w") as f:
            f.write("old")
        self.run_move("minimax", 1)                    # empty board = new game
        self.assertFalse(os.path.exists(stale))

    def test_huge_tree_file_is_marked_summarised(self):
        ai.SAVE_LIMIT = 10
        out = self.run_move("minimax", 3)
        text = self.read_turn()
        self.assertIn("only the top", text)
        self.assertIn("depth limit", text)
        self.assertIn("Summarised tree", out)

    def test_quiet_and_disabled_write_nothing(self):
        ai.QUIET = True
        self.assertEqual(self.run_move("minimax", 2), "")
        ai.QUIET, ai.SAVE_TREES = False, False
        self.run_move("minimax", 2)
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
