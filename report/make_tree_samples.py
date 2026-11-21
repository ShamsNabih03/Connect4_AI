"""
report/make_tree_samples.py - console tree printouts for the report (Person B).

Assignment 3.1 asks for "sample runs and their corresponding minimax trees".
This script plays a few fixed opening moves, asks the agent for its move with
each algorithm, and saves the printed tree (with heuristic values at the
leaves) plus a per-feature breakdown of the heuristic for that board.

Run from the project folder:
    python report/make_tree_samples.py
    python report/make_tree_samples.py --k 2 --depth 2

Writes to report/logs/:
    tree_minimax_k3.txt, tree_alphabeta_k3.txt, tree_expectiminimax_k3.txt
    heuristic_breakdown.txt
"""

import argparse
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # project folder

import ai_interface                                 # noqa: E402
from board import Board, HUMAN, AI                  # noqa: E402
from heuristic import explain, V2_MINIMAX, V2_EXPECTI   # noqa: E402

OPENING = [3, 3, 2, 4, 4, 2, 1]     # human moves first; a mid-opening position


def make_board():
    b = Board(6, 7)
    b.first_player = HUMAN
    piece = HUMAN
    for col in OPENING:
        b.drop(col, piece)
        piece = AI if piece == HUMAN else HUMAN
    return b                         # 7 discs -> computer to move


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--depth", type=int, default=2,
                   help="tree levels printed in full (best line always full)")
    a = p.parse_args()

    out_dir = os.path.join(HERE, "logs")
    os.makedirs(out_dir, exist_ok=True)
    board = make_board()
    header = ("Position (X = human, O = computer, computer to move):\n"
              f"{board}\n\n")

    ai_interface.QUIET = False
    ai_interface.PRINT_DEPTH = a.depth
    for algo in ("minimax", "alphabeta", "expectiminimax"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            col, info = ai_interface.get_move(board.copy(), algo, a.k)
        path = os.path.join(out_dir, f"tree_{algo}_k{a.k}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + buf.getvalue())
        print(f"{algo:>15}: column {col}, nodes {info['nodes']:,} -> {path}")

    path = os.path.join(out_dir, "heuristic_breakdown.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for name, w in (("Minimax / Alpha-Beta weights", V2_MINIMAX),
                        ("Expectiminimax weights", V2_EXPECTI)):
            f.write(f"Heuristic v2 ({name}), computer's point of view:\n")
            for feature, value in explain(board, w).items():
                f.write(f"  {feature:<20} {value:+.3f}\n")
            f.write("\n")
    print(f"{'breakdown':>15}: {path}")


if __name__ == "__main__":
    main()
