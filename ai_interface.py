"""
ai_interface.py - the contract between the GUI (gui.py) and the AI agent.

The GUI calls exactly ONE function, on a background thread, every AI turn:

    column, info = get_move(board, algorithm, k)

Parameters
----------
board     : a COPY of the current board.Board. The AI may drop/undo on it
            freely; the real game board is never touched by the AI.
            The AI plays board.AI, the human plays board.HUMAN.
algorithm : "minimax" | "alphabeta" | "expectiminimax"
k         : depth limit chosen on the settings screen (truncate after K levels)

Returns
-------
column : int, the column the AI AIMS at. In expectiminimax mode the GUI then
         applies the random slip (0.6 / 0.2 / 0.2) itself, so the AI must NOT
         randomise the move.
info   : dict (may be empty). Keys the GUI understands:
           "nodes" : int   - number of nodes expanded this turn
           "value" : float - minimax value of the chosen move
           "tree"  : tree.TreeNode - root of the search tree; shown by the
                     GUI tree viewer (bonus). See tree.py for the format.

Printing the tree to the console is the AI module's job (Section 2.2).
The agent should skip that printing when QUIET is True (set by the
board-size / timing experiments so console output does not distort timings).

The GUI also calls, after every move:

    value = evaluate(board)

which must return the HEURISTIC value of `board` from the computer's point
of view (bigger = computer closer to winning). The GUI shows it for the
current board and for the board after a disc in each column.

The search itself is implemented below: _minimax (with optional alpha-beta
pruning) and _expecti (Expectiminimax). The leaf value is the heuristic from
heuristic.py; the tree is built as tree.TreeNode objects and printed by
tree_printer.py.
"""

import glob
import math
import os
import time

from board import AI, HUMAN, opponent
from heuristic import evaluate_v2, V2_MINIMAX, V2_EXPECTI
from tree_printer import print_tree
from tree import TreeNode, MAX, MIN, CHANCE, LEAF, principal_path, subtree_size

ALGORITHMS = ("minimax", "alphabeta", "expectiminimax")
QUIET = False          # True -> agent must not print trees (experiments)
PRINT_DEPTH = 3        # levels printed on the console for BIG trees (best line always full)

# Tree output (assignment 2.2: "show the minimax tree in each turn"):
#  * console: the WHOLE tree when it has at most FULL_PRINT_LIMIT nodes
#    (e.g. Alpha-Beta or Minimax at K = 4); a bigger tree is summarised below
#    PRINT_DEPTH levels, with the best line always shown down to its leaf.
#  * file: every turn the tree is also written to TREE_DIR/turn_NN.txt (NN =
#    number of discs on the board). It holds the COMPLETE tree when it has at
#    most SAVE_LIMIT nodes; above that the file holds the summarised tree and
#    says so (a 3-million-node tree would be gigabytes of text).
FULL_PRINT_LIMIT = 3000
SAVE_TREES = True
SAVE_LIMIT = 100_000
TREE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees")


def evaluate(board):
    """Heuristic value of `board` for the computer (heuristic v2 from heuristic.py)."""
    return evaluate_v2(board, V2_MINIMAX)


def get_move(board, algorithm, k):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    depth = k
    global _weights
    _weights = V2_EXPECTI if algorithm == "expectiminimax" else V2_MINIMAX
    start = time.perf_counter()
    counter = [0]
    root = TreeNode(MAX)
    if algorithm == "expectiminimax":
        _expecti(board, depth, AI, root, counter)
    elif algorithm == "alphabeta":
        _minimax(board, depth, AI, root, counter, -math.inf, math.inf)
    else:
        _minimax(board, depth, AI, root, counter)
    best = principal_path(root)[1]
    elapsed = time.perf_counter() - start
    if not QUIET:
        _show_tree(root, board, algorithm, depth,
                   {"nodes": counter[0], "column": best.column(), "time": elapsed})
    return best.column(), {"nodes": counter[0], "value": root.value, "tree": root}


NAMES = {"minimax": "Minimax", "alphabeta": "Alpha-Beta",
         "expectiminimax": "Expectiminimax"}


def _save_tree(root, board, title, stats, total):
    """Write this turn's tree to TREE_DIR/turn_NN.txt. Returns the path, or
    None if it could not be written (the game must never crash because of it)."""
    try:
        os.makedirs(TREE_DIR, exist_ok=True)
        turn = board.move_count
        if turn <= 1:                                  # a new game started:
            for old in glob.glob(os.path.join(TREE_DIR, "turn_*.txt")):
                os.remove(old)                         # drop the previous game's files
        path = os.path.join(TREE_DIR, f"turn_{turn:02d}.txt")
        complete = total <= SAVE_LIMIT
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Computer's move after {turn} discs on the board - {title}\n")
            f.write("Position the search started from:\n")
            f.write(str(board) + "\n\n")
            if not complete:
                f.write(f"NOTE: the tree has {total:,} nodes (more than SAVE_LIMIT = "
                        f"{SAVE_LIMIT:,}); only the top {PRINT_DEPTH} levels and the best "
                        "line are saved.\n\n")
            print_tree(root, None if complete else PRINT_DEPTH, title=title,
                       stats=stats, file=f)
        return path, complete
    except OSError as e:
        print(f"(could not save the tree: {e})")
        return None, False


def _show_tree(root, board, algorithm, depth, stats):
    """Console output (whole tree if small, else summary) + the file copy."""
    title = f"{NAMES[algorithm]}, K={depth}"
    total = subtree_size(root)
    saved, complete = (_save_tree(root, board, title, stats, total)
                       if SAVE_TREES else (None, False))
    print_tree(root, None if total <= FULL_PRINT_LIMIT else PRINT_DEPTH,
               title=title, stats=stats)
    if saved:
        what = "Full tree" if complete else "Summarised tree (too big to save in full)"
        print(f"{what} saved to {os.path.relpath(saved)}")


_weights = V2_MINIMAX


def _evaluate(board):
    """Leaf value: heuristic.py with the weights for the algorithm."""
    return evaluate_v2(board, _weights)


def _minimax(board, depth, player, node, counter, alpha=None, beta=None):
    counter[0] += 1
    moves = board.valid_moves()
    if depth == 0 or not moves:
        node.kind, node.value = LEAF, _evaluate(board)
        return node.value
    node.kind = MAX if player == AI else MIN
    use_ab = alpha is not None
    best = -math.inf if player == AI else math.inf
    for i, col in enumerate(moves):
        child = TreeNode(move=col, player=player)
        node.children.append(child)
        board.drop(col, player)
        v = _minimax(board, depth - 1, opponent(player), child, counter, alpha, beta)
        board.undo(col)
        if player == AI:
            best = max(best, v)
            if use_ab:
                alpha = max(alpha, best)
        else:
            best = min(best, v)
            if use_ab:
                beta = min(beta, best)
        if use_ab and alpha >= beta:
            next_kind = MIN if player == AI else MAX
            for rest in moves[i + 1:]:
                node.children.append(TreeNode(next_kind, move=rest,
                                              player=player, pruned=True))
            break
    node.value = best
    if use_ab:
        node.alpha, node.beta = alpha, beta
    return best


def _expecti(board, depth, player, node, counter):
    counter[0] += 1
    moves = board.valid_moves()
    if depth == 0 or not moves:
        node.kind, node.value = LEAF, _evaluate(board)
        return node.value
    node.kind = MAX if player == AI else MIN
    best = -math.inf if player == AI else math.inf
    for col in moves:
        chance = TreeNode(CHANCE, aimed=col, player=player)
        node.children.append(chance)
        counter[0] += 1
        expected = 0.0
        for land, p in board.chance_outcomes(col):
            outcome = TreeNode(move=land, player=player, prob=p)
            chance.children.append(outcome)
            board.drop(land, player)
            expected += p * _expecti(board, depth - 1, opponent(player), outcome, counter)
            board.undo(land)
        chance.value = expected
        best = max(best, expected) if player == AI else min(best, expected)
    node.value = best
    return best
