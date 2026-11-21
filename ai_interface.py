"""
ai_interface.py - the contract between the GUI (Person A) and the AI agent.

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

---------------------------------------------------------------------------
PLACEHOLDER BELOW - replace with the real agent.
It exists only so the GUI and the tree viewer can be built and demonstrated
before the real agent is ready: it uses the CURRENT SCORE DIFFERENCE as a
stand-in heuristic and caps the depth at 3. To plug in the real agent:

    from agent import choose_move, heuristic
    def get_move(board, algorithm, k):
        return choose_move(board, algorithm, k)
    def evaluate(board):
        return heuristic(board)
---------------------------------------------------------------------------
"""

import math
import time

from board import AI, HUMAN, opponent
from heuristic import evaluate_v2, V2_MINIMAX, V2_EXPECTI      # Person B
from tree_printer import print_tree                            # Person B
from tree import TreeNode, MAX, MIN, CHANCE, LEAF, principal_path

ALGORITHMS = ("minimax", "alphabeta", "expectiminimax")
PLACEHOLDER_MAX_DEPTH = 3
QUIET = False          # True -> agent must not print trees (experiments)
PRINT_DEPTH = 3        # tree levels printed in full (best line always full)


def evaluate(board):
    """Heuristic value of `board` for the computer (Person B's heuristic v2)."""
    return evaluate_v2(board, V2_MINIMAX)


def get_move(board, algorithm, k):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    depth = min(k, PLACEHOLDER_MAX_DEPTH)
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
    if not QUIET:
        names = {"minimax": "Minimax", "alphabeta": "Alpha-Beta",
                 "expectiminimax": "Expectiminimax"}
        print_tree(root, PRINT_DEPTH, title=f"{names[algorithm]}, K={depth}",
                   stats={"nodes": counter[0], "column": best.column(),
                          "time": time.perf_counter() - start})
    return best.column(), {"nodes": counter[0], "value": root.value, "tree": root}


_weights = V2_MINIMAX


def _evaluate(board):
    """Leaf value: Person B's heuristic with the weights for the algorithm."""
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
