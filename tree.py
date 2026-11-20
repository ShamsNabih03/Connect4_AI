"""
tree.py - the search-tree data structure shared by the AI and the GUI.

The AI builds one TreeNode per node it visits during a turn and returns the
root in info["tree"] (see ai_interface.py). The GUI tree viewer (bonus,
Section 2.2) draws it; the console printer can walk the same structure.

Node kinds
----------
MAX    : the computer chooses a move (takes the maximum child value)
MIN    : the human chooses a move (takes the minimum child value)
CHANCE : Expectiminimax only - the player AIMED at a column; the children
         are the columns the disc may really land in, each with `prob`
LEAF   : depth limit K reached (value = heuristic) or the board is full

Level order assumed for Expectiminimax:
    MAX -> CHANCE -> MIN -> CHANCE -> MAX -> ... -> LEAF
K counts decision levels (MAX/MIN), chance levels are not counted.

Pruned nodes (Alpha-Beta): the AI should still add the skipped children
with pruned=True (no value) so the viewer can show WHERE pruning happened.
"""

MAX = "max"
MIN = "min"
CHANCE = "chance"
LEAF = "leaf"


class TreeNode:
    # __slots__ keeps each node small - trees can have millions of nodes
    __slots__ = ("kind", "move", "aimed", "player", "prob", "value",
                 "alpha", "beta", "pruned", "children")

    def __init__(self, kind=LEAF, move=None, aimed=None, player=None,
                 prob=None, value=None, pruned=False):
        self.kind = kind        # MAX / MIN / CHANCE / LEAF
        self.move = move        # column the disc LANDED in to reach this node
                                # (None for the root and for CHANCE nodes)
        self.aimed = aimed      # CHANCE nodes: the column the player aimed at
        self.player = player    # board.HUMAN / board.AI - who made that move
        self.prob = prob        # child of a CHANCE node: probability of it
        self.value = value      # minimax / expected / heuristic value
        self.alpha = None       # Alpha-Beta: alpha and beta when the node
        self.beta = None        #             finished (optional, for display)
        self.pruned = pruned    # True if Alpha-Beta skipped this node
        self.children = []

    def column(self):
        """The column this node stands for (landing or aimed column)."""
        return self.move if self.move is not None else self.aimed


def principal_path(root):
    """
    The line of play the search expects: from the root follow the best
    child (max at MAX, min at MIN). Stops at a CHANCE node, because after
    it the outcome is random.
    """
    path, node = [root], root
    while node.children and node.kind in (MAX, MIN):
        options = [c for c in node.children
                   if not c.pruned and c.value is not None]
        if not options:
            break
        pick = max if node.kind == MAX else min
        node = pick(options, key=lambda c: c.value)
        path.append(node)
    return path


def subtree_size(node):
    """Number of nodes in the subtree rooted at `node` (iterative)."""
    count, stack = 0, [node]
    while stack:
        n = stack.pop()
        count += 1
        stack.extend(n.children)
    return count
