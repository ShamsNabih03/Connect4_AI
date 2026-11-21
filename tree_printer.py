"""
tree_printer.py - prints the minimax tree to the console (Person B).

Assignment Section 2.2: "it is required to show the minimax tree in each
turn by printing it to the console in a readable form that we can trace".
Section 2.3: "show the values of board after applying the heuristic ...
readable tree printing is a must".

Every line is one node:

    [MAX] value=0.95                         root, computer to move
    +-- col 3 -> [MIN] value=0.95  <== BEST  the move the computer plays
    |   +-- col 2 -> [LEAF] h=0.95           h = heuristic value of that board
    |   +-- col 3 -> [LEAF] h=1.30
    |   `-- col 4 -> (pruned)                skipped by alpha-beta
    `-- aim 6 -> [CHANCE] E=0.41             Expectiminimax: player aimed at 6
        +-- lands 6 (p=0.6) -> [MIN] value=0.50
        `-- lands 5 (p=0.4) -> [MIN] value=0.28

Values are from the computer's point of view (bigger = better for computer).
Pure ASCII is used so it prints correctly in every Windows/Linux console.

Big trees: a K=6 tree can have >100,000 nodes - nobody can read that. Levels
deeper than `max_depth` are summarised in one line ("... 812 more nodes"), and
the best line of play is always printed down to the leaves, so the decision
can still be traced. Pass max_depth=None to print everything.
"""

import math
import sys

from tree import MAX, MIN, CHANCE, LEAF, principal_path, subtree_size

ROLE = {MAX: "MAX", MIN: "MIN", CHANCE: "CHANCE", LEAF: "LEAF"}
WHO = {1: "human", 2: "computer"}


def fmt(v):
    """Short number: 0.95, 3, +inf."""
    if v is None:
        return "-"
    if v == math.inf:
        return "+inf"
    if v == -math.inf:
        return "-inf"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}"


def _edge(node):
    """How we got to this node from its parent."""
    if node.kind == CHANCE:
        return f"aim {node.aimed}"
    if node.prob is not None:
        return f"lands {node.move} (p={node.prob:g})"
    if node.move is not None:
        return f"col {node.move}"
    return ""


def _label(node):
    """The node itself: role and value (h= for heuristic leaves)."""
    if node.pruned:
        return "(pruned)"
    role = ROLE.get(node.kind, str(node.kind))
    if node.kind == LEAF:
        text = f"[LEAF] h={fmt(node.value)}"
    elif node.kind == CHANCE:
        text = f"[CHANCE] E={fmt(node.value)}"
    else:
        text = f"[{role}] value={fmt(node.value)}"
    if node.alpha is not None or node.beta is not None:
        text += f"  (a={fmt(node.alpha)}, b={fmt(node.beta)})"
    return text


def _best_line(root, max_depth):
    """
    ids of the nodes on the expected line of play: the best child at MAX/MIN
    nodes. At a CHANCE node the outcome is random, so every outcome is
    followed while within max_depth, and only the most likely one below it
    (keeps the printout small for big K).
    """
    ids, stack = set(), [(root, 0)]
    while stack:
        node, depth = stack.pop()
        ids.add(id(node))
        kids = [c for c in node.children if not c.pruned and c.value is not None]
        if not kids:
            continue
        if node.kind == CHANCE:
            if max_depth is None or depth < max_depth:
                stack.extend((c, depth) for c in kids)
            else:
                stack.append((max(kids, key=lambda c: c.prob or 0), depth))
        elif node.kind in (MAX, MIN):
            pick = max if node.kind == MAX else min
            stack.append((pick(kids, key=lambda c: c.value), depth + 1))
    return ids


def tree_lines(root, max_depth=3, show_best=True):
    """
    Return the tree as a list of text lines (see module docstring).
    Depth counts decision levels (MAX/MIN); a CHANCE node and its outcomes
    belong to the same level, matching how K is counted.
    """
    best = _best_line(root, max_depth) if show_best else set()
    chosen = None
    if root.children:
        path = principal_path(root)
        chosen = path[1] if len(path) > 1 else None
    lines = [_label(root)]

    # iterative DFS (a recursive one could hit Python's recursion limit)
    stack = [(child, 1, "", i == len(root.children) - 1)
             for i, child in reversed(list(enumerate(root.children)))]
    while stack:
        node, depth, prefix, last = stack.pop()
        branch = "`-- " if last else "+-- "
        edge = _edge(node)
        text = f"{prefix}{branch}{edge} -> {_label(node)}" if edge else \
               f"{prefix}{branch}{_label(node)}"
        if node is chosen:
            text += "  <== BEST MOVE"
        elif id(node) in best:
            text += "  <== expected"
        lines.append(text)

        if not node.children:
            continue
        child_prefix = prefix + ("    " if last else "|   ")
        on_best = id(node) in best
        if max_depth is not None and depth >= max_depth and not on_best:
            hidden = subtree_size(node) - 1
            lines.append(f"{child_prefix}`-- ... {hidden:,} more node"
                         f"{'s' if hidden != 1 else ''} below (depth limit)")
            continue
        kids = node.children
        nxt = depth if node.kind == CHANCE else depth + 1
        for i in range(len(kids) - 1, -1, -1):
            stack.append((kids[i], nxt, child_prefix, i == len(kids) - 1))
    return lines


def print_tree(root, max_depth=3, title=None, stats=None, file=None):
    """
    Print the search tree of one AI turn.
      root      : tree.TreeNode returned by the search
      max_depth : levels printed in full (best line always printed fully);
                  None = print the whole tree
      title     : optional header, e.g. "Alpha-Beta, K=4"
      stats     : optional dict with "nodes", "time", "column"
    """
    out = file or sys.stdout
    total = subtree_size(root)
    bar = "=" * 72
    print(bar, file=out)
    head = "MINIMAX TREE" + (f" - {title}" if title else "")
    print(head, file=out)
    info = [f"nodes in tree: {total:,}"]
    if stats:
        if "nodes" in stats:
            info.append(f"nodes expanded: {stats['nodes']:,}")
        if "time" in stats:
            info.append(f"time: {stats['time']:.3f} s")
        if "column" in stats:
            info.append(f"chosen column: {stats['column']}")
    print("   ".join(info), file=out)
    print("values = heuristic h(board) at leaves, backed-up minimax/expected "
          "values above; computer's point of view (bigger = better for computer)",
          file=out)
    if max_depth is not None and total > 1:
        print(f"(levels below depth {max_depth} are summarised; "
              "the best line is shown down to its leaf)", file=out)
    print(bar, file=out)
    for line in tree_lines(root, max_depth):
        print(line, file=out)
    print(bar, file=out)
