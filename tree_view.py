"""
tree_view.py - GUI viewer for the AI's search tree (bonus, Section 2.2).

Opened from the game screen with the "Search tree" button. It shows the
tree the computer built for its last move (and up to 4 earlier ones).

A full tree can have hundreds of thousands of nodes, far too many to draw at
once, so the viewer starts with the root and its children, and you open
branches as you go:

    click a node        expand / collapse its children
    drag                pan            mouse wheel    zoom
    + / -               expand / collapse one whole level
    B                   expand the best line of play (gold path)
    Home                fit the tree on screen
    hover a node        details + the board position at that node
    Esc                 back to the game

Colours: MAX (computer) green, MIN (human) red, CHANCE purple,
LEAF (heuristic value) slate, pruned nodes dark grey. The gold path is
what the search expects to happen (principal variation).
"""

import math

import pygame

from board import HUMAN, AI, EMPTY
from tree import MAX, MIN, CHANCE, LEAF, principal_path, subtree_size
from ui import (BG, PANEL, TEXT, MUTED, BOARD_BLUE, HOLE_BG, PIECE_COLORS,
                font, draw_text, Button)

# World-space layout sizes (before zoom)
NODE_W, NODE_H = 76, 46
SLOT = 90             # horizontal space for each visible leaf
LEVEL = 120           # vertical distance between levels
HEADER_H = 112        # title + toolbar
FOOTER_H = 34         # legend
MAX_VISIBLE = 3000    # refuse "expand level" beyond this many drawn nodes
ZOOM_MIN, ZOOM_MAX = 0.2, 2.0
FIT_MIN_ZOOM = 0.6    # "Fit" never zooms out further than this

STYLE = {
    MAX: ((40, 150, 92), "MAX"),
    MIN: ((192, 66, 66), "MIN"),
    CHANCE: ((122, 88, 198), "CHANCE"),
    LEAF: ((74, 84, 112), "h(n)"),
}
GOLD = (250, 200, 60)
PRUNED_FILL = (48, 53, 68)
EDGE = (105, 115, 140)
EDGE_PRUNED = (62, 68, 86)
WHO = {HUMAN: "You", AI: "Computer", None: "?"}


def fmt(v):
    """Short, readable number for a node value."""
    if v is None:
        return "-"
    if v == math.inf:
        return "+inf"
    if v == -math.inf:
        return "-inf"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}"


class TreeScreen:
    W, H = 1240, 720
    TOP = HEADER_H + 50          # screen y of the root

    def __init__(self, app, game):
        self.app = app
        self.game = game
        app.set_window(self.W, self.H)
        self.index = len(game.history) - 1     # newest tree first
        self.dragging = False
        self.notice = ""
        y, s = 64, 17
        self.buttons = [
            Button((16, y, 104, 38), "< Back", self.close, size=s),
            Button((134, y, 108, 38), "< Prev", lambda: self._go(-1),
                   enabled=lambda: self.index > 0, size=s),
            Button((248, y, 108, 38), "Next >", lambda: self._go(+1),
                   enabled=lambda: self.index < len(self.game.history) - 1, size=s),
            Button((372, y, 132, 38), "Expand level", self.expand_level, size=s),
            Button((510, y, 140, 38), "Collapse level", self.collapse_level, size=s),
            Button((656, y, 112, 38), "Best path", self.expand_best, size=s),
            Button((774, y, 80, 38), "Fit", self.fit, size=s),
        ]
        self._load()

    # ------------------------------------------------------------------
    # Loading a tree and laying it out
    # ------------------------------------------------------------------
    def _go(self, step):
        self.index += step
        self._load()

    def _load(self):
        self.entry = self.game.history[self.index]
        self.root = self.entry["tree"]
        if "size" not in self.entry:
            self.entry["size"] = subtree_size(self.root)
        self.best_ids = {id(n) for n in principal_path(self.root)}
        self.expanded = {id(self.root)}
        self.size_cache = {}
        self.notice = ""
        self._layout()
        self.fit()

    def _layout(self):
        """
        Tidy top-down layout of the VISIBLE part of the tree: every visible
        leaf gets its own horizontal slot, every parent is centred above its
        children. Positions are stored in world units (before zoom/pan).
        """
        self.pos, self.parent, self.nodes, self.edges = {}, {}, [], []
        next_slot = [0]

        def place(node, depth):
            self.nodes.append(node)
            kids = node.children if id(node) in self.expanded else ()
            if kids:
                xs = []
                for child in kids:
                    self.parent[id(child)] = node
                    self.edges.append((node, child))
                    xs.append(place(child, depth + 1))
                x = (xs[0] + xs[-1]) / 2
            else:
                x = next_slot[0]
                next_slot[0] += 1
            self.pos[id(node)] = (x * SLOT, depth * LEVEL)
            return x

        place(self.root, 0)

    def _relayout_keeping(self, node):
        """Re-layout but keep `node` at the same place on screen."""
        before = self.to_screen(*self.pos[id(node)])
        self._layout()
        wx, wy = self.pos[id(node)]
        self.cam_x = wx - (before[0] - self.W / 2) / self.zoom
        self.cam_y = wy - (before[1] - self.TOP) / self.zoom

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def to_screen(self, wx, wy):
        return ((wx - self.cam_x) * self.zoom + self.W / 2,
                (wy - self.cam_y) * self.zoom + self.TOP)

    def to_world(self, sx, sy):
        return ((sx - self.W / 2) / self.zoom + self.cam_x,
                (sy - self.TOP) / self.zoom + self.cam_y)

    def fit(self):
        """
        Zoom and pan so the visible tree fits the window (max zoom 1).
        Never zooms out below FIT_MIN_ZOOM so labels stay readable; if the
        tree is still too wide, centre on the deepest visible node of the
        best path instead (drag / wheel to see the rest).
        """
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        width = max(xs) - min(xs) + NODE_W + 40
        height = max(ys) - min(ys) + NODE_H + 60
        avail_h = self.H - self.TOP - FOOTER_H - 20
        wanted = min(1.0, (self.W - 40) / width, avail_h / height)
        self.zoom = max(FIT_MIN_ZOOM, wanted)
        self.cam_x = (min(xs) + max(xs)) / 2
        if wanted < FIT_MIN_ZOOM:
            on_path = [n for n in self.nodes if id(n) in self.best_ids]
            deepest = max(on_path, key=lambda n: self.pos[id(n)][1])
            self.cam_x = self.pos[id(deepest)][0]
        self.cam_y = 0

    def node_rect(self, node):
        sx, sy = self.to_screen(*self.pos[id(node)])
        w, h = NODE_W * self.zoom, NODE_H * self.zoom
        return pygame.Rect(int(sx - w / 2), int(sy - h / 2), int(w), int(h))

    def _node_at(self, point):
        for node in reversed(self.nodes):
            if self.node_rect(node).collidepoint(point):
                return node
        return None

    # ------------------------------------------------------------------
    # Expanding / collapsing
    # ------------------------------------------------------------------
    def _collapse(self, node):
        """Close `node` and every open branch below it."""
        stack = [node]
        while stack:
            n = stack.pop()
            if id(n) in self.expanded:
                self.expanded.discard(id(n))
                stack.extend(n.children)

    def toggle(self, node):
        if id(node) in self.expanded:
            self._collapse(node)
            self.notice = ""
        else:
            if len(self.nodes) + len(node.children) > MAX_VISIBLE:
                self.notice = "Too many nodes on screen - collapse something first."
                return
            self.expanded.add(id(node))
        self._relayout_keeping(node)

    def expand_level(self):
        frontier = [n for n in self.nodes
                    if n.children and id(n) not in self.expanded]
        if not frontier:
            self.notice = "Whole tree is already open."
            return
        extra = sum(len(n.children) for n in frontier)
        if len(self.nodes) + extra > MAX_VISIBLE:
            self.notice = (f"Next level adds {extra:,} nodes - too many. "
                           "Click individual nodes instead.")
            return
        self.expanded.update(id(n) for n in frontier)
        self.notice = ""
        self._layout()
        self.fit()

    def collapse_level(self):
        depth = {}
        for n in self.nodes:      # depth from the parent chain
            p = self.parent.get(id(n))
            depth[id(n)] = 0 if p is None else depth[id(p)] + 1
        open_nodes = [n for n in self.nodes
                      if id(n) in self.expanded and n is not self.root]
        if not open_nodes:
            return
        deepest = max(depth[id(n)] for n in open_nodes)
        for n in open_nodes:
            if depth[id(n)] == deepest:
                self.expanded.discard(id(n))
        self.notice = ""
        self._layout()
        self.fit()

    def expand_best(self):
        """Open the gold path: what the search expects to happen."""
        for n in principal_path(self.root)[:-1]:
            self.expanded.add(id(n))
        self._layout()
        self.fit()

    def close(self):
        self.app.back_to_game(self.game)

    # ------------------------------------------------------------------
    # Main-loop hooks
    # ------------------------------------------------------------------
    def handle(self, event):
        for b in self.buttons:
            if b.handle(event):
                return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[1] < HEADER_H:
                return
            node = self._node_at(event.pos)
            if node is not None and node.children:
                self.toggle(node)
            else:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.cam_x -= event.rel[0] / self.zoom
            self.cam_y -= event.rel[1] / self.zoom
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            wx, wy = self.to_world(mx, my)
            self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * 1.15 ** event.y))
            # keep the point under the mouse fixed while zooming
            self.cam_x = wx - (mx - self.W / 2) / self.zoom
            self.cam_y = wy - (my - self.TOP) / self.zoom
        elif event.type == pygame.KEYDOWN:
            step = 80 / self.zoom
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                self.close()
            elif event.key == pygame.K_LEFT:
                self.cam_x -= step
            elif event.key == pygame.K_RIGHT:
                self.cam_x += step
            elif event.key == pygame.K_UP:
                self.cam_y -= step
            elif event.key == pygame.K_DOWN:
                self.cam_y += step
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.expand_level()
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.collapse_level()
            elif event.key == pygame.K_b:
                self.expand_best()
            elif event.key == pygame.K_HOME:
                self.fit()

    def update(self):
        pass

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, surf, mouse):
        surf.fill(BG)
        z = self.zoom
        area = pygame.Rect(0, HEADER_H, self.W, self.H - HEADER_H - FOOTER_H)
        surf.set_clip(area)

        hovered = self._node_at(mouse) if area.collidepoint(mouse) else None

        # Edges (and their move labels)
        label_font = font(13 * z) if z >= 0.55 else None
        for parent, child in self.edges:
            x1, y1 = self.to_screen(*self.pos[id(parent)])
            x2, y2 = self.to_screen(*self.pos[id(child)])
            y1 += NODE_H * z / 2
            y2 -= NODE_H * z / 2
            if max(x1, x2) < 0 or min(x1, x2) > self.W:
                continue
            on_best = id(parent) in self.best_ids and id(child) in self.best_ids
            color = GOLD if on_best else (EDGE_PRUNED if child.pruned else EDGE)
            pygame.draw.line(surf, color, (x1, y1), (x2, y2), 3 if on_best else 1)
            if label_font:
                draw_text(surf, self._edge_label(child), label_font,
                          GOLD if on_best else MUTED, (x2, y2 - 2), "midbottom")

        # Nodes
        title_font, value_font = font(12 * z, True), font(17 * z, True)
        for node in self.nodes:
            rect = self.node_rect(node)
            if not rect.colliderect(area):
                continue
            self._draw_node(surf, node, rect, node is hovered, title_font, value_font)

        surf.set_clip(None)
        self._draw_header(surf, mouse)
        self._draw_footer(surf)
        if hovered is not None:
            self._draw_tooltip(surf, hovered, mouse)

    def _edge_label(self, child):
        if child.kind == CHANCE:
            return f"aim {child.aimed}"
        if child.prob is not None:
            return f"col {child.move}  p={child.prob:g}"
        return f"col {child.move}"

    def _draw_node(self, surf, node, rect, hovered, title_font, value_font):
        z = self.zoom
        color, title = STYLE[node.kind]
        if node.pruned:
            color, title = PRUNED_FILL, "pruned"
        radius = max(2, int(9 * z))
        if node.kind == CHANCE and not node.pruned:
            pygame.draw.ellipse(surf, color, rect)
        else:
            pygame.draw.rect(surf, color, rect, border_radius=radius)

        if id(node) in self.best_ids:
            outline, width = GOLD, max(2, int(3 * z))
        elif hovered:
            outline, width = TEXT, 2
        else:
            outline, width = None, 0
        if outline:
            if node.kind == CHANCE and not node.pruned:
                pygame.draw.ellipse(surf, outline, rect, width)
            else:
                pygame.draw.rect(surf, outline, rect, width, border_radius=radius)

        txt = MUTED if node.pruned else TEXT
        if z >= 0.45:
            draw_text(surf, title, title_font, txt,
                      (rect.centerx, rect.top + 3 * z), "midtop")
            value = "x" if node.pruned else fmt(node.value)
            draw_text(surf, value, value_font, txt,
                      (rect.centerx, rect.bottom - 3 * z), "midbottom")
        else:
            draw_text(surf, "x" if node.pruned else fmt(node.value),
                      value_font, txt, rect.center, "center")

        # Collapsed node that has children: show how many are hidden
        if node.children and id(node) not in self.expanded and z >= 0.4:
            draw_text(surf, f"+{len(node.children)}", font(12 * z, True), MUTED,
                      (rect.centerx, rect.bottom + 2), "midtop")

    def _draw_header(self, surf, mouse):
        e = self.entry
        pygame.draw.rect(surf, PANEL, (0, 0, self.W, HEADER_H))
        draw_text(surf, f"Search tree - computer's move #{e['move_no']}",
                  font(24, True), TEXT, (16, 10))
        draw_text(surf, f"tree {self.index + 1} of {len(self.game.history)}",
                  font(15), MUTED, (self.W - 16, 18), "topright")
        info = (f"{e['algorithm_name']}  |  K = {e['k']}  |  chose column {e['column']}"
                f"  |  value {fmt(e['value'])}  |  nodes expanded {e['nodes']:,}"
                f"  |  {e['time']:.3f} s  |  showing {len(self.nodes):,} of "
                f"{e['size']:,} nodes")
        draw_text(surf, info, font(15), MUTED, (16, 40))
        for b in self.buttons:
            b.draw(surf, mouse)
        if self.notice:
            draw_text(surf, self.notice, font(15, True), GOLD, (868, 83), "midleft")

    def _draw_footer(self, surf):
        y = self.H - FOOTER_H
        pygame.draw.rect(surf, PANEL, (0, y, self.W, FOOTER_H))
        x = 14
        f = font(14)
        items = [(STYLE[MAX][0], "MAX (computer)"), (STYLE[MIN][0], "MIN (you)"),
                 (STYLE[CHANCE][0], "CHANCE"), (STYLE[LEAF][0], "leaf = heuristic"),
                 (PRUNED_FILL, "pruned"), (GOLD, "best path")]
        for color, label in items:
            pygame.draw.rect(surf, color, (x, y + 10, 14, 14), border_radius=3)
            x = draw_text(surf, label, f, TEXT, (x + 20, y + 17), "midleft").right + 18
        draw_text(surf, "click: open/close  |  drag: pan  |  wheel: zoom  |  "
                        "+/-: levels  |  B: best path  |  Esc: back",
                  f, MUTED, (self.W - 14, y + 17), "midright")

    # ------------------------------------------------------------------
    # Hover details with a mini board
    # ------------------------------------------------------------------
    def _board_at(self, node):
        """Rebuild the position at `node` by replaying moves from the root."""
        path = []
        n = node
        while n is not self.root:
            path.append(n)
            n = self.parent[id(n)]
        board = self.entry["board"].copy()
        for n in reversed(path):
            if n.move is not None:
                board.drop(n.move, n.player)
        return board

    def _draw_tooltip(self, surf, node, mouse):
        kind_text = {MAX: "MAX node - computer to move",
                     MIN: "MIN node - you to move",
                     CHANCE: "CHANCE node - random slip",
                     LEAF: "LEAF - heuristic evaluation"}[node.kind]
        lines = ["PRUNED by alpha-beta (never evaluated)" if node.pruned else kind_text]
        if node.kind == CHANCE:
            lines.append(f"{WHO[node.player]} aimed at column {node.aimed}")
        elif node.move is not None:
            line = f"{WHO[node.player]}'s disc landed in column {node.move}"
            if node.prob is not None:
                line += f"  (p = {node.prob:g})"
            lines.append(line)
        if not node.pruned:
            label = {LEAF: "Heuristic value", CHANCE: "Expected value"}.get(node.kind, "Value")
            lines.append(f"{label}: {fmt(node.value)}")
        if node.alpha is not None or node.beta is not None:
            lines.append(f"alpha = {fmt(node.alpha)},  beta = {fmt(node.beta)}")
        if node.children:
            pruned = sum(1 for c in node.children if c.pruned)
            line = f"Children: {len(node.children)}"
            if pruned:
                line += f"  ({pruned} pruned)"
            lines.append(line)
            if id(node) not in self.size_cache:
                self.size_cache[id(node)] = subtree_size(node)
            lines.append(f"Nodes in this subtree: {self.size_cache[id(node)]:,}")

        board = self._board_at(node)
        f = font(15)
        cell = 18
        bw, bh = board.cols * cell, board.rows * cell
        text_w = max(f.size(t)[0] for t in lines)
        w = max(text_w, bw) + 24
        h = 14 + len(lines) * 21 + 10 + bh + 14
        x = min(mouse[0] + 18, self.W - w - 6)
        y = min(mouse[1] + 18, self.H - h - 6)
        if x < mouse[0] and y < mouse[1]:          # would cover the cursor
            y = max(6, mouse[1] - h - 12)
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (12, 14, 22), box, border_radius=8)
        pygame.draw.rect(surf, MUTED, box, 1, border_radius=8)
        ty = y + 10
        for i, t in enumerate(lines):
            draw_text(surf, t, font(15, i == 0), TEXT if i == 0 else MUTED, (x + 12, ty))
            ty += 21
        ty += 8
        bx = x + 12
        pygame.draw.rect(surf, BOARD_BLUE, (bx, ty, bw, bh), border_radius=4)
        last = node.move if node.move is not None else None
        for r in range(board.rows):
            for c in range(board.cols):
                v = board.grid[r][c]
                center = (bx + c * cell + cell // 2, ty + r * cell + cell // 2)
                pygame.draw.circle(surf, PIECE_COLORS[v] if v != EMPTY else HOLE_BG,
                                   center, cell // 2 - 2)
        # ring the disc that this node's move placed
        if last is not None:
            r = board.rows - board.heights[last]
            pygame.draw.circle(surf, TEXT, (bx + last * cell + cell // 2,
                                            ty + r * cell + cell // 2), cell // 2 - 1, 2)
