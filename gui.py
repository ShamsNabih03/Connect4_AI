"""
gui.py - Pygame GUI for Connect 4, human vs. computer (Assignment Section 2.1).

Run:  python gui.py

Screens
-------
1. Settings screen : board rows (>= 6) and columns (>= 7), depth K,
                     AI algorithm (Minimax / Alpha-Beta / Expectiminimax)
                     and who moves first.
2. Game screen     : the board with falling-disc animation, live scores
                     (number of connected-fours), whose turn it is, the AI's
                     time / nodes for its last move, and a game-over banner
                     once the board is full.

Design notes
------------
* The AI runs on a background thread so the window never freezes while it
  searches (high K can take seconds). The AI receives a COPY of the board.
* Randomness in the real game (assumption): when Expectiminimax is selected,
  EVERY disc - human's and AI's - obeys the slip rule (0.6 chosen column,
  0.2 / 0.2 neighbours, 0.6 / 0.4 at an edge or next to a full column).
  With Minimax / Alpha-Beta the game is deterministic.
* Columns are numbered from 0, matching the console output.
"""

import sys
import threading
import time

import pygame

import ai_interface
from board import Board, HUMAN, AI, EMPTY
from tree_view import TreeScreen, fmt
from ui import (FPS, BG, PANEL, BOARD_BLUE, BOARD_EDGE, HOLE_BG, TEXT, MUTED,
                HUMAN_COLOR, AI_COLOR, PIECE_COLORS, font, draw_text, wrap_text,
                Button)

PANEL_W = 310
TREE_HISTORY = 5      # how many past AI trees the viewer keeps (memory)

ALGORITHMS = [("minimax", "Minimax"),
              ("alphabeta", "Alpha-Beta"),
              ("expectiminimax", "Expectiminimax")]
ALGO_NAMES = dict(ALGORITHMS)

# Allowed ranges on the settings screen
ROWS_RANGE = (6, 10)
COLS_RANGE = (7, 12)
K_RANGE = (1, 10)


# ----------------------------------------------------------------------
# Settings chosen before the game starts
# ----------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.k = 4
        self.algorithm = "alphabeta"
        self.first = HUMAN


class SettingsScreen:
    """Lets the player choose board size, K, algorithm and first player."""

    W, H = 900, 640

    def __init__(self, app, settings):
        self.app = app
        self.s = settings
        app.set_window(self.W, self.H)
        self.title_font = font(46, bold=True)
        self.label_font = font(24, bold=True)
        self.value_font = font(30, bold=True)
        self.hint_font = font(18)
        self.buttons = []
        self._build()

    def _stepper(self, y, attr, lo, hi):
        """[-] value [+] control for an integer setting."""
        def change(d):
            def f():
                setattr(self.s, attr, max(lo, min(hi, getattr(self.s, attr) + d)))
            return f
        self.buttons.append(Button((420, y, 48, 44), "-", change(-1),
                                   enabled=lambda: getattr(self.s, attr) > lo, size=28))
        self.buttons.append(Button((560, y, 48, 44), "+", change(+1),
                                   enabled=lambda: getattr(self.s, attr) < hi, size=28))

    def _build(self):
        self._stepper(150, "rows", *ROWS_RANGE)
        self._stepper(215, "cols", *COLS_RANGE)
        self._stepper(280, "k", *K_RANGE)

        x = 300
        for key, name in ALGORITHMS:
            self.buttons.append(Button(
                (x, 360, 180, 46), name,
                lambda key=key: setattr(self.s, "algorithm", key),
                selected=lambda key=key: self.s.algorithm == key))
            x += 195

        for i, (who, name) in enumerate([(HUMAN, "You"), (AI, "Computer")]):
            self.buttons.append(Button(
                (300 + i * 195, 440, 180, 46), name,
                lambda who=who: setattr(self.s, "first", who),
                selected=lambda who=who: self.s.first == who))

        self.buttons.append(Button((self.W // 2 - 130, 545, 260, 60),
                                   "Start game", self.app.start_game, size=26))

    def handle(self, event):
        for b in self.buttons:
            if b.handle(event):
                return

    def update(self):
        pass

    def draw(self, surf, mouse):
        surf.fill(BG)
        draw_text(surf, "Connect 4  -  Human vs. Computer", self.title_font,
                  TEXT, (self.W // 2, 60), "center")

        rows = [("Rows", self.s.rows, 150), ("Columns", self.s.cols, 215),
                ("Depth K", self.s.k, 280)]
        for label, value, y in rows:
            draw_text(surf, label, self.label_font, TEXT, (110, y + 22), "midleft")
            draw_text(surf, str(value), self.value_font, TEXT, (514, y + 22), "center")
        draw_text(surf, "AI algorithm", self.label_font, TEXT, (110, 383), "midleft")
        draw_text(surf, "First move", self.label_font, TEXT, (110, 463), "midleft")

        draw_text(surf, f"Board {self.s.rows} x {self.s.cols}   (min 6 x 7)",
                  self.hint_font, MUTED, (630, 237), "midleft")
        draw_text(surf, "tree depth", self.hint_font, MUTED, (630, 302), "midleft")
        if self.s.algorithm == "expectiminimax":
            note = "Random drops ON: discs land 0.6 in the chosen column, 0.2 left / 0.2 right."
        else:
            note = "Deterministic game: discs always land in the chosen column."
        draw_text(surf, note, self.hint_font, MUTED, (300, 412), "topleft")

        for b in self.buttons:
            b.draw(surf, mouse)


# ----------------------------------------------------------------------
# The game itself
# ----------------------------------------------------------------------
class GameScreen:
    MARGIN = 28

    def __init__(self, app, settings):
        self.app = app
        self.s = settings
        self.board = Board(settings.rows, settings.cols)
        self.stochastic = settings.algorithm == "expectiminimax"

        # Size the cells so any allowed board fits on a typical laptop screen
        self.cell = int(min(90, 780 / settings.cols, 600 / (settings.rows + 1)))
        self.board_x = self.MARGIN
        self.board_y = self.MARGIN + self.cell          # one row for the preview disc
        self.board_w = settings.cols * self.cell
        self.board_h = settings.rows * self.cell
        self.panel_x = self.board_x + self.board_w + self.MARGIN
        width = self.panel_x + PANEL_W
        height = max(self.board_y + self.board_h + 74, 620)   # room for h row
        app.set_window(width, height)
        self.width, self.height = width, height
        self.history = []             # last few AI search trees (for the viewer)

        self.overlay = self._make_board_overlay()
        self.fonts = {"title": font(28, True), "big": font(40, True),
                      "body": font(19), "bold": font(19, True),
                      "small": font(15), "num": font(16, True)}

        self.turn = settings.first
        self.anim = None              # falling disc currently animating
        self.last_move = None         # (row, col) of the most recent disc
        self.message = "Your move - click a column." if self.turn == HUMAN else ""
        self.ai_stats = ""
        self.game_over = False

        # AI thread state
        self.thinking = False
        self.think_start = 0.0
        self.ai_result = None
        self.ai_error = None

        bx = self.panel_x + 20
        by = height - 70
        self.buttons = [
            Button((bx, by, 130, 46), "Restart", self.app.start_game),
            Button((bx + 140, by, 130, 46), "Settings", self.app.open_settings),
            Button((bx, by - 56, 170, 46), "Search tree", lambda: self.app.open_tree(self),
                   enabled=lambda: bool(self.history)),
            Button((bx + 180, by - 56, 90, 46), "h values", self._toggle_hints,
                   selected=lambda: self.show_hints, size=17),
        ]

        # Heuristic display (extra): h of the current board, and h of the
        # board after the player to move drops a disc in each column.
        self.show_hints = True
        self.h_now = None
        self.h_cols = {}
        self.h_error = None
        print(f"\n=== New game: {self.s.rows}x{self.s.cols}, "
              f"{ALGO_NAMES[self.s.algorithm]}, K={self.s.k}, "
              f"{'random drops' if self.stochastic else 'deterministic'} ===")
        self._refresh_heuristic()

    # ---------------- drawing helpers ----------------
    def _make_board_overlay(self):
        """Blue board with transparent holes, drawn on top of the discs so
        a falling disc looks like it slides down behind the board."""
        surf = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
        surf.fill(BOARD_BLUE)
        r = int(self.cell * 0.42)
        for row in range(self.s.rows):
            for col in range(self.s.cols):
                center = (col * self.cell + self.cell // 2,
                          row * self.cell + self.cell // 2)
                pygame.draw.circle(surf, (0, 0, 0, 0), center, r)
        pygame.draw.rect(surf, BOARD_EDGE, surf.get_rect(), width=4, border_radius=10)
        return surf

    def _cell_center(self, row, col):
        return (self.board_x + col * self.cell + self.cell // 2,
                self.board_y + row * self.cell + self.cell // 2)

    def _draw_disc(self, surf, center, piece, radius=None):
        r = radius or int(self.cell * 0.40)
        color = PIECE_COLORS[piece]
        pygame.draw.circle(surf, color, center, r)
        # A darker ring and a highlight give the disc some depth
        dark = tuple(int(c * 0.72) for c in color)
        pygame.draw.circle(surf, dark, center, r, width=max(2, r // 7))
        pygame.draw.circle(surf, tuple(min(255, c + 45) for c in color),
                           (center[0] - r // 3, center[1] - r // 3), max(2, r // 5))

    def _hover_col(self, mouse):
        mx, my = mouse
        if self.board_x <= mx < self.board_x + self.board_w and my < self.board_y + self.board_h:
            return (mx - self.board_x) // self.cell
        return None

    def _can_human_move(self):
        return (self.turn == HUMAN and not self.anim and not self.game_over)

    # ---------------- moves ----------------
    def play_move(self, aimed, piece):
        """Start a move: apply the random slip if enabled, then animate."""
        actual = self.board.resolve_stochastic(aimed) if self.stochastic else aimed
        who = "You" if piece == HUMAN else "Computer"
        if actual != aimed:
            self.message = f"{who} aimed at column {aimed} - disc slipped to column {actual}!"
        else:
            self.message = f"{who} played column {actual}."
        print(f"{who} aimed at column {aimed}, landed in column {actual}")
        row = self.board.landing_row(actual)
        self.anim = {"col": actual, "row": row, "piece": piece,
                     "y": float(self.MARGIN + self.cell // 2), "vy": 0.0,
                     "target": self._cell_center(row, actual)[1]}

    def _finish_move(self):
        """Animation ended: put the disc on the board and check the game state."""
        a = self.anim
        self.anim = None
        row = self.board.drop(a["col"], a["piece"])
        self.last_move = (row, a["col"])
        gained = self.board.fours_through(row, a["col"])
        if gained:
            self.message += f"  +{gained} four{'s' if gained > 1 else ''}!"
        h, c = self.board.scores()
        print(self.board)
        print(f"Score  You {h} : {c} Computer\n")

        if self.board.is_full():
            self.game_over = True
            w = self.board.winner()
            print("GAME OVER -", {HUMAN: "You win", AI: "Computer wins",
                                  None: "Draw"}[w])
        else:
            self.turn = AI if a["piece"] == HUMAN else HUMAN
            if self.turn == HUMAN:
                self.message += "  Your move."
        self._refresh_heuristic()

    # ---------------- heuristic display ----------------
    def _toggle_hints(self):
        self.show_hints = not self.show_hints

    def _refresh_heuristic(self):
        """
        Recompute the heuristic numbers shown on screen and print them to the
        console. Values are from the computer's point of view (bigger =
        computer closer to winning). Column values assume the disc lands
        where it is aimed (no random slip).
        """
        try:
            self.h_now = ai_interface.evaluate(self.board)
            self.h_cols = {}
            if not self.board.is_full():
                b = self.board.copy()
                for col in b.valid_moves():
                    b.drop(col, self.turn)
                    self.h_cols[col] = ai_interface.evaluate(b)
                    b.undo(col)
            self.h_error = None
        except Exception as exc:             # heuristic not ready / buggy
            self.h_now, self.h_cols, self.h_error = None, {}, str(exc)
            print("Heuristic error:", repr(exc))
            return
        print(f"Heuristic h(board) = {fmt(self.h_now)}")
        if self.h_cols:
            who = "your" if self.turn == HUMAN else "a computer"
            cols = "  ".join(f"c{c}:{fmt(v)}" for c, v in self.h_cols.items())
            print(f"h after {who} disc in each column -> {cols}")

    def resume(self):
        """Back from the tree viewer: restore this screen's window size."""
        self.app.set_window(self.width, self.height)

    def _start_ai(self):
        board_copy = self.board.copy()
        self.ai_board_before = self.board.copy()   # kept for the tree viewer
        self.thinking = True
        self.think_start = time.perf_counter()
        self.ai_result = self.ai_error = None

        def work():
            try:
                self.ai_result = ai_interface.get_move(board_copy, self.s.algorithm, self.s.k)
            except Exception as exc:          # never let the AI crash the GUI
                self.ai_error = exc

        threading.Thread(target=work, daemon=True).start()

    def _collect_ai(self):
        """Called every frame while the AI thinks; plays its move when ready."""
        if self.ai_result is None and self.ai_error is None:
            return
        elapsed = time.perf_counter() - self.think_start
        self.thinking = False
        valid = self.board.valid_moves()
        if self.ai_error is not None:
            print("AI error:", repr(self.ai_error))
            col, info = valid[0], {}
            self.ai_stats = f"AI error ({self.ai_error}); played column {col}"
        else:
            col, info = self.ai_result
            info = info or {}
            if col not in valid:
                print(f"AI returned invalid column {col}; using {valid[0]}")
                col = valid[0]
            parts = [f"Time: {elapsed:.3f} s"]
            if "nodes" in info:
                parts.append(f"Nodes expanded: {info['nodes']:,}")
            if "value" in info:
                parts.append(f"Value: {info['value']:.2f}")
            self.ai_stats = "\n".join(parts)
            if info.get("tree") is not None:
                self.history.append({
                    "tree": info["tree"], "board": self.ai_board_before,
                    "move_no": self.board.move_count // 2 + 1,
                    "column": col, "value": info.get("value"),
                    "nodes": info.get("nodes", 0), "time": elapsed,
                    "algorithm_name": ALGO_NAMES[self.s.algorithm], "k": self.s.k})
                del self.history[:-TREE_HISTORY]
        self.play_move(col, AI)

    # ---------------- main-loop hooks ----------------
    def handle(self, event):
        for b in self.buttons:
            if b.handle(event):
                return
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self._can_human_move()):
            col = self._hover_col(event.pos)
            if col is not None and self.board.is_valid_move(col):
                self.play_move(col, HUMAN)

    def update(self):
        if self.anim:
            a = self.anim
            a["vy"] += self.cell * 0.025          # gravity
            a["y"] += a["vy"]
            if a["y"] >= a["target"]:
                self._finish_move()
        elif not self.game_over and self.turn == AI:
            if not self.thinking:
                self._start_ai()
            else:
                self._collect_ai()

    def draw(self, surf, mouse):
        surf.fill(BG)
        f = self.fonts

        # Hole backgrounds, placed discs, falling disc, then board on top
        pygame.draw.rect(surf, HOLE_BG, (self.board_x, self.board_y,
                                         self.board_w, self.board_h))
        for r in range(self.s.rows):
            for c in range(self.s.cols):
                v = self.board.grid[r][c]
                if v != EMPTY:
                    self._draw_disc(surf, self._cell_center(r, c), v)
        if self.anim:
            x = self._cell_center(0, self.anim["col"])[0]
            self._draw_disc(surf, (x, int(self.anim["y"])), self.anim["piece"])
        surf.blit(self.overlay, (self.board_x, self.board_y))

        # Ring around the last disc played
        if self.last_move and not self.anim:
            pygame.draw.circle(surf, TEXT, self._cell_center(*self.last_move),
                               int(self.cell * 0.44), width=3)

        # Preview disc over the hovered column (human's turn only)
        hc = self._hover_col(mouse)
        if self._can_human_move() and hc is not None and self.board.is_valid_move(hc):
            x = self._cell_center(0, hc)[0]
            self._draw_disc(surf, (x, self.MARGIN + self.cell // 2), HUMAN)

        # Column numbers under the board
        for c in range(self.s.cols):
            x = self._cell_center(0, c)[0]
            draw_text(surf, str(c), f["num"], MUTED,
                      (x, self.board_y + self.board_h + 14), "center")

        # h of the board after the player to move drops in each column;
        # the best one for that player is highlighted in their colour
        if self.show_hints and self.h_cols and not self.anim:
            pick = max if self.turn == AI else min
            best = pick(self.h_cols.values())
            for c, v in self.h_cols.items():
                x = self._cell_center(0, c)[0]
                color = PIECE_COLORS[self.turn] if v == best else TEXT
                draw_text(surf, fmt(v), f["num"], color,
                          (x, self.board_y + self.board_h + 40), "center")

        self._draw_panel(surf, mouse)
        if self.game_over:
            self._draw_game_over(surf)

    def _draw_panel(self, surf, mouse):
        f = self.fonts
        x0 = self.panel_x
        pygame.draw.rect(surf, PANEL, (x0, 0, PANEL_W, self.height))
        x = x0 + 20
        y = 22
        draw_text(surf, "Connect 4", f["title"], TEXT, (x, y))
        y += 44
        draw_text(surf, f"{ALGO_NAMES[self.s.algorithm]}  |  K = {self.s.k}",
                  f["body"], MUTED, (x, y))
        y += 26
        draw_text(surf, "Random drops ON" if self.stochastic else "Deterministic",
                  f["small"], MUTED, (x, y))
        y += 36

        # Scores
        h, c = self.board.scores()
        for piece, name, score in [(HUMAN, "You", h), (AI, "Computer", c)]:
            self._draw_disc(surf, (x + 14, y + 16), piece, radius=13)
            draw_text(surf, name, f["bold"], TEXT, (x + 38, y + 16), "midleft")
            draw_text(surf, str(score), f["big"], TEXT,
                      (x0 + PANEL_W - 24, y + 16), "midright")
            y += 48
        draw_text(surf, "score = connected-fours", f["small"], MUTED, (x, y))
        y += 24

        # Heuristic value of the current board (computer's point of view)
        if self.h_error:
            draw_text(surf, "Heuristic error - see console", f["small"], HUMAN_COLOR, (x, y))
        elif self.h_now is not None:
            lead = (AI_COLOR if self.h_now > 0 else
                    HUMAN_COLOR if self.h_now < 0 else TEXT)
            r = draw_text(surf, "Heuristic h(board) =", f["body"], TEXT, (x, y))
            draw_text(surf, fmt(self.h_now), f["bold"], lead, (r.right + 6, y))
            if self.show_hints and self.h_cols and not self.game_over:
                who = "you drop" if self.turn == HUMAN else "computer drops"
                draw_text(surf, f"under columns: h if {who} there",
                          f["small"], MUTED, (x, y + 24))
        y += 52

        # Whose turn
        if self.game_over:
            status = "Game over"
        elif self.thinking:
            status = f"Computer thinking... {time.perf_counter() - self.think_start:.1f}s"
        elif self.anim:
            status = "..."
        elif self.turn == HUMAN:
            status = "Your turn"
        else:
            status = "Computer's turn"
        draw_text(surf, status, f["bold"], TEXT, (x, y))
        y += 32

        # Last move message
        for line in wrap_text(self.message, f["body"], PANEL_W - 40):
            draw_text(surf, line, f["body"], TEXT, (x, y))
            y += 24
        y += 12

        # AI stats for its last move
        if self.ai_stats:
            draw_text(surf, "Computer's last move", f["bold"], MUTED, (x, y))
            y += 26
            for line in self.ai_stats.split("\n"):
                for wl in wrap_text(line, f["small"], PANEL_W - 40):
                    draw_text(surf, wl, f["small"], TEXT, (x, y))
                    y += 20

        for b in self.buttons:
            b.draw(surf, mouse)

    def _draw_game_over(self, surf):
        h, c = self.board.scores()
        w = self.board.winner()
        text = {HUMAN: "You win!", AI: "Computer wins!", None: "It's a draw!"}[w]
        banner = pygame.Surface((self.board_w, 120), pygame.SRCALPHA)
        banner.fill((10, 12, 20, 215))
        y = self.board_y + self.board_h // 2 - 60
        surf.blit(banner, (self.board_x, y))
        cx = self.board_x + self.board_w // 2
        draw_text(surf, text, self.fonts["big"], TEXT, (cx, y + 42), "center")
        draw_text(surf, f"{h}  :  {c}", self.fonts["title"], MUTED,
                  (cx, y + 88), "center")


# ----------------------------------------------------------------------
# Application: owns the window and switches between screens
# ----------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Connect 4 - Minimax AI")
        self.surface = None
        self.settings = Settings()
        self.screen = None
        self.open_settings()

    def set_window(self, w, h):
        self.surface = pygame.display.set_mode((w, h))

    def open_settings(self):
        self.screen = SettingsScreen(self, self.settings)

    def start_game(self):
        self.screen = GameScreen(self, self.settings)

    def open_tree(self, game):
        self.screen = TreeScreen(self, game)

    def back_to_game(self, game):
        game.resume()
        self.screen = game

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.screen.handle(event)
            self.screen.update()
            self.screen.draw(self.surface, pygame.mouse.get_pos())
            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    App().run()
