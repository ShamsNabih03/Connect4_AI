"""
make_sample_runs.py - plays scripted sample games through the real GUI
(headless) and saves screenshots + console logs for the report.

Run from the project folder:  python report/make_sample_runs.py
Outputs go to report/images/ and report/logs/.
The human's moves are random (fixed seed) so the run is reproducible.
"""

import contextlib
import io
import os
import random
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pygame  # noqa: E402
import gui  # noqa: E402
from board import HUMAN, AI  # noqa: E402

IMG = os.path.join(HERE, "images")
LOG = os.path.join(HERE, "logs")
os.makedirs(IMG, exist_ok=True)
os.makedirs(LOG, exist_ok=True)


def frame(app, mouse=(5, 5)):
    app.screen.update()
    app.screen.draw(app.surface, mouse)


def click(app, pos):
    app.screen.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))


def button(app, label):
    b = next(b for b in app.screen.buttons if b.label == label)
    click(app, b.rect.center)


def save(app, name):
    pygame.image.save(app.surface, os.path.join(IMG, name))


def play_game(app, rows, cols, algo, k, first, seed, tag, snap_after=None,
              snap_slip=False, open_tree_at=None):
    rng = random.Random(seed)
    random.seed(seed)          # the Expectiminimax slips use the global RNG -> reproducible
    s = app.settings
    s.rows, s.cols, s.algorithm, s.k, s.first = rows, cols, algo, k, first
    app.open_settings()
    frame(app)
    save(app, f"{tag}_settings.png")
    button(app, "Start game")
    g = app.screen
    snapped = slip_snapped = tree_done = False
    while not g.game_over:
        frame(app)
        if not g._can_human_move():
            continue
        moves = g.board.move_count
        if snap_after and not snapped and moves >= snap_after:
            save(app, f"{tag}_midgame.png")
            snapped = True
        if snap_slip and not slip_snapped and "slipped" in g.message:
            save(app, f"{tag}_slip.png")
            slip_snapped = True
        if open_tree_at and not tree_done and len(g.history) >= open_tree_at:
            button(app, "Search tree")
            t = app.screen
            button(app, "Best path")
            frame(app)
            best = [n for n in t.nodes if id(n) in t.best_ids]
            frame(app, t.node_rect(best[-1]).center)
            save(app, f"{tag}_tree.png")
            button(app, "< Back")
            tree_done = True
        col = rng.choice(g.board.valid_moves())
        click(app, g._cell_center(1, col))
    frame(app)
    save(app, f"{tag}_gameover.png")
    return g


def main():
    app = gui.App()
    runs = [
        ("run1_alphabeta", dict(rows=6, cols=7, algo="alphabeta", k=3, first=HUMAN,
                                seed=7, snap_after=14, open_tree_at=3)),
        ("run2_expecti", dict(rows=7, cols=8, algo="expectiminimax", k=2, first=AI,
                              seed=11, snap_slip=True, open_tree_at=3)),
    ]
    for tag, kw in runs:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g = play_game(app, tag=tag, **kw)
        with open(os.path.join(LOG, f"{tag}.txt"), "w") as f:
            f.write(buf.getvalue())
        print(tag, "final score (you, computer):", g.board.scores())


if __name__ == "__main__":
    main()
