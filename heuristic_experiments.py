"""
heuristic_experiments.py - proves which heuristic is better.

The agent always picks the move with the best heuristic-backed value, so the
only fair way to compare heuristics is to make them PLAY each other with
everything else equal.

Method
------
* Both agents use the same search (alpha-beta, same depth K, centre-first
  move ordering). Only the evaluation function differs.
* Alpha-beta is deterministic: two identical games would repeat forever. So
  every game starts from one of the 49 possible 2-move openings (first mover
  plays column a, second plays column b), and every opening is played TWICE
  with the two agents swapping who moves first.  -> 98 games per match.
* Every match is run at an even and an odd K (search depth parity changes
  results in Connect 4).
* Reported per match: wins / draws / losses of agent A, score % =
  (W + D/2) / N, average final point margin (A's fours - B's fours), and the
  Elo difference with a 95% confidence interval.
* Stochastic mode (--stochastic): every disc slips like in the real
  Expectiminimax game (0.6 / 0.2 / 0.2), both agents use expectiminimax, and
  each opening is replayed with fixed random seeds.

This file contains its OWN small search (no tree building, no printing) so
the experiment is fast and does not depend on the GUI agent.

Usage
-----
    python heuristic_experiments.py                  # ladder + ablation, K=4,5
    python heuristic_experiments.py --k 4 --quick    # 7 openings only (fast check)
    python heuristic_experiments.py --stochastic --k 2
    python heuristic_experiments.py --only ladder

Outputs (results/):  heuristic_matches.csv / .md
                     (heuristic_matches_stochastic.* with --stochastic)
"""

import argparse
import csv
import math
import os
import random
import time
from multiprocessing import Pool

from board import Board, HUMAN, AI
import heuristic as H

EVALS = {
    "v2": lambda b: H.evaluate_v2(b, H.V2_MINIMAX),
    "v2-expecti-weights": lambda b: H.evaluate_v2(b, H.V2_EXPECTI),
    "v1": H.evaluate_v1,
    "score-diff (placeholder)": H.evaluate_score_diff,
    "galli (classic)": H.evaluate_galli,
    "big-constants (typical student)": H.evaluate_big_constants,
}
for _name, _w in H.ABLATIONS.items():
    EVALS[_name] = (lambda w: (lambda b: H.evaluate_v2(b, w)))(_w)


# ---------------------------------------------------------------------------
# Search used by the experiment
# ---------------------------------------------------------------------------
def order(board):
    """Centre-first column order (the best pruning order in Connect 4)."""
    mid = (board.cols - 1) / 2
    return sorted(board.valid_moves(), key=lambda c: abs(c - mid))


def alphabeta(board, depth, maximizing, alpha, beta, ev):
    """Value from the computer's (AI) point of view."""
    if depth == 0 or board.is_full():
        return ev(board)
    piece = AI if maximizing else HUMAN
    if maximizing:
        best = -math.inf
        for c in order(board):
            board.drop(c, piece)
            best = max(best, alphabeta(board, depth - 1, False, alpha, beta, ev))
            board.undo(c)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    best = math.inf
    for c in order(board):
        board.drop(c, piece)
        best = min(best, alphabeta(board, depth - 1, True, alpha, beta, ev))
        board.undo(c)
        beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def expecti(board, depth, maximizing, ev):
    """Expectiminimax: MAX/MIN -> CHANCE (0.6/0.2/0.2) -> ..."""
    if depth == 0 or board.is_full():
        return ev(board)
    piece = AI if maximizing else HUMAN
    best = -math.inf if maximizing else math.inf
    for c in order(board):
        exp = 0.0
        for land, p in board.chance_outcomes(c):
            board.drop(land, piece)
            exp += p * expecti(board, depth - 1, not maximizing, ev)
            board.undo(land)
        best = max(best, exp) if maximizing else min(best, exp)
    return best


def search_move(board, piece, depth, ev, stochastic):
    """Best column for `piece` (the eval is from AI's view, so HUMAN minimises)."""
    maximizing = piece == AI
    best_col, best_val = None, None
    for c in order(board):
        if stochastic:
            v = 0.0
            for land, p in board.chance_outcomes(c):
                board.drop(land, piece)
                v += p * expecti(board, depth - 1, not maximizing, ev)
                board.undo(land)
        else:
            board.drop(c, piece)
            v = alphabeta(board, depth - 1, not maximizing, -math.inf, math.inf, ev)
            board.undo(c)
        if best_val is None or (v > best_val if maximizing else v < best_val):
            best_col, best_val = c, v
    return best_col


# ---------------------------------------------------------------------------
# Agents that do not search
# ---------------------------------------------------------------------------
def random_move(board, piece, rng):
    return rng.choice(board.valid_moves())


def greedy_move(board, piece, rng):
    """Take the column that scores most now; else block the opponent's best;
    else play near the centre."""
    opp = HUMAN if piece == AI else AI

    def gain(col, who):
        row = board.drop(col, who)
        g = board.fours_through(row, col)
        board.undo(col)
        return g

    moves = order(board)
    mine = max(moves, key=lambda c: gain(c, piece))
    if gain(mine, piece) > 0:
        return mine
    theirs = max(moves, key=lambda c: gain(c, opp))
    if gain(theirs, opp) > 0:
        return theirs
    return moves[0] if rng.random() < 0.5 else rng.choice(moves)


SUFFIX = ""        # "_stochastic" in --stochastic mode (separate result files)

SIMPLE = {"random": random_move, "greedy (take point / block)": greedy_move}


# ---------------------------------------------------------------------------
# Games and matches
# ---------------------------------------------------------------------------
def play_game(a_name, b_name, a_first, opening, k, stochastic, seed, rows, cols):
    """Returns (A's fours - B's fours) at the end of the game."""
    rng = random.Random(seed)
    board = Board(rows, cols)
    first_piece = HUMAN                     # first mover always plays HUMAN
    board.first_player = first_piece
    a_piece = first_piece if a_first else AI
    names = {a_piece: a_name, (AI if a_piece == HUMAN else HUMAN): b_name}

    piece = first_piece
    plies = 0
    while not board.is_full():
        if plies < len(opening):
            aimed = opening[plies]
            if not board.is_valid_move(aimed):
                aimed = order(board)[0]
        else:
            name = names[piece]
            if name in SIMPLE:
                aimed = SIMPLE[name](board, piece, rng)
            else:
                aimed = search_move(board, piece, k, EVALS[name], stochastic)
        col = board.resolve_stochastic(aimed, rng) if stochastic else aimed
        board.drop(col, piece)
        piece = AI if piece == HUMAN else HUMAN
        plies += 1
    human, ai = board.scores()
    return (human - ai) if a_piece == HUMAN else (ai - human)


def _play(args):
    return play_game(*args)


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def summarize(a, b, k, margins, stochastic, seconds):
    n = len(margins)
    w = sum(m > 0 for m in margins)
    d = sum(m == 0 for m in margins)
    l = n - w - d
    pts = [1.0 if m > 0 else 0.5 if m == 0 else 0.0 for m in margins]
    score = sum(pts) / n
    sd = math.sqrt(sum((p - score) ** 2 for p in pts) / max(1, n - 1))
    half = 1.96 * sd / math.sqrt(n)
    lo, hi = elo(max(score - half, 0.0)), elo(min(score + half, 1.0))
    return {"agent_a": a, "agent_b": b, "k": k,
            "mode": "stochastic" if stochastic else "deterministic",
            "games": n, "wins": w, "draws": d, "losses": l,
            "score_pct": round(100 * score, 1),
            "avg_margin": round(sum(margins) / n, 2),
            "elo": round(elo(score)), "elo_lo": round(lo), "elo_hi": round(hi),
            "seconds": round(seconds, 1)}


def match(a, b, k, stochastic, openings, pool, rows, cols, repeats):
    jobs = []
    for i, op in enumerate(openings):
        for rep in range(repeats):
            seed = 1000 * i + rep
            jobs.append((a, b, True, op, k, stochastic, seed, rows, cols))
            jobs.append((a, b, False, op, k, stochastic, seed, rows, cols))
    t = time.perf_counter()
    margins = pool.map(_play, jobs)
    res = summarize(a, b, k, margins, stochastic, time.perf_counter() - t)
    print(f"  {a:>24} vs {b:<32} K={k}  W/D/L {res['wins']:>3}/{res['draws']:>2}/"
          f"{res['losses']:<3} score {res['score_pct']:5.1f}%  margin "
          f"{res['avg_margin']:+5.2f}  Elo {res['elo']:+5d} "
          f"[{res['elo_lo']:+d}, {res['elo_hi']:+d}]  ({res['seconds']}s)",
          flush=True)
    return res


def save(results, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"heuristic_matches{SUFFIX}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    lines = ["| Agent A | Agent B | Mode | K | Games | W / D / L (A) | Score % (A) "
             "| Avg margin (A) | Elo diff [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['agent_a']} | {r['agent_b']} | {r['mode']} | {r['k']} | "
                     f"{r['games']} | {r['wins']} / {r['draws']} / {r['losses']} | "
                     f"{r['score_pct']} | {r['avg_margin']:+.2f} | "
                     f"{r['elo']:+d} [{r['elo_lo']:+d}, {r['elo_hi']:+d}] |")
    with open(os.path.join(out, f"heuristic_matches{SUFFIX}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Connect 4 heuristic tournament")
    p.add_argument("--k", nargs="+", type=int, default=[4, 5])
    p.add_argument("--only", choices=["ladder", "ablation", "all"], default="all")
    p.add_argument("--stochastic", action="store_true",
                   help="random slips + expectiminimax (use small K, e.g. 2 or 3)")
    p.add_argument("--quick", action="store_true", help="7 openings instead of 49")
    p.add_argument("--repeats", type=int, default=None,
                   help="games per opening and colour (default 1, stochastic 2)")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results")
    a = p.parse_args()
    global SUFFIX
    SUFFIX = "_stochastic" if a.stochastic else ""

    cols = a.cols
    openings = [(x, y) for x in range(cols) for y in range(cols)]
    if a.quick:
        openings = [(x, x) for x in range(cols)]
    repeats = a.repeats or (2 if a.stochastic else 1)

    if a.stochastic:
        ladder = [("v2-expecti-weights", "random"),
                  ("v2-expecti-weights", "greedy (take point / block)"),
                  ("v2-expecti-weights", "big-constants (typical student)"),
                  ("v2-expecti-weights", "galli (classic)"),
                  ("v2-expecti-weights", "v1"),
                  ("v2-expecti-weights", "v2")]
        ablation = []
    else:
        ladder = [("v2", "random"),
                  ("v2", "greedy (take point / block)"),
                  ("v2", "score-diff (placeholder)"),
                  ("v2", "galli (classic)"),
                  ("v2", "big-constants (typical student)"),
                  ("v2", "v1")]
        ablation = [("v2", name) for name in H.ABLATIONS]
    pairs = (ladder if a.only in ("ladder", "all") else []) + \
            (ablation if a.only in ("ablation", "all") else [])

    print(f"{len(openings)} openings x 2 colours x {repeats} = "
          f"{2 * len(openings) * repeats} games per match, "
          f"{a.rows}x{cols} board, {'stochastic' if a.stochastic else 'deterministic'}")
    results = []
    with Pool(a.workers) as pool:
        for k in a.k:
            print(f"\nK = {k}")
            for x, y in pairs:
                results.append(match(x, y, k, a.stochastic, openings, pool,
                                     a.rows, cols, repeats))
                save(results, a.out)
    print(f"\nSaved {os.path.join(a.out, f"heuristic_matches{SUFFIX}.md")}")


if __name__ == "__main__":
    main()
