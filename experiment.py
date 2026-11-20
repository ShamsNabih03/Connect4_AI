"""
experiment.py - board-size experiment (extra work for the report).

Question: how do time and nodes expanded grow with the BOARD SIZE, for each
algorithm and depth K? A wider board means a bigger branching factor b, and
the tree has about b^K nodes (b^(K/2) in the best case for alpha-beta), so
this shows the effect of b directly.

Method
------
* For every board size, generate N random mid-game positions (a few random
  opening moves, fixed seed), so every algorithm is timed on the SAME boards.
* For every K and algorithm, call ai_interface.get_move() on each position
  and record wall-clock time and info["nodes"]; report the averages.
* Tree printing is switched off (ai_interface.QUIET = True, and stdout is
  silenced) so console output does not distort the timings.
* If one configuration averages more than --max-seconds per move, larger K
  are skipped for that algorithm and board size.

Usage
-----
    python experiment.py
    python experiment.py --sizes 6x7 7x8 8x9 9x10 --k 1 2 3 4 5 --positions 5

Outputs (in --out, default "results/"):
    board_size.csv          every measurement
    board_size.md           the same as a Markdown table for the report
    board_size_nodes.png    nodes expanded vs board size (log scale)
    board_size_time.png     time per move vs board size (log scale)

Run it again after the real agent is plugged into ai_interface.py - the
placeholder agent caps the depth at 3, so K > 3 shows nothing new with it.
"""

import argparse
import contextlib
import csv
import os
import random
import statistics
import time

import ai_interface
from board import Board, HUMAN, AI

ALGO_NAMES = {"minimax": "Minimax", "alphabeta": "Alpha-Beta",
              "expectiminimax": "Expectiminimax"}


def parse_size(text):
    rows, cols = text.lower().split("x")
    return int(rows), int(cols)


def make_positions(rows, cols, count, opening_moves, seed):
    """Random mid-game boards (same seed -> same boards every run)."""
    rng = random.Random(seed)
    boards = []
    for _ in range(count):
        b = Board(rows, cols)
        piece = HUMAN
        for _ in range(opening_moves):
            b.drop(rng.choice(b.valid_moves()), piece)
            piece = AI if piece == HUMAN else HUMAN
        boards.append(b)
    return boards


def measure(board, algorithm, k):
    """One AI move: returns (seconds, nodes). Console output is discarded."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        start = time.perf_counter()
        _, info = ai_interface.get_move(board.copy(), algorithm, k)
        elapsed = time.perf_counter() - start
    return elapsed, (info or {}).get("nodes", 0)


def run(sizes, ks, algorithms, positions, opening, max_seconds, seed):
    ai_interface.QUIET = True
    results = []
    for rows, cols in sizes:
        boards = make_positions(rows, cols, positions, opening, seed)
        for algorithm in algorithms:
            for k in ks:
                times, nodes = [], []
                for b in boards:
                    t, n = measure(b, algorithm, k)
                    times.append(t)
                    nodes.append(n)
                row = {"rows": rows, "cols": cols, "size": f"{rows}x{cols}",
                       "k": k, "algorithm": algorithm,
                       "avg_time": statistics.mean(times),
                       "max_time": max(times),
                       "avg_nodes": statistics.mean(nodes)}
                results.append(row)
                print(f"{row['size']:>6}  {ALGO_NAMES[algorithm]:<15} K={k:<2}  "
                      f"time {row['avg_time']:9.4f} s   nodes {row['avg_nodes']:12,.0f}")
                if row["avg_time"] > max_seconds:
                    print(f"        (over {max_seconds}s - skipping larger K)")
                    break
    return results


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


def save_markdown(results, path):
    lines = ["| Board | K | Algorithm | Avg time (s) | Max time (s) | Avg nodes expanded |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['size']} | {r['k']} | {ALGO_NAMES[r['algorithm']]} | "
                     f"{r['avg_time']:.4f} | {r['max_time']:.4f} | {r['avg_nodes']:,.0f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_charts(results, out_dir, algorithms, ks):
    """One panel per algorithm, one line per K, x = board size, log y."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping charts")
        return
    sizes = list(dict.fromkeys(r["size"] for r in results))
    for metric, label, fname in [("avg_nodes", "Nodes expanded", "board_size_nodes.png"),
                                 ("avg_time", "Time per move (s)", "board_size_time.png")]:
        fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 4),
                                 sharey=True, squeeze=False)
        for ax, algorithm in zip(axes[0], algorithms):
            for k in ks:
                pts = [(sizes.index(r["size"]), r[metric]) for r in results
                       if r["algorithm"] == algorithm and r["k"] == k and r[metric] > 0]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                            label=f"K={k}")
            ax.set_title(ALGO_NAMES[algorithm])
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            ax.set_xlabel("Board (rows x columns)")
            ax.set_yscale("log")
            ax.grid(True, which="major", alpha=0.3)
            ax.legend(fontsize=8)
        axes[0][0].set_ylabel(label)
        fig.suptitle(f"{label} vs board size")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Connect 4 board-size experiment")
    p.add_argument("--sizes", nargs="+", default=["6x7", "7x8", "8x9", "9x10"],
                   help="board sizes as ROWSxCOLS (min 6x7)")
    p.add_argument("--k", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--algorithms", nargs="+", default=list(ALGO_NAMES),
                   choices=list(ALGO_NAMES))
    p.add_argument("--positions", type=int, default=5,
                   help="random positions averaged per configuration")
    p.add_argument("--opening", type=int, default=6,
                   help="random moves played to create each position")
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    sizes = [parse_size(s) for s in a.sizes]
    cap = getattr(ai_interface, "PLACEHOLDER_MAX_DEPTH", None)
    if cap is not None and max(a.k) > cap:
        print(f"NOTE: the placeholder agent caps depth at {cap}; "
              "plug in the real agent for meaningful K > 3 results.\n")

    results = run(sizes, a.k, a.algorithms, a.positions, a.opening,
                  a.max_seconds, a.seed)
    os.makedirs(a.out, exist_ok=True)
    save_csv(results, os.path.join(a.out, "board_size.csv"))
    save_markdown(results, os.path.join(a.out, "board_size.md"))
    save_charts(results, a.out, a.algorithms, a.k)
    print(f"\nSaved results to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
