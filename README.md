# Connect4_AI

Human vs. computer Connect 4 (pygame GUI) with three AI options: Minimax,
Minimax with alpha-beta pruning, and Expectiminimax (0.6 / 0.2 / 0.2 slip rule).
The search is cut off after K levels and leaves are scored by a heuristic that
estimates the final score difference. The game runs until the board is full;
the player with more connected-fours wins.

## Run

    pip install pygame-ce
    python gui.py

Choose board size (>= 6x7), depth K, algorithm and first player, then play.
The agent prints its search tree and heuristic values to the console on every
turn: the whole tree if it has at most 3,000 nodes, otherwise a summary (top `PRINT_DEPTH`
levels plus the best line). Every turn's tree is also saved to `trees/turn_NN.txt` (complete up to
100,000 nodes; limits are constants at the top of `ai_interface.py`). The "Search tree" button opens the GUI tree viewer.
Expectiminimax is slow above K = 3 on a 6x7 board.

## Tests and experiments

    python -m unittest test_board test_heuristic test_tree_output     # 46 tests
    python experiment.py --k 1 2 3 4 5 6 --positions 3 --max-seconds 15
                                                     # exact command behind the report's tables (several minutes;
                                                     # writes results/board_size.*; defaults are 5 positions, 20 s)
    python heuristic_experiments.py                  # heuristic tournaments
    python report/make_sample_runs.py                # sample-run screenshots and logs
    python report/make_tree_samples.py               # sample trees for the report

## Files

- `gui.py`, `ui.py` - GUI
- `board.py` - board, gravity, scoring, random drops
- `ai_interface.py` - Minimax / Alpha-Beta / Expectiminimax agent
- `heuristic.py` - heuristic evaluation
- `tree.py`, `tree_printer.py`, `tree_view.py` - tree structure, console printer, GUI viewer
- `report/Connect4_Report.pdf` - the report; `results/` - experiment output
