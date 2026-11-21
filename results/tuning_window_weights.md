# Tuning check: weight of 1- and 2-disc windows (W1, W2)

Sample tree (report/logs/tree_alphabeta_k3.txt) showed the agent letting the
human take a point in exchange for many 2-disc windows (+0.8). To check that
W2 = 0.10 is not too high, lower values were played against the final v2
(98 games each, all 49 openings x both colours, alpha-beta).

| Variant vs v2 (W1=0.02, W2=0.10) | K | W / D / L | Score % | Avg margin | Elo [95% CI] |
|---|---|---|---|---|---|
| W2=0.05 | 4 | 25 / 38 / 35 | 44.9 | -0.21 | -36 [-91, +18] |
| W2=0.07 | 4 | 35 / 29 / 34 | 50.5 | +0.12 | +4 [-55, +62] |
| W2=0.05, W1=0.01 | 4 | 15 / 33 / 50 | 32.1 | -0.78 | -130 [-192, -75] |
| W2=0.05 | 5 | 21 / 37 / 40 | 40.3 | -0.19 | -68 [-125, -14] |
| W2=0.07 | 5 | 28 / 43 / 27 | 50.5 | +0.01 | +4 [-49, +56] |
| W2=0.05, W1=0.01 | 5 | 16 / 37 / 45 | 35.2 | -0.58 | -106 [-165, -53] |

Conclusion: lowering the window weights makes the agent weaker or changes
nothing, so W1 = 0.02 and W2 = 0.10 are kept. In this variant building many
live windows early really does pay off later in points.
