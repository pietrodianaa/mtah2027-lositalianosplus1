# Edoardo — log

| Date | Issue / PR | Change | dead_F1 (P/R) | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|
| 2026-11-11 | #10 / #20 | v1: ball > 1 m past a touchline/byline; dead while the ball stays missing (max 40 s, same half); drop runs < 2 s | 0.0 → 0.4144 (0.9316 / 0.2688) | 1 | yes | football reasoning: whole ball over the line + ~0.5 m tracking error; restart times (Opta); gain far above the ~0.04 noise |

## Ideas queue
- Ball still: exclude the keeper holding the ball (ball stays with him; 2024-25 keepers held it long, especially when leading); keep a free kick in the box (ball placed, taker steps away). --tune on A-League (fewest fouls) will under-reward foul stoppages.
- Precision check of MAX_FILL_S: try 20-60 s with --tune, keep a flat stretch.
