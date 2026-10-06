# Report map: required section -> evidence in this repo

Run `python main.py experiments` and `python make_figures.py` first; all outputs land in `outputs/`.

| # | Required section | What to write | Evidence (file / command / figure) |
|---|---|---|---|
| 1 | Problem domain and statement | game, agents, goals, interaction | `README.md` intro, `docs/PEAS_and_properties.md` |
| 2 | PEAS | crew and impostor tables | `docs/PEAS_and_properties.md` |
| 3 | Environment properties | partially observable, stochastic, sequential, dynamic, discrete, multi-agent | same file, each row cites code |
| 4 | Type of agent | goal-based (crew), utility-based (impostor); cooperative + competitive | same file; `agents.py` header; `impostor_ai.py` header |
| 5 | Searching technique | A*: state, actions, cost, heuristic, search tree, expansion order | `search.py` docstring; `python main.py trace`; **`outputs/figures/fig_astar_tree.png`**; `fig_map.png` (state space = the room graph); Minimax-style tree: **`fig_minimax_tree.png`** |
| 6 | Working and edge cases | 2 working + 2 edge (we have 2+1b working, 4 edge) | `python main.py cases` -> `outputs/cases/*.txt` (each states its seed + replay command); belief plot `fig_belief_evolution.png` |
| 7 | Complexity | time/space of A* in b and d; measured scaling | `outputs/scaling_study.png/csv` (rooms, agents, branching factor); `search_log.csv` per game |
| 8 | Comparison | BFS, Dijkstra, IDA*, greedy best-first, heuristic quality | `outputs/heuristic_study.png/csv` (expansions, memory, % optimal); `README.md` section 4.1 |
| + | Evidence for claims | which rules / tricks matter | `outputs/ablation_study.png`, `sensitivity_study.png`, `belief_quality.png`, `crew_policy_study.png`, `policy_study.png` |

## Facts and numbers to quote (re-generate; do not trust a stale copy)
* A* vs BFS on the 10-room map: `heuristic_study.csv`, rows `skeld10`, `vents_allowed=False`.
* IDA*: same cost as A*, much less memory (`mean_memory` = depth) but many more expansions (re-expansion).
* Greedy best-first: fewest expansions but not always optimal (`optimal_fraction` < 1).
* Smart vs dumb impostor: `policy_study.csv` (random ~0 %, greedy ~0-1 %, minimax 25-45 % depending on depth).
* What drives the impostor's edge: `ablation_study.csv`. In the reference run, removing alibi lies drops the impostor
  to ~0 % and unsafe venting to ~12 %, while the Minimax gate and tree depth change the win rate by only a few points
  (confidence intervals overlap) - say this honestly: the Minimax-style layer is a modest part of the edge.
* Crew belief quality: `belief_quality.csv` (Brier score, log-loss, top-suspect accuracy per meeting) and
  `belief_calibration.csv` (reliability curve).

## Viva Q&A cheat sheet
**Why A*?** Static weighted graph, single goal, we need the cheapest path and have an admissible heuristic (scaled straight-line
distance), so A* is optimal and expands fewer nodes than BFS/Dijkstra.
**Is the heuristic admissible / consistent?** Yes: `s = min(cost/straight-line)` over corridors, so any path costs >= s x straight
line by the triangle inequality. With vents the bound is min(direct bound, via-vent bound). `tests.py` checks both for every pair.
**What if the heuristic is inadmissible?** `manhattan` / `weighted` can return costlier paths: see `optimal_fraction` in the study.
**Why not plain BFS?** BFS minimises hops, not cost: it returns a more expensive path for ~7 % of room pairs on this map.
**What is the search tree for the impostor's decision?** 3 plies (target -> crew reply -> escape), branching ~ 7 x 4 x 3; see `fig_minimax_tree.png`.
**Is this really Minimax?** No - it borrows the worst-case decision rule over a suspicion utility. No alternating board game, no
mirror-image payoff. State this first.
**Where does uncertainty come from?** hidden roles, limited line of sight, forgetting (`memory_prob`), lies, random task assignment and order.
**Complexity of A\***: worst case O(b^d) time and space; with a consistent heuristic each node is expanded once, O((V + E) log V) with a heap.
**What happens as the map grows?** expansions grow roughly linearly with rooms on these sparse graphs (`scaling_study.png`); A* keeps a
constant-factor advantage.
**Limitations:** hand-tuned weights, impostor sees all positions, crew never lie, small map.
