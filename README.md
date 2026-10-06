# Social Deduction Under Uncertainty
### A multi-agent simulation of task-based deception and belief-driven accusation (Among-Us style)

Crewmates walk a graph of rooms doing tasks, the hidden Impostor kills and lies, and rounds alternate
between a **Task/Movement phase** and a **Discussion/Voting phase**.

| Algorithm | Where | Role |
|---|---|---|
| **A\*** (primary) | `search.py` | every agent's path-finding; instrumented and compared with BFS / Dijkstra |
| **Minimax-style evaluation** (secondary) | `impostor_ai.py` | the Impostor's "whom to kill, when, how to escape" decision |
| Transparent belief update | `suspicion.py` | each crewmate's probability distribution over "who is the impostor" |

Nothing is scripted: every outcome (including the demo/edge cases) emerges from the rules.
The simulation is **deterministic** for a given `(config, seed)`.

---------------------------------------------------------------------------------------------------

## 1. Setup

Python 3.9+ (developed on 3.13).

```bash
pip install -r requirements.txt        # matplotlib, numpy, pillow
python tests.py                        # 12 sanity checks (A* optimality, admissibility, determinism ...)
```

## 2. How to run

```bash
python main.py run                              # one narrated game, writes logs to outputs/
python main.py run --animate                    # + live matplotlib replay (SPACE = pause)
python main.py run --animate --hide-impostor    # live demo mode: impostor/vents not revealed
python main.py run --animate --interval 900     # slower playback (ms per frame, default 600)
python main.py run --save outputs/game.gif      # export the replay instead of opening a window

# toggle smart / dumb impostor, scale agents and rooms
python main.py run --impostor minimax           # smart (default)   --depth 1|2|3 = tree depth
python main.py run --impostor greedy            # dumb baseline 1: nearest target, flees to nearest vent
python main.py run --impostor random            # dumb baseline 2: random wandering
python main.py run --crew 10 --rooms 24 --seed 4

python main.py run --heuristic zero             # swap the A* heuristic (see below)
python main.py trace --start Reactor --goal Navigation            # print the A* search tree
python main.py trace --start Reactor --goal Navigation --vents --heuristic weighted

python main.py cases                            # find + log the working / edge cases -> outputs/cases/
python main.py compare --games 200              # smart vs dumb impostor win rates
python main.py experiments [--quick]            # all complexity/comparison studies -> CSV + PNG
```

Everything can also be set from Python:

```python
from config import Config
from game_loop import run_game
r = run_game(Config(n_crew=7, n_rooms=10, seed=3, impostor_policy="minimax", minimax_depth=3))
print(r.winner, r.reason, r.stats)
```

`--rooms 10` uses the hand-built "Skeld-like" map; any other value generates a random connected map
(`ship_map.generate_map`) with loops, at least one vent-less dead end, and vents.

### Inputs
`Config` (`config.py`) is the single input: counts (`n_crew`, `n_rooms`), `seed`, task/timing numbers,
`heuristic`, `impostor_policy`, `minimax_depth`, every Minimax utility weight (`mm_*`), every suspicion weight
(`w_*`), and voting thresholds (`skip_factor`, `vote_margin`, `tie_epsilon`, `agg_floor`).

### Outputs
| file | content |
|---|---|
| `outputs/game_log.txt` | narrated game (kills, alibi conflicts, votes, ejections) |
| `outputs/movement_log.csv` | every room-to-room move with tick timestamp (`depart`/`arrive`/`vent`/`task_done`/`kill`) |
| `outputs/search_log.csv` | **one row per A\* call**: nodes expanded/generated + path cost for A\*, **and** for BFS and Dijkstra on the same query |
| `outputs/round_times.csv` | per-round computation time, #agents, #rooms, branching factor |
| `outputs/cases/*.txt` | the demo / edge-case logs (each states the seed to replay it) |
| `outputs/*_study.csv/png` | charts and tables for the complexity & comparison sections |

---------------------------------------------------------------------------------------------------

## 3. Files

| file | purpose |
|---|---|
| `config.py` | all tunables (weights, thresholds, scale) |
| `ship_map.py` | environment: weighted room graph, corridors, **vents (impostor only)**, dead ends, distance tables, 10-room map + random map generator |
| `search.py` | **A\*** with pluggable heuristics and cost function, BFS and Dijkstra baselines, instrumentation (`SearchLog`), search-tree trace |
| `agents.py` | agent state (location, plan, tasks, per-round memory, belief) |
| `impostor_ai.py` | Impostor brain: **Minimax-style evaluation**, greedy/random fall-backs, alibi lying, false-sighting "framing", voting |
| `suspicion.py` | crew belief update rules, alibi-conflict detection, voting and tie handling |
| `game_loop.py` | engine: task phase → discovery → meeting → resolution, win conditions, logging, frames |
| `visualize.py` | matplotlib replay: graph, moving agents, suspicion heat-map, **chat box** (meeting discussion + chatter while walking) and event log, both scrollable (mouse wheel over the box; `F` = back to live; `SPACE` = pause) |
| `scenarios.py` | seed sweep that **finds** the required working / edge cases and writes their logs |
| `experiments.py` | heuristic study, scaling study, smart-vs-dumb study (CSV + PNG) |
| `main.py` | CLI |
| `tests.py` | unit tests |

---------------------------------------------------------------------------------------------------

## 4. The algorithms (viva notes)

### 4.1 A\* — search problem
* **State**: a room. **Initial**: the agent's room. **Goal test**: `room == goal room`.
* **Actions**: walk a corridor `(u,v)`; the Impostor may also hop a vent (`ShipMap.neighbors(room, allow_vents)`).
* **Step cost**: corridor cost (≥ straight-line distance) or `vent_cost`. A swappable `cost_fn` may only *add* to it
  (e.g. `crowd_averse_cost`: the smart Impostor pays extra for rooms containing crew) so admissibility survives.
* **Search tree**: root = start room; node = (room, g, parent); open list ordered by `f = g + h`.
  Re-opening is supported, so it stays correct even with an inconsistent heuristic.
* `python main.py trace ...` prints the exact expansion order with `g, h, f` and the open list.

**Heuristics** (`Config.heuristic`, `search.HEURISTICS`, add yours with `register_heuristic`):

| name | admissible | idea |
|---|---|---|
| `zero` | yes | h = 0 → Dijkstra |
| `euclidean` (default) | yes (+consistent) | `s·‖pos(n) − pos(goal)‖`, `s` = tightest `cost/straight-line` ratio over all corridors. With vents allowed it is `min(direct bound, via-vent bound)`, still admissible |
| `euclidean_raw` | corridors only | plain straight line (inadmissible once vents exist: demonstrates why the vent-aware bound is needed) |
| `manhattan` | **no** | `|dx|+|dy|` can overestimate |
| `weighted` | **no** | `2.5 × euclidean`: fewer expansions but can return non-optimal paths |
| `perfect` | yes | exact distance h\* — the best any heuristic can do (lower bound on expansions) |

Admissibility/consistency and "A\* cost == Dijkstra cost" are verified in `tests.py` on three maps.

**Instrumentation.** Every A\* call is logged with the BFS and Dijkstra result for the *same* query
(`search_log.csv`): `astar_expanded, astar_cost, bfs_expanded, bfs_cost, dij_expanded, ...`.
"Expanded" = nodes popped from the open list (the goal included) for all three algorithms.

**Complexity.** Worst case A\* is `O(b^d)` time and space (`b` = branching factor = average room degree,
`d` = number of corridors in the solution); with a good heuristic the effective branching factor drops.
BFS is `O(b^d)` too but ignores costs (fewest *hops*, not cheapest path). On a graph with `V` nodes and `E`
edges A\* with a binary heap is bounded by `O((V+E) log V)` (each node expanded once when `h` is consistent).
Measured (see `outputs/heuristic_study.*`, `outputs/scaling_study.*`): on the 10-room map A\* expands ≈ 3.3 nodes
per query versus 6.0 for BFS/Dijkstra, and the advantage persists as the map grows (≈ 33 % fewer than BFS at 50 rooms in
the scaling study). BFS also returns a costlier path on ≈ 7 % of the pairs.
Honest caveat: on a map this small the absolute differences are small, and adding corridors to a fixed
number of rooms *reduces* the path length `d` as well, so measured expansions do not grow like `b^d`.

### 4.2 Minimax-**style** evaluation (Impostor) — and why it is *not* classical Minimax
**This is a worst-case decision rule over a suspicion utility, not a two-player zero-sum board game.**
There are no alternating turns on a shared board and the crew is not one rational opponent with the opposite
payoff. What is borrowed from Minimax is "choose the option whose *worst plausible reaction* is least bad".

```
ply 1  MAX  Impostor : KILL(target t)              (or LAY_LOW, utility = -urgency * crew task progress)
ply 2  MIN  Crewmates: reply r  in {PROXIMITY_ACCUSE, ALIBI_CROSSCHECK, BYSTANDER_TESTIMONY, IGNORE}
                       (the reply that yields the lowest utility for the impostor)
ply 3  MAX  Impostor : escape f in {VENT, WALK, BLEND}  (the lowest suspicion)
leaf   U(t,r,f) = progress(t) - mm_lambda * suspicion(t,r,f) - mm_travel_w * approach_cost(t)
value(t) = min_r max_f U(t,r,f)       kill the argmax target iff value > U(LAY_LOW)
```
`progress = 1/(alive_crew−1)`; `suspicion = w_caught·p_caught + reply term − blend bonus` where
`p_caught = logistic((T_escape − T_discovery)/τ)` (chance somebody walks into the kill room before the impostor
has left), `T_discovery` = ticks for the nearest other crewmate to reach the room, `T_escape` = ticks for the
escape (A\*-style distance to nearest vent + hop, etc.). Reply terms use proximity, how many crew recently saw the
impostor (they can refute a fake alibi) and how many are within 2 hops.
`minimax_depth` 3 = full tree, 2 = escape fixed to WALK, 1 = no adversary. The full tree of every decision is stored
in `result.minimax_log`. The same machinery runs when the impostor is already alone with a victim (kill or hold fire).

**Toggle** `--impostor minimax | greedy | random`. Smart impostor additionally: never vents in front of (or into) a
room with crew, picks fake alibis with a small DP (feasible travel, empty rooms, far from the body), plants one
false sighting on an innocent, and votes with the crew majority (bandwagon).

*Information assumption:* the impostor knows where every crewmate is (e.g. via cameras/admin table); crewmates
only know what they saw.

### 4.3 Crew belief / suspicion update (all weights in `Config`)
Per crewmate `i`, `P_i(x)` over all other living agents (uniform start). After each meeting, with evidence built **only
from public testimony** (claims "I was in room R at tick t" and sightings "I saw X in R at t"):

```
s_i(x) = (1-d)·ln P_i(x) + d·ln(1/n)
         + g_i · [ w_prox·prox(x) + w_scene·scene(x) + A_i(x) + F_i(x) + w_vent·vent(x) − w_group·group(x) ]
P_i'(x) = softmax_x s_i(x)          # = prior × exp(weights): naive-Bayes style, fully transparent
after the vote:  s_i(x) += w_dissent   for x who voted against the eventual majority
```
| rule | feature |
|---|---|
| proximity to a reported body | `prox(x)` = max over the window *(victim last seen alive, body found]* of `1/(1+hops(pos_x(t), body_room))`; `scene(x)` = at the body when found (the finder itself is excluded — it is there *because* it found the body) |
| inconsistent alibi (sharp) | `A_i(x)`: conflicts of type *seen_elsewhere* (claims X, seen in Y), *absent* ("I was there, I didn't see you", weight × `absent_factor`), *impossible travel*, *self-inconsistent report*. Blame goes to the claimant if the sighting is corroborated, to the reporter if the claim is corroborated, otherwise it is **split in proportion to the current beliefs**. Scaled by `trust(reporter) = 1 − P_i(reporter)`; capped by `alibi_cap` |
| voting against the majority | `w_dissent` (slight) |
| last seen with a missing player | `last_with(x)`: players missing from the meeting are known dead (roster check, announced in chat/log on its own frame). For each, everyone placed with them at their last sighting gets `1/#companions`; term `w_last_seen·min(1, last_with)` |
| seen in a group, no kill | `group(x)` reduces suspicion; needs sightings by *someone else* in a room of ≥ 3 |
| uncertainty / disagreement | per-crewmate gullibility `g_i ~ N(1, 0.25)` and imperfect memory (`memory_prob`): a forgotten sighting looks like a missing alibi |

**Voting.** Each crewmate votes for its argmax suspect, or SKIPs if its top posterior < `skip_factor/n`. Ejection
requires a plurality lead of ≥ `vote_margin` votes and no near-tie in aggregated (mean) suspicion
(`tie_epsilon`, `agg_floor`); otherwise **no elimination** (`tie`, `margin`, `near_tie`, `skip_plurality`, …).

### 4.4 Win conditions
Crew: impostor voted out **or** every living crewmate finished its tasks (no ghosts: dead crew's tasks no longer count).
Impostor: living crew ≤ 1 (the crew can no longer outvote it). `max_rounds` → "timeout".

---------------------------------------------------------------------------------------------------

## 5. Required demo cases (`python main.py cases`)
Found by sweeping seeds and testing predicates on the *recorded events* (`scenarios.py`); each log in
`outputs/cases/` prints the seed and the exact replay command, the evidence, the per-term suspicion breakdown,
the full event log and (for the smart impostor) its Minimax decisions.

| # | case | type |
|---|---|---|
| 1 | Impostor identified because its alibi was contradicted (seen elsewhere) — dumb impostor | working |
| 1b | **Smart** impostor still caught by an impossible-travel alibi (vent jump) | working |
| 2 | Impostor wins through unwitnessed vent escapes + a planted false sighting | working |
| 3 | An innocent is voted out because of a coincidental bad alibi (a forgotten sighting / false testimony) | edge |
| 4 | Tied vote → nobody ejected | edge |
| 5 | Impostor kills in the vent-less dead end `Comms`, is still standing there when the body is found | edge |
| 6 | A crewmate is **missing** at the meeting (body never found); the vote is driven by who was last seen with them | edge |

Typical outputs (seed numbers can change if you change any config value): `python main.py run --seed 5 --impostor greedy`,
`--seed 4 --impostor minimax`, … — see `outputs/cases/summary.json`.

## 6. Experiments (`python main.py experiments`, ~2 min; `--quick` ≈ 10 s)
* `heuristic_study` — A\* with each heuristic vs BFS/Dijkstra on all room pairs of the 10-room map (with and without
  vents) and on generated 20/40-room maps: expansions, cost ratio vs optimal, % optimal paths.
* `scaling_study` — rooms 10…50, agents 5…15, branching factor 1.9…4.8: nodes expanded (A\*/BFS/Dijkstra),
  computation per tick (task phase) and per meeting. *Timings include running the BFS+Dijkstra baselines next to every A\* call
  (`--no-baselines` disables that) and are wall-clock, so they vary a little between machines.*
* `policy_study` — smart (depth 1/2/3), greedy, random impostor, 200 games each, Wilson 95 % CIs.
  Reference run (200 games): random 0.5 %, greedy 1 %, minimax 25.5 / 31 / 40.5 % impostor wins for depth 1/2/3 (with the missing-player rule, deeper search now pays off because the tree models "was I seen with the victim?").
  The dumb impostors are caught almost every time; most of the smart impostor's advantage comes from
  unwitnessed venting, unrefutable alibis and misdirection, and depth 1 vs 3 differs by about 15 points; the 95 % intervals still overlap for neighbouring depths, so quote them.

## 7. Modelling assumptions / limitations (be ready to state these)
* Discrete ticks; agents see each other only inside a room at the end of a tick; the viewer is omniscient.
* The impostor has perfect position information; crewmates have only their own sightings plus public testimony.
* Crewmates always tell the truth; the impostor lies about its alibi and may plant **one** false sighting per round.
* Memory noise (`memory_prob`) deliberately produces some *false* alibi conflicts for innocents — that is the source of
  the "wrongly ejected" edge case. Set `memory_prob=1.0` to remove it.
* Weights are hand-set and tuned for a sensible balance (not learned); `Config` exposes all of them.
* An unreported kill is handled by a roster check at the next meeting: a player who was present at round start but is absent is treated as dead (as in Among Us) and removed from beliefs on a labelled step. Unreported bodies stay on the map and can be found in later rounds.
* Bodies are reported only by crewmates; the impostor never self-reports; roles are not revealed on ejection.
