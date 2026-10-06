# PEAS, environment properties and agent type

Every statement below is backed by a fact in the code (file in brackets), so it can be defended in the viva.

## Problem statement
A group of crewmates must finish tasks spread over a graph of rooms while one hidden impostor kills them
one at a time. Rounds alternate between a task/movement phase and a discussion/voting phase. The crew wins by
ejecting the impostor or by finishing all tasks; the impostor wins when the crew can no longer outvote it
(`game_loop.check_wins`, `Game.meeting_phase`).

## Agents
| agent | goal | knows |
|---|---|---|
| Crewmate (n = 7 by default) | finish tasks, identify the impostor | its own track, what it saw (with forgetting), public testimony, a belief `P_i(x)` over who is the impostor |
| Impostor (1) | reduce crew to <= 1 without being ejected | its own role, **true positions of all crewmates** (`impostor_ai.py` header), what the crew has seen of it |

## PEAS

### Crewmate
| | |
|---|---|
| **Performance measure** | crew wins (impostor ejected / all tasks done); secondary: tasks finished, innocents wrongly ejected (`stats["innocents_ejected"]`), belief quality (Brier score, `experiments.belief_quality_study`) |
| **Environment** | weighted room graph, corridors with costs, 10 rooms (`ship_map.default_map`), tasks assigned per crewmate, other agents, bodies |
| **Actuators** | move along a corridor, work on a task, report a body, speak (alibi claims, sightings), vote or skip |
| **Sensors** | the other agents in its own room at the end of a tick (each sighting kept with probability `memory_prob`), bodies in its room, vent hops it sees, public testimony in meetings |

### Impostor
| | |
|---|---|
| **Performance measure** | win (crew <= 1); secondary: kills, avoiding ejection |
| **Environment** | same graph **plus vents** (impostor-only edges with cost 1) |
| **Actuators** | move (corridor or vent), kill when alone with exactly one crewmate, fake tasks, lie in testimony, plant a false sighting, vote |
| **Sensors** | positions of all crewmates (information-asymmetry assumption), who has recently seen it (`ImpostorBrain.seen_by`) |

## Environment properties
| property | value | justification (code fact) |
|---|---|---|
| Observability | **partially observable** (for the crew) | a crewmate only sees agents in its own room (`Game.observe`); roles and other rooms are hidden; the belief state `Agent.belief` exists *because* of this |
| Deterministic / stochastic | **stochastic** (deterministic given a seed) | random task assignment, random per-tick agent order (`rng.shuffle` in `task_phase`), forgetting (`memory_prob`), random gullibility; the same `(config, seed)` reproduces a game exactly (`tests.test_deterministic`) |
| Episodic / sequential | **sequential** | tasks done, kills, ejections and beliefs carry over from round to round; an early decision (kill, vote) changes all later states |
| Static / dynamic | **dynamic** | the world changes while an agent deliberates: agents move, bodies appear, crew die (A* plans are re-made when the target moves, `ImpostorBrain._follow`) |
| Discrete / continuous | **discrete** | rooms are nodes, time is ticks, edges take `ceil(cost / edge_speed)` ticks (`ShipMap.edge_ticks`); only drawing interpolates positions |
| Single / multi-agent | **multi-agent**, cooperative + competitive | crew-crew cooperation (share testimony), crew-impostor competition |
| Known / unknown | known map, hidden roles | the full map (including that vents exist) is common knowledge |

## Agent types and why
* **Crewmate = goal-based agent with an internal belief state.** Its goal (tasks, find the impostor) is explicit; it keeps
  a belief distribution because the world is only partially observable; navigation is search-based (A*). With
  `crew_policy="cautious"` its movement also depends on its belief (flee / avoid the top suspect), which makes the
  belief state drive *behaviour* and not only the vote.
* **Impostor = utility-based agent.** Its decision is "maximise a utility that trades progress against suspicion"
  (`ImpostorBrain.evaluate`). The worst-case (min over crew replies, max over own escapes) rule is a **Minimax-style**
  evaluation, not a classical zero-sum game: there is no shared board, no alternating turns and no mirror-image payoff.
* A simple reflex agent would not do: the crew must remember sightings and combine evidence over several rounds, and the
  impostor must anticipate how its actions will be interpreted.

## Cooperative or competitive?
Both. Crewmates cooperate with each other (their testimonies are pooled and corroborate each other), and compete with the
impostor, whose objective is opposed to theirs. The impostor's lies exploit the crew's cooperation (e.g. `_frame` plants a
false sighting that honest crewmates then weigh).
