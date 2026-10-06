"""
config.py - every tunable knob of the simulation lives here.

All suspicion-update weights and all Minimax-style utility weights are plain
numbers on this dataclass so that the maths in the report can quote them
directly and so that experiments can sweep them.
"""
from dataclasses import dataclass, asdict, replace
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ scale
    n_crew: int = 7              # crewmates (plus exactly 1 impostor => n_crew+1 agents)
    n_rooms: int = 10            # 10 -> hand-built Skeld-like map; otherwise a generated map
    seed: int = 0                # RNG seed: same seed + same config => identical game
    map_seed: Optional[int] = None   # seed for generated maps (None -> use `seed`)
    map_extra_edges: float = 0.35    # generated maps: extra loops as a fraction of n_rooms (raises branching factor)

    # ------------------------------------------------------- tasks and timing
    tasks_per_crew: int = 8      # tasks assigned to each crewmate (random rooms)
    task_duration: int = 3       # ticks a crewmate must stay in the room to finish a task
    round_ticks: int = 30        # a task phase ends after this many ticks if nobody finds a body
    max_rounds: int = 15         # safety cap -> outcome "timeout"
    kill_cooldown: int = 8       # ticks between two kills
    initial_cooldown: int = 5    # cooldown at the start of every round
    edge_speed: float = 1.0      # corridor cost units travelled per tick (ticks = ceil(cost/speed))
    vent_cost: float = 1.0       # cost of one vent hop (1 tick)

    # ----------------------------------------------------------- crew behaviour
    crew_policy: str = "tasks"   # "tasks" = just do tasks | "cautious" = belief-aware movement
    avoid_threshold: float = 0.35    # cautious: a crewmate avoids its top suspect once P(suspect) >= this
    suspect_penalty: float = 3.0     # cautious: extra A* cost for entering the room the suspect was last seen in
    suspect_memory: int = 6          # cautious: ticks a last-seen position stays usable
    alarm_belief: float = 0.9        # cautious: belief set on someone seen using a vent (private, immediate)

    # ----------------------------------------------------------------- search
    heuristic: str = "euclidean"         # key of search.HEURISTICS used by every A* call
    compare_baselines: bool = True       # also run BFS + Dijkstra on every A* query and log them
    crowd_penalty: float = 1.5           # smart impostor: extra cost for entering a room with crew

    # --------------------------------------------------------------- impostor
    impostor_policy: str = "minimax"     # "minimax" (smart) | "greedy" | "random"
    minimax_depth: int = 3               # 1 = greedy utility, 2 = + crew reply, 3 = + impostor escape
    # Minimax-style utility  U = progress - lambda*suspicion - travel_w*approach_cost
    mm_lambda: float = 0.2               # how much the impostor dislikes suspicion
    mm_travel_w: float = 0.01            # penalty per unit of approach cost
    mm_lay_low_value: float = 0.0        # utility of "do not kill now"; kill only if value is higher
    mm_urgency: float = 1.0              # LAY_LOW loses this much utility as crew task progress -> 100%
    mm_tau: float = 2.0                  # logistic temperature (ticks) of P(caught at the body)
    mm_w_caught: float = 5.0             # suspicion if the impostor is found standing at the body
    mm_w_prox_acc: float = 0.5           # crew reply "accuse whoever was nearest the body"
    mm_w_alibi_chk: float = 0.6          # crew reply "cross-check alibis" (scaled by witnesses of impostor)
    mm_w_witness: float = 0.6            # crew reply "bystanders testify" (crew near the kill room)
    mm_w_blend_bonus: float = 0.25       # suspicion reduction for walking into a group
    mm_w_last_seen: float = 0.5          # crew reply 'who was last seen with the victim?'
    mm_trail_window: int = 10            # ticks: crew that saw the impostor this recently are "witnesses"
    mm_vent_wait: int = 3                # extra ticks if a crewmate stands in the vent room
    alibi_escape_window: int = 6         # ticks after a kill / vent hop whose positions the impostor lies about
    alibi_lookback: int = 8              # ticks BEFORE a kill / vent hop it also lies about (room to build a feasible story)
    imp_lie: bool = True                 # ablation switch: smart impostor lies about its alibi
    imp_frame: bool = True               # ablation switch: plants a false sighting
    imp_vent_safe: bool = True           # ablation switch: never vents in front of / into a watched room
    imp_bandwagon: bool = True           # ablation switch: votes with the crew majority
    frame_probability: float = 1.0       # smart impostor: chance to plant a false sighting on a crewmate

    # -------------------------------------------- crew belief / suspicion rules
    # log-score added to suspect x (see suspicion.py for the exact formula)
    w_prox: float = 2.0          # proximity to the reported body (value in [0,1])
    w_scene: float = 2.5         # being seen at the body when it was found
    w_alibi: float = 2.5         # per refuted-alibi conflict ("sharp" increase)
    w_false: float = 2.0         # reporter whose accusation is refuted by a corroborated alibi
    w_last_seen: float = 1.5     # last seen with a player who is missing from the meeting
    w_vent: float = 6.0          # seen using a vent
    w_group: float = 1.0         # (negative) seen in a group -> alibi corroboration
    w_dissent: float = 0.4       # voted against the eventual majority (slight)
    alibi_cap: float = 6.0       # upper bound on total alibi penalty per suspect per meeting
    absent_factor: float = 0.5   # an "I was there and did not see you" claim counts this much
    group_full_ticks: int = 3    # group presence ticks needed for the full group credit
    prox_unknown: float = 0.25   # proximity assigned to a suspect nobody can place in the window
    trust_slope: float = 1.0     # trust(reporter) = 1 - slope * P(reporter is impostor)
    belief_decay: float = 0.05   # shrink old beliefs towards uniform each meeting
    memory_prob: float = 0.97    # chance a crewmate remembers one sighting (noise -> uncertainty)
    gullibility_sd: float = 0.25 # per-crewmate scale on evidence weights ~ N(1, sd)

    # ----------------------------------------------------------------- voting
    skip_factor: float = 1.25    # vote skip if top posterior < skip_factor / (#suspects)
    vote_margin: int = 1         # leader must beat runner-up by at least this many votes
    tie_epsilon: float = 0.01    # near-tie on aggregated suspicion (with a lead < 2 votes) -> no elimination
    agg_floor: float = 0.0       # leader's aggregated suspicion must reach this value

    # ---------------------------------------------------------------- outputs
    record_frames: bool = False  # keep per-tick frames for the animation
    verbose: bool = False        # print the narrative while the game runs

    @classmethod
    def from_json(cls, path: str, **overrides) -> "Config":
        """Load a preset: a JSON object whose keys are Config field names (unknown keys are an error)."""
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = set(cls.__dataclass_fields__)
        bad = sorted(set(data) - known)
        if bad:
            raise ValueError(f"unknown Config field(s) in {path}: {bad}")
        data.update(overrides)
        return cls(**data)

    def copy(self, **kw) -> "Config":
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)
