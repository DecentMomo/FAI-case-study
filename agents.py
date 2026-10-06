"""
agents.py - agent state.  (Behaviour: crewmates in game_loop.py, impostor in impostor_ai.py.)

Agent type: GOAL-BASED agents with an internal (belief) state.
  * Crewmate: goal = finish tasks + find the impostor.  Keeps a probability
    distribution over who the impostor is (a *belief state*), updated by the rules
    in suspicion.py.  Navigation is search-based (A*).
  * Impostor: goal = reduce the crew to <= 1 member without being voted out.
    Utility-based: scores kill options with a Minimax-style utility (impostor_ai.py).
Cooperative (crew <-> crew) and competitive (crew <-> impostor) at the same time.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PALETTE = [("Red", "#d62728"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"), ("Yellow", "#e6b800"),
           ("Pink", "#e377c2"), ("Orange", "#ff7f0e"), ("Cyan", "#17becf"), ("Purple", "#9467bd"),
           ("Brown", "#8c564b"), ("Lime", "#8fce00"), ("Gray", "#7f7f7f"), ("Teal", "#008080"),
           ("Navy", "#223a8f"), ("Maroon", "#800000"), ("Olive", "#808000"), ("Coral", "#ff6f61")]


@dataclass
class Task:
    room: str
    duration: int
    progress: int = 0
    done: bool = False


class Agent:
    def __init__(self, name: str, color: str, role: str):
        self.name = name
        self.color = color
        self.role = role                    # "crew" | "impostor"
        self.alive = True
        # --- location: either in a room, or travelling along an edge ---
        self.room: Optional[str] = None     # None while in a corridor / vent
        self.edge: Optional[Tuple[str, str, bool, int]] = None   # (src, dst, is_vent, total_ticks)
        self.remaining = 0                  # ticks left on the current edge
        # --- navigation ---
        self.plan: List[Tuple[str, float, bool]] = []   # remaining steps (room, cost, is_vent)
        self.goal: Optional[str] = None
        # --- tasks (crew only; the impostor 'fakes' tasks) ---
        self.tasks: List[Task] = []
        # --- per-round memory (cleared after every meeting) ---
        self.track: Dict[int, str] = {}                  # tick -> room I was in (certain knowledge)
        self.seen: List[Tuple[int, str, str]] = []       # (tick, room, other) sightings I remember
        self.vent_seen: List[Tuple[int, str, str]] = [] # (tick, room, who) saw someone vent
        # --- beliefs (crew only) ---
        self.belief: Dict[str, float] = {}               # P(x is the impostor), sums to 1
        self.gullibility = 1.0                           # scale on evidence weights
        # --- rendering ---
        self.xy = (0.0, 0.0)

    # convenience --------------------------------------------------------
    @property
    def loc(self) -> str:
        """Room the agent is in, or the room it is heading to if travelling."""
        return self.room if self.room is not None else self.edge[1]

    def current_task(self) -> Optional[Task]:
        for t in self.tasks:
            if not t.done:
                return t
        return None

    def tasks_done(self) -> int:
        return sum(t.done for t in self.tasks)

    def clear_round_memory(self):
        self.track = {}
        self.seen = []
        self.vent_seen = []
        self.plan = []
        self.goal = None
