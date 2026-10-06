"""
ship_map.py - the environment: a weighted, undirected graph of rooms.

* nodes  = rooms (each has an (x, y) position used for drawing AND for the
           straight-line heuristic)
* edges  = corridors with a traversal cost >= the Euclidean distance between the
           two rooms (this is what makes the straight-line heuristic admissible)
* vents  = a second, Impostor-only set of edges with a small fixed cost
* dead-end rooms = rooms with exactly one corridor; the map marks which of them
           have no vent (the "trap" rooms for the Impostor)

One tick of simulation time = `edge_speed` cost units of walking, so an edge of
cost c takes ceil(c / edge_speed) ticks, and a vent hop takes 1 tick.
"""
import math
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

INF = float("inf")


def _ceil3(x: float) -> float:
    """Round UP to 3 decimals so a corridor can never be shorter than the straight line."""
    return math.ceil(x * 1000 - 1e-9) / 1000


class ShipMap:
    def __init__(self, pos: Dict[str, Tuple[float, float]],
                 corridors: Iterable[Tuple[str, str, float]],
                 vents: Iterable[Tuple[str, str]] = (),
                 vent_cost: float = 1.0, edge_speed: float = 1.0,
                 name: str = "custom", hub: Optional[str] = None):
        self.name = name
        self.pos = dict(pos)
        self.rooms: List[str] = list(pos)
        self.vent_cost = vent_cost
        self.edge_speed = edge_speed
        self.adj: Dict[str, Dict[str, float]] = {r: {} for r in self.rooms}
        self.vent_adj: Dict[str, Set[str]] = {r: set() for r in self.rooms}
        for u, v, c in corridors:
            if c + 1e-9 < self.euclid(u, v):
                raise ValueError(f"corridor {u}-{v} cheaper than straight line: "
                                 "heuristic would not be admissible")
            self.adj[u][v] = c
            self.adj[v][u] = c
        for a, b in vents:
            self.vent_adj[a].add(b)
            self.vent_adj[b].add(a)
        self.hub = hub or max(self.rooms, key=lambda r: len(self.adj[r]))
        # tightest constant s with  s * euclid(u,v) <= cost(u,v)  over all corridors.
        # h(n) = s * euclid(n, goal) is then admissible AND consistent (triangle inequality).
        self.corridor_scale = min(c / max(self.euclid(u, v), 1e-9)
                                  for u in self.rooms for v, c in self.adj[u].items())
        self._tables: Dict[tuple, Dict[str, Dict[str, float]]] = {}
        self._vlb: Dict[tuple, float] = {}

    # ------------------------------------------------------------- structure
    def euclid(self, a: str, b: str) -> float:
        (x1, y1), (x2, y2) = self.pos[a], self.pos[b]
        return math.hypot(x1 - x2, y1 - y2)

    def neighbors(self, room: str, allow_vents: bool = False):
        """Yield (neighbour, base_cost, is_vent) - the ACTIONS available in `room`."""
        for n, c in self.adj[room].items():
            yield n, c, False
        if allow_vents:
            for n in self.vent_adj[room]:
                yield n, self.vent_cost, True

    def edge_ticks(self, cost: float, is_vent: bool = False) -> int:
        if is_vent:
            return 1
        return max(1, math.ceil(cost / self.edge_speed - 1e-9))

    def degree(self, room: str) -> int:
        return len(self.adj[room])

    def dead_ends(self) -> List[str]:
        return [r for r in self.rooms if self.degree(r) == 1]

    def vent_rooms(self) -> List[str]:
        return [r for r in self.rooms if self.vent_adj[r]]

    def trap_rooms(self) -> List[str]:
        """Dead-end rooms without a vent: an impostor killing here has to walk out."""
        return [r for r in self.dead_ends() if not self.vent_adj[r]]

    def n_corridors(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    def avg_branching(self) -> float:
        """Average room connectivity = branching factor b of the corridor graph."""
        return 2.0 * self.n_corridors() / len(self.rooms)

    # ------------------------------------------------------ distance tables
    def _floyd(self, weight, allow_vents: bool) -> Dict[str, Dict[str, float]]:
        R = self.rooms
        d = {a: {b: INF for b in R} for a in R}
        for a in R:
            d[a][a] = 0.0
            for b, c in self.adj[a].items():
                d[a][b] = min(d[a][b], weight(c, False))
            if allow_vents:
                for b in self.vent_adj[a]:
                    d[a][b] = min(d[a][b], weight(self.vent_cost, True))
        for k in R:
            dk = d[k]
            for i in R:
                dik = d[i][k]
                if dik == INF:
                    continue
                di = d[i]
                for j in R:
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
        return d

    def _table(self, kind: str, allow_vents: bool):
        key = (kind, allow_vents)
        if key not in self._tables:
            w = {"cost": lambda c, v: c,
                 "hops": lambda c, v: 1.0,
                 "ticks": lambda c, v: float(self.edge_ticks(c, v))}[kind]
            self._tables[key] = self._floyd(w, allow_vents)
        return self._tables[key]

    def dist(self, a, b, allow_vents=False) -> float:
        """Exact shortest-path cost. ORACLE: used for the 'perfect' heuristic and for
        the Impostor's *model* of how fast crew can reach a room - never for navigation."""
        return self._table("cost", allow_vents)[a][b]

    def hops(self, a, b, allow_vents=False) -> float:
        return self._table("hops", allow_vents)[a][b]

    def tick_dist(self, a, b, allow_vents=False) -> float:
        """Fewest ticks needed to get from a to b."""
        return self._table("ticks", allow_vents)[a][b]

    def vent_lower_bound(self, n: str, goal: str) -> float:
        """Admissible lower bound on any path from n to goal that uses >= 1 vent hop:
        walk n -> some vent room (>= s*euclid), take >= 1 vent hop (>= vent_cost),
        walk from some vent room -> goal (>= s*euclid).
            LB = min_vin s*d(n,vin)  +  vent_cost  +  min_vout s*d(vout,goal)
        (the two minima are independent, so no need to scan pairs).  Used by the vent-aware
        'euclidean' heuristic when the searching agent may use vents."""
        ends = self.vent_rooms()
        s = self.corridor_scale
        key_in, key_out = ("in", n), ("out", goal)
        if key_in not in self._vlb:
            self._vlb[key_in] = min(s * self.euclid(n, v) for v in ends)
        if key_out not in self._vlb:
            self._vlb[key_out] = min(s * self.euclid(v, goal) for v in ends)
        return self._vlb[key_in] + self.vent_cost + self._vlb[key_out]


# ======================================================================
#  The hand-built 10-room map (a simplified Skeld)
# ======================================================================
_DEFAULT_POS = {
    "Cafeteria":  (5.0, 3.0),   # hub, 5 corridors
    "Admin":      (7.0, 1.5),
    "Storage":    (5.0, 0.5),
    "Electrical": (3.0, 1.0),
    "Security":   (2.2, 3.0),
    "Reactor":    (0.5, 3.0),   # dead end WITH a vent
    "MedBay":     (3.5, 4.5),
    "Navigation": (8.5, 3.0),
    "Weapons":    (7.2, 4.8),
    "Comms":      (4.5, 5.8),   # dead end WITHOUT a vent  -> the trap room
}
# (room, room, multiplier >= 1 applied to the straight-line distance)
_DEFAULT_EDGES = [
    ("Cafeteria", "Admin", 1.10), ("Cafeteria", "Storage", 1.00),
    ("Cafeteria", "Security", 1.20), ("Cafeteria", "MedBay", 1.00),
    ("Cafeteria", "Weapons", 1.15), ("Admin", "Storage", 1.00),
    ("Admin", "Navigation", 1.05), ("Storage", "Electrical", 1.00),
    ("Electrical", "Security", 1.10), ("Security", "Reactor", 1.00),
    ("MedBay", "Comms", 1.00), ("Weapons", "Navigation", 1.10),
    ("MedBay", "Security", 1.05),
]
_DEFAULT_VENTS = [("Reactor", "Electrical"), ("Admin", "Weapons"), ("Storage", "Navigation")]


def default_map(vent_cost=1.0, edge_speed=1.0) -> ShipMap:
    pos = _DEFAULT_POS
    tmp = ShipMap.__new__(ShipMap)
    tmp.pos = pos
    edges = []
    for u, v, m in _DEFAULT_EDGES:
        edges.append((u, v, _ceil3(tmp.euclid(u, v) * m)))
    return ShipMap(pos, edges, _DEFAULT_VENTS, vent_cost, edge_speed,
                   name="skeld10", hub="Cafeteria")


# ======================================================================
#  Random connected maps for the scaling experiments
# ======================================================================
_THEMED = list(_DEFAULT_POS)


def generate_map(n_rooms: int, seed: int = 0, vent_cost=1.0, edge_speed=1.0,
                 extra_edge_frac: float = 0.35) -> ShipMap:
    """Random geometric map: jittered grid positions, Euclidean MST (guarantees
    connectivity), a few extra short edges (creates loops / multiple entrances),
    at least one dead-end room WITHOUT a vent, and ~n/3 vents elsewhere."""
    rng = random.Random(seed)
    cols = math.ceil(math.sqrt(n_rooms * 1.3))
    names = [(_THEMED[i] if i < len(_THEMED) else f"Room{i:02d}") for i in range(n_rooms)]
    pos = {}
    for i, nm in enumerate(names):
        pos[nm] = (round((i % cols) * 2.2 + rng.uniform(-0.6, 0.6), 3),
                   round((i // cols) * 2.0 + rng.uniform(-0.6, 0.6), 3))
    tmp = ShipMap.__new__(ShipMap)
    tmp.pos = pos
    # Prim's algorithm on the complete Euclidean graph
    in_tree = {names[0]}
    tree = []
    while len(in_tree) < n_rooms:
        best = min(((tmp.euclid(a, b), a, b) for a in in_tree for b in names if b not in in_tree))
        tree.append((best[1], best[2]))
        in_tree.add(best[2])
    deg = {n: 0 for n in names}
    for a, b in tree:
        deg[a] += 1
        deg[b] += 1
    leaves = [n for n in names if deg[n] == 1]
    cx = sum(p[0] for p in pos.values()) / n_rooms
    cy = sum(p[1] for p in pos.values()) / n_rooms
    trap = max(leaves, key=lambda n: math.hypot(pos[n][0] - cx, pos[n][1] - cy))
    edges = {frozenset(e) for e in tree}
    # extra edges: shortest non-tree pairs, never touching the protected dead end
    cand = sorted(((tmp.euclid(a, b), a, b) for i, a in enumerate(names) for b in names[i + 1:]
                   if frozenset((a, b)) not in edges and trap not in (a, b)))
    want = int(extra_edge_frac * n_rooms)
    for d, a, b in cand:
        if want <= 0:
            break
        if deg[a] < 5 and deg[b] < 5:
            edges.add(frozenset((a, b)))
            deg[a] += 1
            deg[b] += 1
            want -= 1
    corridors = []
    for e in edges:
        a, b = sorted(e)
        corridors.append((a, b, _ceil3(tmp.euclid(a, b) * rng.uniform(1.0, 1.3))))
    # vents between non-adjacent rooms, never in the trap room
    vents = []
    pool = [n for n in names if n != trap]
    used = set()
    for _ in range(max(1, n_rooms // 3)):
        for _try in range(30):
            a, b = rng.sample(pool, 2)
            if frozenset((a, b)) in edges or frozenset((a, b)) in used:
                continue
            used.add(frozenset((a, b)))
            vents.append((a, b))
            break
    return ShipMap(pos, corridors, vents, vent_cost, edge_speed, name=f"gen{n_rooms}_s{seed}")


def build_map(cfg) -> ShipMap:
    if cfg.n_rooms == 10:
        return default_map(cfg.vent_cost, cfg.edge_speed)
    return generate_map(cfg.n_rooms, cfg.map_seed if cfg.map_seed is not None else cfg.seed,
                        cfg.vent_cost, cfg.edge_speed, cfg.map_extra_edges)
