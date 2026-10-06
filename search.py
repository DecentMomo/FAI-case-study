"""
search.py - A* (primary algorithm) plus the baselines it is compared against.

SEARCH PROBLEM (exactly what the viva will ask about)
-----------------------------------------------------
* STATE      : a room name (the searching agent's current location).
* INITIAL    : the agent's current room.
* ACTIONS    : "walk along corridor (u,v)"  and, for the Impostor only, "hop through
               vent (u,v)".  ShipMap.neighbors(room, allow_vents) enumerates them.
* STEP COST  : corridor traversal cost (or vent_cost).  An optional `cost_fn` can
               ADD extra cost (e.g. crowd avoidance) but must never go below the
               base cost, otherwise the heuristics below stop being admissible.
* GOAL TEST  : state == goal room.
* SEARCH TREE: root = start room; each node holds (room, g, parent).  A* keeps a
               priority queue ordered by f(n) = g(n) + h(n).

A* BOOKKEEPING
--------------
g(n) = cost of the cheapest path found so far from start to n
h(n) = heuristic estimate of the remaining cost n -> goal
f(n) = g(n) + h(n)
`expanded`  = number of nodes popped from the open list and expanded
              (the goal is counted too, so BFS and A* are counted identically)
`generated` = number of nodes pushed on the open list.

HEURISTICS (swap with Config.heuristic, or register your own)
-------------------------------------------------------------
name              admissible?  notes
zero              yes          h=0 -> A* degenerates into Dijkstra / uniform-cost
euclidean         yes          s*straight-line distance, s = tightest corridor ratio;
                               when vents are allowed also takes the min with the
                               best "go via a vent" lower bound (still admissible)
euclidean_raw     corridors    unscaled straight line; NOT admissible once vents are allowed
manhattan         NO           |dx|+|dy| can exceed the true cost -> may return suboptimal paths
weighted          NO           2.5 * euclidean: expands few nodes but loses optimality
perfect           yes          the exact distance h*: the best any heuristic can do
"""
import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ship_map import ShipMap, INF

CostFn = Callable[[str, str, float, bool], float]   # (u, v, base_cost, is_vent) -> cost >= base_cost
WEIGHTED_W = 2.5


# ----------------------------------------------------------------- results
@dataclass
class SearchResult:
    algo: str
    heuristic: str
    start: str
    goal: str
    found: bool
    path: List[str]
    steps: List[Tuple[str, float, bool]]     # (room, edge_cost, is_vent) for each hop
    cost: float
    expanded: int
    generated: int
    max_frontier: int
    micros: float
    trace: Optional[List[dict]] = None       # expansion order (only if trace=True)
    iterations: int = 0                      # IDA*: number of f-bound iterations
    parents: Optional[Dict[str, str]] = None # search-tree edges child -> parent (only if trace=True)


# -------------------------------------------------------------- heuristics
@dataclass
class Heuristic:
    name: str
    fn: Callable[[ShipMap, str, str, bool], float]
    admissible: bool
    doc: str


def _h_zero(g, n, goal, av):
    return 0.0


def _h_euclid(g, n, goal, av):
    # h(n) = s * ||pos(n) - pos(goal)||  where s = min_edge cost/euclid >= 1.
    # Admissible: any path is a chain of edges, each costing >= s * its own straight
    # length, and a chain is never shorter than the straight line (triangle inequality).
    d = g.corridor_scale * g.euclid(n, goal)
    if av and g.vent_rooms():
        # A path that uses a vent has to walk to a vent room, hop (>= vent_cost), then walk
        # from a vent room to the goal. Taking the min of both lower bounds stays admissible.
        d = min(d, g.vent_lower_bound(n, goal))
    return d


def _h_euclid_raw(g, n, goal, av):
    return g.euclid(n, goal)


def _h_manhattan(g, n, goal, av):
    (x1, y1), (x2, y2) = g.pos[n], g.pos[goal]
    return abs(x1 - x2) + abs(y1 - y2)


def _h_weighted(g, n, goal, av):
    return WEIGHTED_W * _h_euclid(g, n, goal, av)


def _h_perfect(g, n, goal, av):
    return g.dist(n, goal, av)


HEURISTICS: Dict[str, Heuristic] = {h.name: h for h in [
    Heuristic("zero", _h_zero, True, "h=0 (Dijkstra)"),
    Heuristic("euclidean", _h_euclid, True, "scaled straight-line (vent aware)"),
    Heuristic("euclidean_raw", _h_euclid_raw, True, "plain straight-line (inadmissible with vents)"),
    Heuristic("manhattan", _h_manhattan, False, "L1 distance (overestimates)"),
    Heuristic("weighted", _h_weighted, False, "2.5 x euclidean (greedy-ish)"),
    Heuristic("perfect", _h_perfect, True, "exact distance h*"),
]}


def register_heuristic(name, fn, admissible=False, doc=""):
    """Plug in your own: fn(map, node, goal, allow_vents) -> float."""
    HEURISTICS[name] = Heuristic(name, fn, admissible, doc)


# --------------------------------------------------------------- cost fns
def crowd_averse_cost(crowd: Dict[str, int], penalty: float) -> CostFn:
    """Impostor cost function: entering a room that currently holds crewmates costs
    `penalty` extra per crewmate.  Only ever ADDS cost, so admissibility is kept."""
    def fn(u, v, base, is_vent):
        return base + penalty * crowd.get(v, 0)
    return fn


def suspect_averse_cost(room: Optional[str], penalty: float) -> Optional[CostFn]:
    """Crew cost function: entering the room where my suspect was last seen costs `penalty` extra.
    Only ever ADDS cost, so the admissible heuristics stay admissible."""
    if room is None:
        return None

    def fn(u, v, base, is_vent):
        return base + (penalty if v == room else 0.0)
    return fn


def _step_cost(cost_fn, u, v, base, is_vent):
    c = cost_fn(u, v, base, is_vent) if cost_fn else base
    if c + 1e-12 < base:
        raise ValueError("cost_fn must not go below the base edge cost")
    return c


# ------------------------------------------------------------------- A*
def a_star(g: ShipMap, start: str, goal: str, heuristic: str = "euclidean",
           cost_fn: Optional[CostFn] = None, allow_vents: bool = False,
           trace: bool = False) -> SearchResult:
    t0 = time.perf_counter()
    h = HEURISTICS[heuristic].fn
    best_g = {start: 0.0}
    parent: Dict[str, Tuple[str, float, bool]] = {}
    counter = 0
    open_heap = [(h(g, start, goal, allow_vents), counter, 0.0, start)]
    expanded = 0
    generated = 1
    max_frontier = 1
    tr = [] if trace else None
    found = False
    while open_heap:
        f, _, gc, cur = heapq.heappop(open_heap)
        if gc > best_g[cur] + 1e-12:
            continue                      # stale queue entry (a cheaper route was found later)
        expanded += 1
        if tr is not None:
            tr.append({"node": cur, "g": round(gc, 3), "h": round(f - gc, 3), "f": round(f, 3),
                       "open": sorted({n for _, _, _, n in open_heap})})
        if cur == goal:
            found = True
            break
        for nb, base, is_vent in g.neighbors(cur, allow_vents):
            ng = gc + _step_cost(cost_fn, cur, nb, base, is_vent)
            if ng + 1e-12 < best_g.get(nb, INF):
                best_g[nb] = ng
                parent[nb] = (cur, base, is_vent)
                counter += 1
                generated += 1
                heapq.heappush(open_heap, (ng + h(g, nb, goal, allow_vents), counter, ng, nb))
                max_frontier = max(max_frontier, len(open_heap))
    path, steps, cost = _reconstruct(start, goal, parent, found, best_g)
    res = SearchResult("A*", heuristic, start, goal, found, path, steps, cost, expanded,
                       generated, max_frontier, (time.perf_counter() - t0) * 1e6, tr)
    if trace:
        res.parents = {k: v[0] for k, v in parent.items()}
    return res


def _reconstruct(start, goal, parent, found, best_g):
    if not found:
        return [], [], INF
    steps = []
    node = goal
    while node != start:
        p, base, is_vent = parent[node]
        steps.append((node, base, is_vent))
        node = p
    steps.reverse()
    return [start] + [s[0] for s in steps], steps, best_g[goal]


# ---------------------------------------------------------- baselines
def bfs(g: ShipMap, start: str, goal: str, cost_fn: Optional[CostFn] = None,
        allow_vents: bool = False) -> SearchResult:
    """Plain breadth-first search: ignores edge costs, returns the path with the FEWEST
    HOPS (not necessarily the cheapest).  The goal is tested when popped, so `expanded`
    is counted the same way as for A*.  The returned cost is the weighted cost of the path
    BFS found, so solution quality can be compared."""
    t0 = time.perf_counter()
    q = deque([start])
    parent: Dict[str, Tuple[str, float, bool]] = {}
    seen = {start}
    expanded = 0
    generated = 1
    max_frontier = 1
    found = False
    while q:
        cur = q.popleft()
        expanded += 1
        if cur == goal:
            found = True
            break
        for nb, base, is_vent in g.neighbors(cur, allow_vents):
            if nb not in seen:
                seen.add(nb)
                parent[nb] = (cur, base, is_vent)
                q.append(nb)
                generated += 1
                max_frontier = max(max_frontier, len(q))
    path, steps, _ = _reconstruct(start, goal, parent, found, {goal: 0})
    cost = INF
    if found:
        cost = 0.0
        node = start
        for (nxt, base, is_vent) in steps:
            cost += _step_cost(cost_fn, node, nxt, base, is_vent)
            node = nxt
    return SearchResult("BFS", "-", start, goal, found, path, steps, cost, expanded,
                        generated, max_frontier, (time.perf_counter() - t0) * 1e6)


def dijkstra(g, start, goal, cost_fn=None, allow_vents=False) -> SearchResult:
    """Uniform-cost search = A* with h = 0 (the 'no heuristic' extreme)."""
    r = a_star(g, start, goal, "zero", cost_fn, allow_vents)
    r.algo = "Dijkstra"
    return r


def compare_searches(g, start, goal, heuristic="euclidean", cost_fn=None, allow_vents=False):
    return {"astar": a_star(g, start, goal, heuristic, cost_fn, allow_vents),
            "bfs": bfs(g, start, goal, cost_fn, allow_vents),
            "dijkstra": dijkstra(g, start, goal, cost_fn, allow_vents)}


# ---------------------------------------------------------- more baselines
def ida_star(g: ShipMap, start: str, goal: str, heuristic: str = "euclidean",
             cost_fn: Optional[CostFn] = None, allow_vents: bool = False) -> SearchResult:
    """Iterative-deepening A*: depth-first search limited by an f = g + h BOUND that grows to the
    smallest f that exceeded the previous bound.  Memory is O(d) (only the current path is kept) but
    nodes are RE-EXPANDED on every iteration.  `max_frontier` reports the deepest path (memory proxy)
    and `iterations` the number of bound increases.  Optimal for an admissible heuristic."""
    t0 = time.perf_counter()
    h = HEURISTICS[heuristic].fn
    path, steps = [start], []
    stats = {"exp": 0, "gen": 1, "depth": 1, "cost": INF}

    def dfs(node, gc, bound):
        f = gc + h(g, node, goal, allow_vents)
        if f > bound + 1e-12:
            return f, False
        stats["exp"] += 1
        stats["depth"] = max(stats["depth"], len(path))
        if node == goal:
            stats["cost"] = gc
            return f, True
        nxt = INF
        for nb, base, is_vent in g.neighbors(node, allow_vents):
            if nb in path:
                continue                      # no cycles on the current path
            stats["gen"] += 1
            c = _step_cost(cost_fn, node, nb, base, is_vent)
            path.append(nb)
            steps.append((nb, base, is_vent))
            t, found = dfs(nb, gc + c, bound)
            if found:
                return t, True
            path.pop()
            steps.pop()
            nxt = min(nxt, t)
        return nxt, False

    bound = h(g, start, goal, allow_vents)
    iterations, found = 0, False
    while bound < INF:
        iterations += 1
        bound, found = dfs(start, 0.0, bound)
        if found:
            break
    return SearchResult("IDA*", heuristic, start, goal, found, list(path) if found else [],
                        list(steps) if found else [], stats["cost"] if found else INF,
                        stats["exp"], stats["gen"], stats["depth"], (time.perf_counter() - t0) * 1e6,
                        None, iterations)


def greedy_best_first(g: ShipMap, start: str, goal: str, heuristic: str = "euclidean",
                      cost_fn: Optional[CostFn] = None, allow_vents: bool = False) -> SearchResult:
    """Greedy best-first: priority = h(n) only (ignores the cost already paid).  Fast, but the path
    it returns is NOT guaranteed optimal.  Reported cost is the true weighted cost of that path."""
    t0 = time.perf_counter()
    h = HEURISTICS[heuristic].fn
    parent: Dict[str, Tuple[str, float, bool]] = {}
    gscore = {start: 0.0}
    seen = {start}
    counter = 0
    heap = [(h(g, start, goal, allow_vents), counter, start)]
    expanded, generated, max_frontier, found = 0, 1, 1, False
    closed = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            found = True
            break
        for nb, base, is_vent in g.neighbors(cur, allow_vents):
            if nb not in seen:
                seen.add(nb)
                parent[nb] = (cur, base, is_vent)
                gscore[nb] = gscore[cur] + _step_cost(cost_fn, cur, nb, base, is_vent)
                counter += 1
                generated += 1
                heapq.heappush(heap, (h(g, nb, goal, allow_vents), counter, nb))
                max_frontier = max(max_frontier, len(heap))
    path, steps, cost = _reconstruct(start, goal, parent, found, gscore)
    return SearchResult("Greedy", heuristic, start, goal, found, path, steps, cost, expanded,
                        generated, max_frontier, (time.perf_counter() - t0) * 1e6)


# -------------------------------------------------------------- logging
class SearchLog:
    """Collects one row per A* query together with the BFS / Dijkstra results on the
    SAME query -> directly usable for the node-expansion comparison chart."""
    COLUMNS = ["tick", "round", "agent", "role", "purpose", "start", "goal", "allow_vents",
               "heuristic", "astar_expanded", "astar_generated", "astar_cost", "astar_us",
               "bfs_expanded", "bfs_cost", "bfs_us", "dij_expanded", "dij_cost"]

    def __init__(self):
        self.rows: List[dict] = []

    def add(self, tick, rnd, agent, role, purpose, allow_vents, a, b=None, d=None):
        row = {"tick": tick, "round": rnd, "agent": agent, "role": role, "purpose": purpose,
               "start": a.start, "goal": a.goal, "allow_vents": allow_vents, "heuristic": a.heuristic,
               "astar_expanded": a.expanded, "astar_generated": a.generated,
               "astar_cost": round(a.cost, 3), "astar_us": round(a.micros, 1),
               "bfs_expanded": b.expanded if b else "", "bfs_cost": round(b.cost, 3) if b else "",
               "bfs_us": round(b.micros, 1) if b else "",
               "dij_expanded": d.expanded if d else "", "dij_cost": round(d.cost, 3) if d else ""}
        self.rows.append(row)

    def to_csv(self, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            w.writeheader()
            w.writerows(self.rows)

    def summary(self) -> dict:
        rows = [r for r in self.rows if r["bfs_expanded"] != ""]
        if not rows:
            return {"queries": len(self.rows)}
        n = len(rows)
        return {"queries": len(self.rows),
                "astar_expanded_mean": sum(r["astar_expanded"] for r in rows) / n,
                "bfs_expanded_mean": sum(r["bfs_expanded"] for r in rows) / n,
                "dij_expanded_mean": sum(r["dij_expanded"] for r in rows) / n,
                "astar_cost_mean": sum(r["astar_cost"] for r in rows) / n,
                "bfs_cost_mean": sum(r["bfs_cost"] for r in rows) / n}


def print_trace(res: SearchResult):
    """Human-readable expansion order of an A* run (use for the 'search tree' slide)."""
    print(f"A* {res.start} -> {res.goal}   heuristic={res.heuristic}")
    for i, t in enumerate(res.trace or [], 1):
        print(f"  {i:2d}. expand {t['node']:<11} g={t['g']:<6} h={t['h']:<6} f={t['f']:<6} open={t['open']}")
    print(f"  path={' -> '.join(res.path)}  cost={res.cost:.2f}  expanded={res.expanded}  generated={res.generated}")
