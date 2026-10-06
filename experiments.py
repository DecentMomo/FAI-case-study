"""
experiments.py - the numbers and charts for the Complexity / Comparison sections of the report.

  heuristic_study : A* with every heuristic vs BFS vs Dijkstra on the SAME (start, goal) pairs
  scaling_study   : nodes expanded + per-tick / per-meeting computation time as rooms, agents and
                    branching factor grow
  policy_study    : smart (Minimax-style, depth 1/2/3) vs greedy vs random impostor outcomes

Every study writes a CSV and a PNG into outputs/.   Run:  python main.py experiments [--quick]
"""
import csv
import itertools
import math
import os
import random
import time
from collections import defaultdict

from config import Config
from game_loop import run_game
from search import HEURISTICS, a_star, bfs, dijkstra
from ship_map import default_map, generate_map


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ======================================================================
def heuristic_study(outdir="outputs", pairs_per_map=400, sizes=(10, 20, 40), seed=0):
    rng = random.Random(seed)
    rows = []
    maps = [("skeld10", default_map())] + [(f"gen{n}", generate_map(n, seed=n)) for n in sizes if n != 10]
    for mname, g in maps:
        for av in (False, True):
            allpairs = [(a, b) for a in g.rooms for b in g.rooms if a != b]
            pairs = allpairs if len(allpairs) <= pairs_per_map else rng.sample(allpairs, pairs_per_map)
            agg = defaultdict(lambda: {"exp": 0, "gen": 0, "cost_ratio": 0.0, "optimal": 0, "us": 0.0})
            for a, b in pairs:
                opt = dijkstra(g, a, b, None, av)
                for hname in HEURISTICS:
                    r = a_star(g, a, b, hname, None, av)
                    x = agg[hname]
                    x["exp"] += r.expanded
                    x["gen"] += r.generated
                    x["cost_ratio"] += r.cost / opt.cost if opt.cost > 0 else 1.0
                    x["optimal"] += r.cost <= opt.cost + 1e-9
                    x["us"] += r.micros
                r = bfs(g, a, b, None, av)
                x = agg["BFS"]
                x["exp"] += r.expanded
                x["gen"] += r.generated
                x["cost_ratio"] += r.cost / opt.cost if opt.cost > 0 else 1.0
                x["optimal"] += r.cost <= opt.cost + 1e-9
                x["us"] += r.micros
            n = len(pairs)
            for hname, x in agg.items():
                adm = HEURISTICS[hname].admissible if hname in HEURISTICS else ""
                if hname == "euclidean_raw" and av:
                    adm = False            # not admissible once vents exist
                rows.append({"map": mname, "rooms": len(g.rooms), "branching": round(g.avg_branching(), 2),
                             "vents_allowed": av, "algorithm": hname, "admissible": adm, "pairs": n,
                             "mean_expanded": round(x["exp"] / n, 3), "mean_generated": round(x["gen"] / n, 3),
                             "mean_cost_ratio": round(x["cost_ratio"] / n, 4),
                             "optimal_fraction": round(x["optimal"] / n, 3), "mean_us": round(x["us"] / n, 1)})
    _write_csv(os.path.join(outdir, "heuristic_study.csv"), rows)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, av, ttl in zip(axes, (False, True), ("Impostor-free graph (corridors only)", "Impostor graph (corridors + vents)")):
        sub = [r for r in rows if r["map"] == "skeld10" and r["vents_allowed"] == av]
        order = ["BFS", "zero", "euclidean_raw", "euclidean", "manhattan", "weighted", "perfect"]
        sub = sorted(sub, key=lambda r: order.index(r["algorithm"]))
        xs = range(len(sub))
        cols = ["#d62728" if (r["optimal_fraction"] < 0.999) else "#1f77b4" for r in sub]
        ax.bar(xs, [r["mean_expanded"] for r in sub], color=cols)
        for i, r in enumerate(sub):
            ax.text(i, r["mean_expanded"] + 0.1, f"{r['mean_expanded']:.1f}\n{r['optimal_fraction']*100:.0f}% opt",
                    ha="center", fontsize=7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["algorithm"] for r in sub], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mean nodes expanded per query")
        ax.set_title(ttl + "\n(10-room map, all room pairs; red = sometimes suboptimal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heuristic_study.png"), dpi=130)
    plt.close(fig)

    # expansions vs graph size
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for alg, col in (("BFS", "#7f7f7f"), ("zero", "#ff7f0e"), ("euclidean", "#1f77b4"), ("perfect", "#2ca02c")):
        pts = sorted((r["rooms"], r["mean_expanded"]) for r in rows if r["algorithm"] == alg and not r["vents_allowed"])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=("Dijkstra (h=0)" if alg == "zero" else alg), color=col)
    ax.set_xlabel("rooms")
    ax.set_ylabel("mean nodes expanded")
    ax.set_title("Search effort vs map size (corridor graph)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "expansions_vs_size.png"), dpi=130)
    plt.close(fig)
    return rows


# ======================================================================
def _run_cfg(cfg):
    t0 = time.perf_counter()
    r = run_game(cfg)
    wall = time.perf_counter() - t0
    sm = r.search_log.summary()
    ticks = max(1, sum(x["ticks"] for x in r.round_times))
    return {"winner": r.winner, "rounds": r.rounds, "ticks": r.ticks, "wall_s": wall,
            "task_s": sum(x["task_s"] for x in r.round_times), "meeting_s": sum(x["meeting_s"] for x in r.round_times),
            "per_tick_ms": 1000 * sum(x["task_s"] for x in r.round_times) / ticks,
            "per_meeting_ms": 1000 * sum(x["meeting_s"] for x in r.round_times) / max(1, len(r.meetings)),
            "per_round_ms": 1000 * sum(x["total_s"] for x in r.round_times) / max(1, r.rounds),
            "queries": sm.get("queries", 0), "astar_exp": sm.get("astar_expanded_mean", float("nan")),
            "bfs_exp": sm.get("bfs_expanded_mean", float("nan")), "dij_exp": sm.get("dij_expanded_mean", float("nan")),
            "astar_cost": sm.get("astar_cost_mean", float("nan")), "bfs_cost": sm.get("bfs_cost_mean", float("nan")),
            "branching": r.map.avg_branching(), "rooms": len(r.map.rooms), "agents": cfg.n_crew + 1}


def scaling_study(outdir="outputs", trials=4, rooms_list=(10, 16, 24, 36, 50), crew_list=(4, 7, 10, 14),
                  extra_list=(0.0, 0.35, 0.8, 1.5), base=None):
    base = base or Config()
    rows = []

    def sweep(kind, values, make):
        for v in values:
            res = [_run_cfg(make(v, t)) for t in range(trials)]
            row = {"sweep": kind, "value": v, "trials": trials}
            for k in ("rooms", "agents", "branching", "rounds", "ticks", "per_tick_ms", "per_meeting_ms", "per_round_ms",
                      "queries", "astar_exp", "bfs_exp", "dij_exp", "astar_cost", "bfs_cost"):
                row[k] = round(_mean(x[k] for x in res), 4)
            row["impostor_win_rate"] = round(_mean(1.0 if x["winner"] == "impostor" else 0.0 for x in res), 3)
            rows.append(row)
            print(f"  {kind}={v}: rooms={row['rooms']:.0f} agents={row['agents']:.0f} b={row['branching']:.2f} "
                  f"A*={row['astar_exp']:.2f} BFS={row['bfs_exp']:.2f} per-tick={row['per_tick_ms']:.2f}ms "
                  f"per-meeting={row['per_meeting_ms']:.2f}ms")

    print("sweep rooms (7 crew)")
    sweep("rooms", rooms_list, lambda v, t: base.copy(n_rooms=v, n_crew=7, seed=100 + t, compare_baselines=True))
    print("sweep agents (10 rooms)")
    sweep("agents", crew_list, lambda v, t: base.copy(n_rooms=10, n_crew=v, seed=200 + t, compare_baselines=True))
    print("sweep branching factor (24 rooms, 7 crew)")
    sweep("branching", extra_list, lambda v, t: base.copy(n_rooms=24, n_crew=7, map_extra_edges=v, map_seed=7,
                                                          seed=300 + t, compare_baselines=True))
    _write_csv(os.path.join(outdir, "scaling_study.csv"), rows)

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    r1 = [r for r in rows if r["sweep"] == "rooms"]
    ax = axes[0]
    ax.plot([r["rooms"] for r in r1], [r["bfs_exp"] for r in r1], "o-", label="BFS", color="#7f7f7f")
    ax.plot([r["rooms"] for r in r1], [r["dij_exp"] for r in r1], "s-", label="Dijkstra (h=0)", color="#ff7f0e")
    ax.plot([r["rooms"] for r in r1], [r["astar_exp"] for r in r1], "^-", label="A* (euclidean)", color="#1f77b4")
    ax.set_xlabel("rooms")
    ax.set_ylabel("mean nodes expanded per in-game query")
    ax.set_title("Search effort vs map size")
    ax.legend()
    r2 = [r for r in rows if r["sweep"] == "agents"]
    ax = axes[1]
    ax.plot([r["agents"] for r in r2], [r["per_tick_ms"] for r in r2], "o-", label="task phase (ms / tick)")
    ax.plot([r["agents"] for r in r2], [r["per_meeting_ms"] for r in r2], "s-", label="meeting (ms / meeting)")
    ax.set_xlabel("agents")
    ax.set_ylabel("computation time (ms)")
    ax.set_title("Computation vs number of agents (10 rooms)")
    ax.legend()
    r3 = [r for r in rows if r["sweep"] == "branching"]
    ax = axes[2]
    ax.plot([r["branching"] for r in r3], [r["bfs_exp"] for r in r3], "o-", label="BFS", color="#7f7f7f")
    ax.plot([r["branching"] for r in r3], [r["dij_exp"] for r in r3], "s-", label="Dijkstra (h=0)", color="#ff7f0e")
    ax.plot([r["branching"] for r in r3], [r["astar_exp"] for r in r3], "^-", label="A*", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot([r["branching"] for r in r3], [r["per_tick_ms"] for r in r3], "d--", color="#d62728", label="ms / tick")
    ax2.set_ylabel("ms per tick", color="#d62728")
    ax.set_xlabel("average branching factor b (24 rooms)")
    ax.set_ylabel("nodes expanded")
    ax.set_title("Effect of connectivity")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scaling_study.png"), dpi=130)
    plt.close(fig)
    return rows


# ======================================================================
def _wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def policy_study(outdir="outputs", games=200, base=None):
    base = base or Config()
    variants = [("random", base.copy(impostor_policy="random")),
                ("greedy", base.copy(impostor_policy="greedy")),
                ("minimax d=1", base.copy(impostor_policy="minimax", minimax_depth=1)),
                ("minimax d=2", base.copy(impostor_policy="minimax", minimax_depth=2)),
                ("minimax d=3", base.copy(impostor_policy="minimax", minimax_depth=3))]
    rows = []
    for name, cfg in variants:
        wins = rounds = kills = innocents = 0
        for s in range(games):
            r = run_game(cfg.copy(seed=s, compare_baselines=False))
            wins += r.winner == "impostor"
            rounds += r.rounds
            kills += r.stats.get("kills", 0)
            innocents += r.stats.get("innocents_ejected", 0)
        lo, hi = _wilson(wins, games)
        rows.append({"policy": name, "games": games, "impostor_wins": wins,
                     "impostor_win_rate": round(wins / games, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "mean_rounds": round(rounds / games, 2), "mean_kills": round(kills / games, 2),
                     "mean_innocents_ejected": round(innocents / games, 2)})
        print(f"  {name:<12} impostor wins {wins}/{games} = {wins / games:.1%}  (95% CI {lo:.1%}-{hi:.1%})  "
              f"kills/game={kills / games:.2f}")
    _write_csv(os.path.join(outdir, "policy_study.csv"), rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = range(len(rows))
    ax.bar(xs, [r["impostor_win_rate"] for r in rows], color=["#7f7f7f", "#ff7f0e", "#9ecae1", "#6baed6", "#1f77b4"],
           yerr=[[r["impostor_win_rate"] - r["ci_low"] for r in rows], [r["ci_high"] - r["impostor_win_rate"] for r in rows]],
           capsize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["policy"] for r in rows])
    ax.set_ylabel("impostor win rate")
    ax.set_title(f"Smart vs dumb impostor ({games} games each, 95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "policy_study.png"), dpi=130)
    plt.close(fig)
    return rows


def run_all(outdir="outputs", quick=False):
    os.makedirs(outdir, exist_ok=True)
    print("== heuristic study ==")
    heuristic_study(outdir, pairs_per_map=150 if quick else 400, sizes=(10, 20) if quick else (10, 20, 40))
    print("== scaling study ==")
    if quick:
        scaling_study(outdir, trials=2, rooms_list=(10, 20, 30), crew_list=(4, 7, 10), extra_list=(0.0, 0.6, 1.5))
    else:
        scaling_study(outdir, trials=4)
    print("== policy study ==")
    policy_study(outdir, games=60 if quick else 200)
    print(f"\nCSV + PNG files written to {outdir}/")


if __name__ == "__main__":
    run_all()
