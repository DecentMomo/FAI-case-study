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
from search import HEURISTICS, a_star, bfs, dijkstra, ida_star, greedy_best_first
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
#  Batch runner (optionally parallel).  Results depend only on (config, seed), so
#  jobs=1 and jobs=N give identical numbers.
# ======================================================================
CAL_BINS = 10


def outcome(cfg):
    """Run one game and keep only small, picklable summary numbers (incl. belief-quality metrics)."""
    r = run_game(cfg)
    per_meeting = []
    cal = [[0, 0.0, 0.0] for _ in range(CAL_BINS)]      # per bin: count, sum(P), sum(is_impostor)
    first_id = None
    for i, m in enumerate(r.meetings, 1):
        imp = m["impostor"]
        ps, br = [], []
        for obs, b in m["beliefs"].items():
            if imp not in b:
                continue
            ps.append(b[imp])
            br.append(sum((p - (1.0 if x == imp else 0.0)) ** 2 for x, p in b.items()))
            for x, p in b.items():
                k = min(CAL_BINS - 1, int(p * CAL_BINS))
                cal[k][0] += 1
                cal[k][1] += p
                cal[k][2] += 1.0 if x == imp else 0.0
        if not ps:
            continue
        agg = m["agg"]
        top1 = bool(agg) and max(agg, key=agg.get) == imp
        if top1 and first_id is None:
            first_id = i
        per_meeting.append({"meeting": i, "p_true": sum(ps) / len(ps), "brier": sum(br) / len(br),
                            "logloss": -math.log(max(sum(ps) / len(ps), 1e-9)), "top1": top1})
    s = r.stats
    return {"winner": r.winner, "rounds": r.rounds, "ticks": r.ticks, "kills": s.get("kills", 0),
            "innocents": s.get("innocents_ejected", 0), "no_elim": s.get("no_elimination", 0),
            "reason": r.reason, "per_meeting": per_meeting, "first_identify": first_id, "cal": cal,
            "trapped": s.get("trapped_in_dead_end", 0), "frames": s.get("frames_planted", 0),
            "flee": s.get("flee_moves", 0)}


def run_batch(cfgs, jobs=1):
    cfgs = list(cfgs)
    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(outcome, cfgs, chunksize=max(1, len(cfgs) // (jobs * 4))))
    return [outcome(c) for c in cfgs]


def _rate(res, pred):
    k = sum(1 for r in res if pred(r))
    lo, hi = _wilson(k, len(res))
    return k / len(res), lo, hi



# ======================================================================
def heuristic_study(outdir="outputs", pairs_per_map=400, sizes=(10, 20, 40), seed=0):
    rng = random.Random(seed)
    rows = []
    maps = [("skeld10", default_map())] + [(f"gen{n}", generate_map(n, seed=n)) for n in sizes if n != 10]
    for mname, g in maps:
        for av in (False, True):
            allpairs = [(a, b) for a in g.rooms for b in g.rooms if a != b]
            pairs = allpairs if len(allpairs) <= pairs_per_map else rng.sample(allpairs, pairs_per_map)
            agg = defaultdict(lambda: {"exp": 0, "gen": 0, "cost_ratio": 0.0, "optimal": 0, "us": 0.0, "mem": 0})
            for a, b in pairs:
                opt = dijkstra(g, a, b, None, av)
                runs = [(h, a_star(g, a, b, h, None, av)) for h in HEURISTICS]
                runs += [("BFS", bfs(g, a, b, None, av)),
                         ("IDA*", ida_star(g, a, b, "euclidean", None, av)),
                         ("Greedy", greedy_best_first(g, a, b, "euclidean", None, av))]
                for label, r in runs:
                    x = agg[label]
                    x["exp"] += r.expanded
                    x["gen"] += r.generated
                    x["mem"] += r.max_frontier           # A*/BFS/Greedy: peak open-list size; IDA*: max depth
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
                             "mean_expanded": round(x["exp"] / n, 3), "mean_memory": round(x["mem"] / n, 3), "mean_generated": round(x["gen"] / n, 3),
                             "mean_cost_ratio": round(x["cost_ratio"] / n, 4),
                             "optimal_fraction": round(x["optimal"] / n, 3), "mean_us": round(x["us"] / n, 1)})
    _write_csv(os.path.join(outdir, "heuristic_study.csv"), rows)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, av, ttl in zip(axes, (False, True), ("Impostor-free graph (corridors only)", "Impostor graph (corridors + vents)")):
        sub = [r for r in rows if r["map"] == "skeld10" and r["vents_allowed"] == av]
        order = ["BFS", "zero", "euclidean_raw", "euclidean", "manhattan", "weighted", "perfect", "IDA*", "Greedy"]
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


def policy_study(outdir="outputs", games=200, base=None, jobs=1):
    base = base or Config()
    variants = [("random", base.copy(impostor_policy="random")),
                ("greedy", base.copy(impostor_policy="greedy")),
                ("minimax d=1", base.copy(impostor_policy="minimax", minimax_depth=1)),
                ("minimax d=2", base.copy(impostor_policy="minimax", minimax_depth=2)),
                ("minimax d=3", base.copy(impostor_policy="minimax", minimax_depth=3))]
    rows = []
    for name, cfg in variants:
        res = run_batch([cfg.copy(seed=s, compare_baselines=False) for s in range(games)], jobs)
        wins = sum(r["winner"] == "impostor" for r in res)
        lo, hi = _wilson(wins, games)
        rows.append({"policy": name, "games": games, "impostor_wins": wins,
                     "impostor_win_rate": round(wins / games, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "mean_rounds": round(_mean(r["rounds"] for r in res), 2),
                     "mean_kills": round(_mean(r["kills"] for r in res), 2),
                     "mean_innocents_ejected": round(_mean(r["innocents"] for r in res), 2)})
        print(f"  {name:<12} impostor wins {wins}/{games} = {wins / games:.1%}  (95% CI {lo:.1%}-{hi:.1%})  "
              f"kills/game={rows[-1]['mean_kills']:.2f}")
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


# ======================================================================
def belief_quality_study(outdir="outputs", games=200, base=None, jobs=1):
    """How good are the crew's beliefs?  Per meeting: mean P(true impostor), Brier score, log-loss,
    top-suspect accuracy, time-to-identify, and a reliability (calibration) curve."""
    base = base or Config()
    rows, cal_rows = [], []
    fig, axes = _plt().subplots(1, 3, figsize=(15, 4.4))
    for pol, col in (("minimax", "#1f77b4"), ("greedy", "#ff7f0e")):
        res = run_batch([base.copy(seed=s, impostor_policy=pol, compare_baselines=False) for s in range(games)], jobs)
        by = defaultdict(list)
        for r in res:
            for m in r["per_meeting"]:
                by[m["meeting"]].append(m)
        for k in sorted(by):
            if len(by[k]) < 5:
                continue
            rows.append({"policy": pol, "meeting": k, "n_games": len(by[k]),
                         "mean_p_true_impostor": round(_mean(x["p_true"] for x in by[k]), 4),
                         "mean_brier": round(_mean(x["brier"] for x in by[k]), 4),
                         "mean_logloss": round(_mean(x["logloss"] for x in by[k]), 4),
                         "top1_accuracy": round(_mean(1.0 if x["top1"] else 0.0 for x in by[k]), 3)})
        sub = [r_ for r_ in rows if r_["policy"] == pol]
        axes[0].plot([r_["meeting"] for r_ in sub], [r_["mean_p_true_impostor"] for r_ in sub], "o-", label=pol, color=col)
        axes[1].plot([r_["meeting"] for r_ in sub], [r_["top1_accuracy"] for r_ in sub], "s-", label=pol, color=col)
        tot = [[0, 0.0, 0.0] for _ in range(CAL_BINS)]
        for r in res:
            for i, (n, sp, si) in enumerate(r["cal"]):
                tot[i][0] += n
                tot[i][1] += sp
                tot[i][2] += si
        xs = [t[1] / t[0] for t in tot if t[0] > 20]
        ys = [t[2] / t[0] for t in tot if t[0] > 20]
        for i, t in enumerate(tot):
            cal_rows.append({"policy": pol, "bin": i, "count": t[0], "mean_predicted": round(t[1] / t[0], 4) if t[0] else "",
                             "observed_frequency": round(t[2] / t[0], 4) if t[0] else ""})
        axes[2].plot(xs, ys, "o-", label=pol, color=col)
        ident = [r["first_identify"] for r in res if r["first_identify"]]
        print(f"  {pol}: top suspect was the impostor at some meeting in {len(ident)}/{games} games; "
              f"mean first-identify meeting = {_mean(ident):.2f}")
    axes[0].set_title("Mean belief in the true impostor")
    axes[0].set_xlabel("meeting #")
    axes[1].set_title("Top aggregated suspect = impostor")
    axes[1].set_xlabel("meeting #")
    axes[1].set_ylim(0, 1.02)
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_title("Reliability: predicted P vs observed frequency")
    axes[2].set_xlabel("predicted P(x is impostor)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "belief_quality.png"), dpi=130)
    plt_ = _plt()
    plt_.close(fig)
    _write_csv(os.path.join(outdir, "belief_quality.csv"), rows)
    _write_csv(os.path.join(outdir, "belief_calibration.csv"), cal_rows)
    return rows


def crew_policy_study(outdir="outputs", games=300, base=None, jobs=1):
    """Does belief-aware crew movement (flee / suspect-averse A*) change the game?"""
    base = base or Config()
    variants = [("tasks only", {}),
                ("cautious thr=0.5", {"crew_policy": "cautious", "avoid_threshold": 0.5}),
                ("cautious thr=0.35", {"crew_policy": "cautious", "avoid_threshold": 0.35}),
                ("cautious thr=0.25", {"crew_policy": "cautious", "avoid_threshold": 0.25}),
                ("cautious thr=0.2", {"crew_policy": "cautious", "avoid_threshold": 0.2})]
    rows = []
    for name, kw in variants:
        res = run_batch([base.copy(seed=s, compare_baselines=False, **kw) for s in range(games)], jobs)
        p, lo, hi = _rate(res, lambda r: r["winner"] == "impostor")
        rows.append({"crew_policy": name, "games": games, "impostor_win_rate": round(p, 3), "ci_low": round(lo, 3),
                     "ci_high": round(hi, 3), "mean_kills": round(_mean(r["kills"] for r in res), 2),
                     "mean_ticks": round(_mean(r["ticks"] for r in res), 1),
                     "flee_moves_per_game": round(_mean(r["flee"] for r in res), 2),
                     "innocents_ejected_per_game": round(_mean(r["innocents"] for r in res), 2)})
        print(f"  {name:<18} impostor wins {p:5.1%} (CI {lo:.1%}-{hi:.1%})  kills/game {rows[-1]['mean_kills']:.2f}  "
              f"flee moves/game {rows[-1]['flee_moves_per_game']:.2f}")
    _write_csv(os.path.join(outdir, "crew_policy_study.csv"), rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = range(len(rows))
    ax.bar(xs, [r["impostor_win_rate"] for r in rows], color=["#7f7f7f"] + ["#2ca02c"] * (len(rows) - 1),
           yerr=[[r["impostor_win_rate"] - r["ci_low"] for r in rows], [r["ci_high"] - r["impostor_win_rate"] for r in rows]],
           capsize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["crew_policy"] for r in rows], rotation=20, fontsize=8)
    ax.set_ylabel("impostor win rate")
    ax.set_title(f"Crew movement policy vs smart impostor ({games} paired games)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crew_policy_study.png"), dpi=130)
    plt.close(fig)
    return rows


ABLATIONS = [
    ("full model", {}),
    # --- crew-side rules switched off -------------------------------------------------
    ("crew: no proximity", {"w_prox": 0.0, "w_scene": 0.0}),
    ("crew: no alibi checks", {"w_alibi": 0.0, "w_false": 0.0}),
    ("crew: no group credit", {"w_group": 0.0}),
    ("crew: no dissent rule", {"w_dissent": 0.0}),
    ("crew: no last-seen-with", {"w_last_seen": 0.0}),
    ("crew: perfect memory", {"memory_prob": 1.0}),
    # --- impostor tricks switched off ------------------------------------------------
    ("imp: no alibi lies", {"imp_lie": False}),
    ("imp: no false sighting", {"imp_frame": False}),
    ("imp: vents unsafely", {"imp_vent_safe": False}),
    ("imp: no bandwagon vote", {"imp_bandwagon": False}),
    ("imp: no Minimax gate (always kill)", {"mm_lay_low_value": -99.0, "mm_urgency": 0.0}),
    ("imp: depth 1", {"minimax_depth": 1}),
]


def ablation_study(outdir="outputs", games=300, base=None, jobs=1):
    """Paired seeds, same games for every variant -> differences are due to the switch."""
    base = base or Config()
    rows = []
    for name, kw in ABLATIONS:
        res = run_batch([base.copy(seed=s, compare_baselines=False, **kw) for s in range(games)], jobs)
        p, lo, hi = _rate(res, lambda r: r["winner"] == "impostor")
        rows.append({"variant": name, "games": games, "impostor_win_rate": round(p, 3), "ci_low": round(lo, 3),
                     "ci_high": round(hi, 3), "mean_kills": round(_mean(r["kills"] for r in res), 2),
                     "innocents_ejected_per_game": round(_mean(r["innocents"] for r in res), 2),
                     "mean_rounds": round(_mean(r["rounds"] for r in res), 2)})
        print(f"  {name:<38} impostor wins {p:5.1%}  (CI {lo:.1%}-{hi:.1%})  innocents ejected/game "
              f"{rows[-1]['innocents_ejected_per_game']:.2f}")
    _write_csv(os.path.join(outdir, "ablation_study.csv"), rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ys = list(range(len(rows)))[::-1]
    cols = ["#1f77b4" if r["variant"] == "full model" else ("#2ca02c" if r["variant"].startswith("crew") else "#d62728")
            for r in rows]
    ax.barh(ys, [r["impostor_win_rate"] for r in rows], color=cols,
            xerr=[[r["impostor_win_rate"] - r["ci_low"] for r in rows], [r["ci_high"] - r["impostor_win_rate"] for r in rows]],
            capsize=3)
    ax.axvline(rows[0]["impostor_win_rate"], color="k", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["variant"] for r in rows], fontsize=8)
    ax.set_xlabel("impostor win rate (higher = better for the impostor)")
    ax.set_title(f"Ablations: green = crew rule removed, red = impostor trick removed ({games} paired games)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ablation_study.png"), dpi=130)
    plt.close(fig)
    return rows


SWEEPS = {"w_alibi": [0.0, 1.0, 2.5, 4.0, 6.0],
          "w_prox": [0.0, 1.0, 2.0, 3.0, 4.0],
          "memory_prob": [0.85, 0.9, 0.95, 0.97, 1.0],
          "skip_factor": [1.0, 1.25, 1.6, 2.0, 3.0]}


def sensitivity_study(outdir="outputs", games=200, base=None, jobs=1, sweeps=None):
    base = base or Config()
    sweeps = sweeps or SWEEPS
    rows = []
    for param, values in sweeps.items():
        for v in values:
            res = run_batch([base.copy(seed=s, compare_baselines=False, **{param: v}) for s in range(games)], jobs)
            p, lo, hi = _rate(res, lambda r: r["winner"] == "crew")
            rows.append({"param": param, "value": v, "games": games, "crew_win_rate": round(p, 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                         "innocents_ejected_per_game": round(_mean(r["innocents"] for r in res), 2),
                         "mean_kills": round(_mean(r["kills"] for r in res), 2)})
        print(f"  swept {param}: " + ", ".join(f"{r['value']}->{r['crew_win_rate']:.2f}" for r in rows if r["param"] == param))
    _write_csv(os.path.join(outdir, "sensitivity_study.csv"), rows)
    plt = _plt()
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax, param in zip(axes[0], sweeps):
        sub = [r for r in rows if r["param"] == param]
        ax.errorbar([r["value"] for r in sub], [r["crew_win_rate"] for r in sub],
                    yerr=[[r["crew_win_rate"] - r["ci_low"] for r in sub], [r["ci_high"] - r["crew_win_rate"] for r in sub]],
                    fmt="o-", capsize=3, color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot([r["value"] for r in sub], [r["innocents_ejected_per_game"] for r in sub], "s--", color="#d62728")
        ax2.set_ylabel("innocents ejected / game", color="#d62728", fontsize=8)
        ax.set_xlabel(param)
        ax.set_ylim(0, 1)
        ax.set_ylabel("crew win rate")
    fig.suptitle("Sensitivity of the crew's performance to its own parameters", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sensitivity_study.png"), dpi=130)
    plt.close(fig)
    return rows


def run_all(outdir="outputs", quick=False, jobs=1, which=None):
    os.makedirs(outdir, exist_ok=True)
    which = which or ["heuristic", "scaling", "policy", "belief", "ablation", "sensitivity", "crew"]
    g = 60 if quick else 200
    if "heuristic" in which:
        print("== heuristic / algorithm comparison ==")
        heuristic_study(outdir, pairs_per_map=150 if quick else 400, sizes=(10, 20) if quick else (10, 20, 40))
    if "scaling" in which:
        print("== scaling study ==")
        if quick:
            scaling_study(outdir, trials=2, rooms_list=(10, 20, 30), crew_list=(4, 7, 10), extra_list=(0.0, 0.6, 1.5))
        else:
            scaling_study(outdir, trials=4)
    if "policy" in which:
        print("== policy study ==")
        policy_study(outdir, games=g, jobs=jobs)
    if "belief" in which:
        print("== belief quality ==")
        belief_quality_study(outdir, games=g, jobs=jobs)
    if "ablation" in which:
        print("== ablation study ==")
        ablation_study(outdir, games=100 if quick else 300, jobs=jobs)
    if "crew" in which:
        print("== crew movement policy ==")
        crew_policy_study(outdir, games=100 if quick else 300, jobs=jobs)
    if "sensitivity" in which:
        print("== sensitivity study ==")
        sensitivity_study(outdir, games=50 if quick else 200, jobs=jobs)
    print(f"\nCSV + PNG files written to {outdir}/")


if __name__ == "__main__":
    run_all()
