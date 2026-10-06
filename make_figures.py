"""
make_figures.py - generates the figures the report / slides need, straight from the simulation
(so every picture is reproducible).   Run:  python make_figures.py [--outdir outputs/figures]

  fig_map.png            the room graph: corridor costs, vents (dashed), dead ends (red outline)
  fig_astar_tree.png     the A* SEARCH TREE for one query (expansion order, g/h/f, generated-but-unexpanded
                         leaves, final path highlighted) + the BFS expansion count for comparison
  fig_minimax_tree.png   the Minimax-style tree for one REAL impostor decision recorded in a game
                         (ply 1 targets, ply 2 crew replies, ply 3 escapes, leaf utilities)
  fig_belief_evolution.png  aggregated suspicion per agent over the meetings of one game
Study charts (heuristic / scaling / ablation / ...) come from  python main.py experiments.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from game_loop import run_game
from search import a_star, bfs, dijkstra
from ship_map import default_map


# ---------------------------------------------------------------- map
def fig_map(outdir):
    g = default_map()
    fig, ax = plt.subplots(figsize=(8, 6))
    for u in g.rooms:
        for v, c in g.adj[u].items():
            if u < v:
                (x1, y1), (x2, y2) = g.pos[u], g.pos[v]
                ax.plot([x1, x2], [y1, y2], color="#999", lw=3, zorder=1)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{c:.1f}", fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=2)
        for v in g.vent_adj[u]:
            if u < v:
                (x1, y1), (x2, y2) = g.pos[u], g.pos[v]
                ax.plot([x1, x2], [y1, y2], color="#d62728", lw=1.3, ls="--", zorder=1)
    traps = set(g.trap_rooms())
    for r, (x, y) in g.pos.items():
        ax.scatter([x], [y], s=2400, c="#f4f4f4", edgecolors="#d62728" if r in traps else "#444",
                   linewidths=2.5 if r in traps else 1.2, zorder=3)
        ax.text(x, y, r, ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{g.name}: {len(g.rooms)} rooms, {g.n_corridors()} corridors, branching b = {g.avg_branching():.1f}\n"
                 "grey = corridors (cost), red dashed = impostor-only vents, red outline = dead end without vent",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_map.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- A* tree
def fig_astar_tree(outdir, start="Reactor", goal="Navigation", heuristic="euclidean"):
    g = default_map()
    r = a_star(g, start, goal, heuristic, None, False, trace=True)
    b = bfs(g, start, goal)
    d = dijkstra(g, start, goal)
    parents = r.parents or {}
    order = {t["node"]: (i + 1, t) for i, t in enumerate(r.trace)}
    # tree depth of every generated node
    depth = {start: 0}

    def dep(n):
        if n not in depth:
            depth[n] = dep(parents[n]) + 1
        return depth[n]
    nodes = [start] + list(parents)
    for n in nodes:
        dep(n)
    levels = {}
    for n in nodes:
        levels.setdefault(depth[n], []).append(n)
    pos = {}
    for lv, ns in levels.items():
        ns.sort(key=lambda n: (order[n][0] if n in order else 99, n))
        for k, n in enumerate(ns):
            pos[n] = (k - (len(ns) - 1) / 2.0, -lv)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    on_path = set(r.path)
    for n, p in parents.items():
        (x1, y1), (x2, y2) = pos[p], pos[n]
        hot = n in on_path and p in on_path
        ax.plot([x1, x2], [y1, y2], color="#d62728" if hot else "#aaa", lw=2.5 if hot else 1.2, zorder=1)
    for n in nodes:
        x, y = pos[n]
        if n in order:
            idx, t = order[n]
            lab = f"{n}\n#{idx}  g={t['g']}  h={t['h']}\nf={t['f']}"
            fc = "#ffd9d9" if n in on_path else "#e8f1fb"
            ec = "#d62728" if n in on_path else "#1f77b4"
            ls = "-"
        else:
            lab = f"{n}\n(generated,\nnot expanded)"
            fc, ec, ls = "#f7f7f7", "#999", "--"
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.5, zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, ls=ls))
    ax.set_xlim(min(p[0] for p in pos.values()) - 1, max(p[0] for p in pos.values()) + 1)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.6, 0.6)
    ax.axis("off")
    ax.set_title(f"A* search tree  {start} -> {goal}  (heuristic: {heuristic})\n"
                 f"expanded {r.expanded}, generated {r.generated}, path cost {r.cost:.2f}   |   "
                 f"BFS expanded {b.expanded} (cost {b.cost:.2f}),  Dijkstra expanded {d.expanded}\n"
                 "number = expansion order (lowest f first); red = final path", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_astar_tree.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- Minimax tree
def _find_decision(seed_limit=60):
    for seed in range(seed_limit):
        r = run_game(Config(seed=seed, compare_baselines=False))
        for e in r.minimax_log:
            if e["decision"] == "HUNT" and e.get("candidates") and e["chosen"]["depth"] == 3:
                return seed, e
    raise RuntimeError("no HUNT decision found")


def fig_minimax_tree(outdir):
    seed, e = _find_decision()
    t = e["chosen"]
    cands = e["candidates"]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    ax.text(0.0, 1.0, f"Minimax-STYLE tree (game seed {seed}, tick {e['tick']}):  MAX = impostor, MIN = crew replies  -  "
                      "NOT a classical zero-sum board game", transform=ax.transAxes, fontsize=9, va="top")
    # ply 1: candidates (left column)
    ys = [0.86 - i * (0.78 / max(1, len(cands) - 1)) for i in range(len(cands))]
    for y, c in zip(ys, cands):
        chosen = c["target"] == t["target"]
        ax.text(0.04, y, f"KILL {c['target']}\nvalue={c['value']:+.3f}", fontsize=7.5, va="center", ha="left",
                bbox=dict(boxstyle="round", fc="#ffd9d9" if chosen else "#f2f2f2", ec="#d62728" if chosen else "#aaa"))
        if chosen:
            cy = y
    ax.text(0.04, 0.97, "ply 1  MAX: choose target", fontsize=8, weight="bold", va="top")
    # ply 2: replies for the chosen target
    replies = list(t["replies"])
    ry = [0.80 - i * 0.21 for i in range(len(replies))]
    ax.text(0.36, 0.97, "ply 2  MIN: crew reply (lowest U)", fontsize=8, weight="bold", va="top")
    ax.text(0.70, 0.97, "ply 3  MAX: impostor escape (highest U)", fontsize=8, weight="bold", va="top")
    for y, rp in zip(ry, replies):
        d = t["replies"][rp]
        worst = rp == t["worst_reply"]
        ax.annotate("", xy=(0.355, y), xytext=(0.17, cy), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color="#d62728" if worst else "#bbb", lw=2 if worst else 1))
        ax.text(0.36, y, f"{rp}\nbest escape: {d['best_escape']}\nU = {d['U']:+.3f}", fontsize=7.5, va="center",
                bbox=dict(boxstyle="round", fc="#ffe9c9" if worst else "#eef5ff", ec="#d62728" if worst else "#1f77b4"))
        opts = d["options"]
        oy = [y + 0.07 - j * 0.07 for j in range(len(opts))]
        for yy, (f, o) in zip(oy, opts.items()):
            best = f == d["best_escape"]
            ax.annotate("", xy=(0.695, yy), xytext=(0.53, y), xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-", color="#2ca02c" if best else "#ddd", lw=1.6 if best else 0.8))
            ax.text(0.70, yy, f"{f}: suspicion={o['suspicion']:.2f}  U={o['U']:+.3f}", fontsize=7, va="center",
                    bbox=dict(boxstyle="round", fc="#e6f5e6" if best else "#fafafa", ec="#2ca02c" if best else "#ddd"))
    ax.text(0.04, 0.02,
            f"leaf  U = progress({t['progress']}) - mm_lambda * suspicion - mm_travel_w * approach_cost({t['approach_cost']})    |    "
            f"value(target) = min over replies of max over escapes of U = {t['value']:+.3f}    |    "
            f"T_disc={t['T_disc']} ticks, trail={t['trail']}, nearby={t['nearby']}", fontsize=7.5, transform=ax.transAxes)
    fig.savefig(os.path.join(outdir, "fig_minimax_tree.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- belief evolution
def fig_belief_evolution(outdir, seeds=range(80)):
    best = None
    for seed in seeds:
        r = run_game(Config(seed=seed, compare_baselines=False))
        if len(r.meetings) >= 3 and r.winner == "crew":
            best = r
            break
    if best is None:
        best = run_game(Config(seed=0, compare_baselines=False))
    r = best
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = r.agent_order
    for n in names:
        xs = [i + 1 for i, m in enumerate(r.meetings) if n in m["agg"]]
        ys = [m["agg"][n] for m in r.meetings if n in m["agg"]]
        if not xs:
            continue
        imp = n == r.impostor
        ax.plot(xs, ys, "o-", color=r.agent_colors[n], lw=3.2 if imp else 1.3, alpha=1 if imp else 0.7,
                label=n + (" (IMPOSTOR)" if imp else ""))
    ax.set_xlabel("meeting #")
    ax.set_ylabel("aggregated suspicion (mean over crew)")
    ax.set_xticks(range(1, len(r.meetings) + 1))
    ax.set_title(f"Belief evolution, seed {r.cfg.seed}: {r.winner} wins - {r.reason}", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_belief_evolution.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for f in (fig_map, fig_astar_tree, fig_minimax_tree, fig_belief_evolution):
        f(a.outdir)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
