"""
main.py - command line entry point.

  python main.py run                      one game, narrated in the terminal
  python main.py run --animate            ... and replay it with matplotlib (live window)
  python main.py run --animate --save out.gif
  python main.py cases                    find + log the 2 working and 3 edge cases (outputs/cases/)
  python main.py experiments [--quick]    A* vs BFS, scaling, smart-vs-dumb impostor (CSV + PNG)
  python main.py compare --games 200      only the smart-vs-dumb impostor comparison
  python main.py trace --start Reactor --goal Navigation [--heuristic zero]   print an A* search tree
"""
import argparse
import os
import sys

from config import Config
from search import HEURISTICS


def add_game_args(p):
    p.add_argument("--crew", type=int, default=7, help="number of crewmates (default 7; +1 impostor)")
    p.add_argument("--rooms", type=int, default=10, help="rooms (10 = hand-built map, otherwise generated)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impostor", choices=["minimax", "greedy", "random"], default="minimax",
                   help="impostor policy: minimax (smart) | greedy | random (dumb)")
    p.add_argument("--depth", type=int, choices=[1, 2, 3], default=3, help="minimax-style tree depth")
    p.add_argument("--heuristic", choices=list(HEURISTICS), default="euclidean", help="A* heuristic")
    p.add_argument("--tasks", type=int, default=None, help="tasks per crewmate")
    p.add_argument("--round-ticks", type=int, default=None)
    p.add_argument("--crew-policy", choices=["tasks", "cautious"], default="tasks",
                   help="crew movement: tasks only, or cautious (flee/avoid the top suspect)")
    p.add_argument("--config", default=None, help="JSON preset (see presets/); typed flags override it")
    p.add_argument("--no-baselines", action="store_true", help="do not run BFS/Dijkstra next to every A* call")


def cfg_from(a, **extra) -> Config:
    if getattr(a, "config", None):
        base = Config.from_json(a.config)           # preset file; explicit CLI flags below still override it
    else:
        base = None
    cfg = Config(n_crew=a.crew, n_rooms=a.rooms, seed=a.seed, impostor_policy=a.impostor,
                 minimax_depth=a.depth, heuristic=a.heuristic, crew_policy=a.crew_policy, compare_baselines=not a.no_baselines)
    if base is not None:
        # keep every preset value except the flags the user actually typed
        defaults = {"crew": 7, "rooms": 10, "seed": 0, "impostor": "minimax", "depth": 3, "heuristic": "euclidean"}
        pairs = {"crew": "n_crew", "rooms": "n_rooms", "seed": "seed", "impostor": "impostor_policy",
                 "depth": "minimax_depth", "heuristic": "heuristic"}
        for flag, field in pairs.items():
            if getattr(a, flag) == defaults[flag]:
                setattr(cfg, field, getattr(base, field))
        for field in base.__dataclass_fields__:
            if field not in pairs.values() and field not in ("compare_baselines", "record_frames", "verbose"):
                setattr(cfg, field, getattr(base, field))
        if a.crew_policy == "tasks":
            cfg.crew_policy = base.crew_policy
    if a.tasks is not None:
        cfg.tasks_per_crew = a.tasks
    if a.round_ticks is not None:
        cfg.round_ticks = a.round_ticks
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def cmd_run(a):
    from game_loop import run_game
    cfg = cfg_from(a, record_frames=bool(a.animate or a.save), verbose=not a.quiet)
    r = run_game(cfg)
    print("\n" + "=" * 70)
    print(f"RESULT: {r.winner or 'nobody'} wins - {r.reason}")
    print(f"impostor was {r.impostor}; rounds={r.rounds}, ticks={r.ticks}; stats={r.stats}")
    sm = r.search_log.summary()
    if "astar_expanded_mean" in sm:
        print(f"A* queries={sm['queries']}: mean nodes expanded A*={sm['astar_expanded_mean']:.2f} "
              f"BFS={sm['bfs_expanded_mean']:.2f} Dijkstra={sm['dij_expanded_mean']:.2f}; "
              f"mean cost A*={sm['astar_cost_mean']:.2f} BFS={sm['bfs_cost_mean']:.2f}")
    os.makedirs(a.out, exist_ok=True)
    r.search_log.to_csv(os.path.join(a.out, "search_log.csv"))
    import csv
    with open(os.path.join(a.out, "movement_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tick", "round", "agent", "role", "event", "from", "to"])
        w.writeheader()
        w.writerows(r.move_log)
    with open(os.path.join(a.out, "game_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(r.events) + "\n")
    with open(os.path.join(a.out, "round_times.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.round_times[0]))
        w.writeheader()
        w.writerows(r.round_times)
    print(f"logs written to {a.out}/  (search_log.csv, movement_log.csv, game_log.txt, round_times.csv)")
    if a.animate or a.save:
        from visualize import animate
        animate(r, reveal=not a.hide_impostor, interval=a.interval, save=a.save, show=not a.save)


def cmd_cases(a):
    from scenarios import run_cases
    base = Config.from_json(a.config) if a.config else None
    run_cases(a.max_seeds, os.path.join(a.out, "cases"), gifs=a.gifs, base=base)


def cmd_experiments(a):
    import experiments
    experiments.run_all(a.out, quick=a.quick, jobs=a.jobs, which=a.only.split(",") if a.only else None)


def cmd_compare(a):
    import experiments
    base = Config.from_json(a.config) if a.config else None
    experiments.policy_study(a.out, games=a.games, jobs=a.jobs, base=base)


def cmd_trace(a):
    from search import a_star, bfs, print_trace
    from ship_map import default_map
    g = default_map()
    if a.algo == "astar":
        r = a_star(g, a.start, a.goal, a.heuristic, None, a.vents, trace=True)
        print_trace(r)
    else:
        from search import ida_star, greedy_best_first
        fn = ida_star if a.algo == "ida" else greedy_best_first
        r = fn(g, a.start, a.goal, a.heuristic, None, a.vents)
        extra = f" iterations={r.iterations}" if a.algo == "ida" else ""
        print(f"{r.algo}: path={' -> '.join(r.path)} cost={r.cost:.2f} expanded={r.expanded} "
              f"generated={r.generated} memory(max depth/frontier)={r.max_frontier}{extra}")
    b = bfs(g, a.start, a.goal, None, a.vents)
    print(f"BFS: path={' -> '.join(b.path)} cost={b.cost:.2f} expanded={b.expanded}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Social deduction under uncertainty - Among-Us-style multi-agent simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run one game")
    add_game_args(p)
    p.add_argument("--animate", action="store_true", help="show the matplotlib replay")
    p.add_argument("--save", default=None, help="save the replay (.gif / .mp4) instead of showing it")
    p.add_argument("--hide-impostor", action="store_true", help="do not reveal the impostor / vents in the animation")
    p.add_argument("--interval", type=int, default=600, help="ms per animation frame")
    p.add_argument("--quiet", action="store_true", help="do not narrate the game")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("cases", help="find and log the required working / edge cases")
    p.add_argument("--max-seeds", type=int, default=1500)
    p.add_argument("--gifs", action="store_true", help="also export an animated GIF per case (slow)")
    p.add_argument("--config", default=None, help="JSON preset used as the base config")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_cases)

    p = sub.add_parser("experiments", help="complexity / comparison studies (CSV + PNG)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes (results identical to --jobs 1)")
    p.add_argument("--only", default=None, help="comma list: heuristic,scaling,policy,belief,ablation,sensitivity,crew")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_experiments)

    p = sub.add_parser("compare", help="smart vs dumb impostor win rates")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--config", default=None, help="JSON preset used as the base config")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("trace", help="print the A* search tree for one query on the 10-room map")
    p.add_argument("--start", default="Reactor")
    p.add_argument("--goal", default="Navigation")
    p.add_argument("--heuristic", choices=list(HEURISTICS), default="euclidean")
    p.add_argument("--vents", action="store_true", help="allow vent edges (impostor graph)")
    p.add_argument("--algo", choices=["astar", "ida", "greedy"], default="astar", help="which search to trace/print")
    p.set_defaults(fn=cmd_trace)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
