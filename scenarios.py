"""
scenarios.py - the required working cases and edge cases.

NOTHING HERE IS SCRIPTED.  The simulation is deterministic given (config, seed), so each case is
found by SWEEPING SEEDS and keeping the first game whose recorded events satisfy a predicate.
The predicate only inspects what happened; it never changes the game.  Every case log states the
seed + policy so the exact game can be replayed:  python main.py run --seed N --impostor P

  WORKING CASES
    1 alibi_contradiction_win    crew ejects the impostor mainly because its alibi was refuted
    1b smart_impostor_impossible_travel  same, but against the smart impostor (internal inconsistency)
    2 impostor_vent_misdirection impostor wins; escaped via an unwitnessed vent and misdirected the
                                 crew into ejecting innocents (false sighting planted)
  EDGE CASES
    3 innocent_wrongly_ejected   an innocent is voted out because of a (coincidental) bad alibi
    4 tie_no_elimination         the vote ends in a tie -> nobody is ejected
    5 impostor_trapped_dead_end  impostor kills in a dead-end room without a vent and is found there
    6 missing_crewmate_last_seen a player is missing (body unfound); the vote is driven by who was last seen with them
"""
import json
import os
from typing import Callable, Dict, List, Optional

from config import Config
from game_loop import run_game, GameResult

BAD_ALIBI_KINDS = ("seen_elsewhere", "impossible", "self_incons")


def _imp_conflicts(m: dict) -> List[dict]:
    return [c for c in m["conflicts"] if c["claimant"] == m["impostor"]
            and c["kind"] in BAD_ALIBI_KINDS]


def _case1(r: GameResult, need_kind: str) -> Optional[int]:
    if r.winner != "crew" or "voted out" not in r.reason or r.stats.get("kills", 0) < 1:
        return None
    m = r.meetings[-1]
    mine = _imp_conflicts(m)
    if m["ejected"] != m["impostor"] or len(mine) < 2 or not any(c["kind"] == need_kind for c in mine):
        return None
    # the refuted alibi must have been the biggest contributor to the impostor's score
    bd = [m["breakdown"][o][m["impostor"]] for o in m["breakdown"]]
    alibi = sum(b["alibi"] for b in bd) / len(bd)
    others = max(sum(b[k] for b in bd) / len(bd) for k in ("prox", "scene", "vent"))
    return len(r.meetings) - 1 if alibi > others and alibi > 0 else None


def case1(r: GameResult) -> Optional[int]:
    """'claims room X but was seen in room Y' - the literal alibi contradiction."""
    return _case1(r, "seen_elsewhere")


def case1b(r: GameResult) -> Optional[int]:
    """smart impostor, whose lies are never seen-elsewhere, is still caught by IMPOSSIBLE travel
    (an unavoidable vent jump contradicts its own corridor-only alibi)."""
    return _case1(r, "impossible")


def case2(r: GameResult) -> Optional[int]:
    s = r.stats
    if (r.winner == "impostor" and s.get("vent_escapes_unseen", 0) >= 1 and s.get("vent_witnessed", 0) == 0
            and s.get("innocents_ejected", 0) >= 1 and s.get("frames_planted", 0) >= 1):
        return len(r.meetings) - 1
    return None


def case3(r: GameResult) -> Optional[int]:
    for i, m in enumerate(r.meetings):
        e = m["ejected"]
        if e and m.get("ejected_role") == "crew":
            mine = [c for c in m["conflicts"] if c["claimant"] == e and c["kind"] in ("seen_elsewhere", "absent")]
            imp_bad = len(_imp_conflicts(m))
            if mine and imp_bad == 0:
                return i
    return None


def case4(r: GameResult) -> Optional[int]:
    for i, m in enumerate(r.meetings):
        if m["ejected"] is None and m["outcome_reason"] == "tie":
            return i
    return None


def case5(r: GameResult) -> Optional[int]:
    for i, m in enumerate(r.meetings):
        if m.get("trapped"):
            return i
    return None


def case6(r: GameResult) -> Optional[int]:
    for i, m in enumerate(r.meetings):
        e = m["ejected"]
        if e and m.get("missing"):
            bd = [m["breakdown"][o][e] for o in m["breakdown"] if e in m["breakdown"][o]]
            if not bd:
                continue
            avg = {k: sum(b[k] for b in bd) / len(bd) for k in ("prox", "scene", "alibi", "vent", "last_seen")}
            if avg["last_seen"] > 0 and avg["last_seen"] == max(avg.values()):
                return i
    return None


CASES: Dict[str, dict] = {
    "case1_alibi_contradiction_win": dict(pred=case1, kind="working", policies=["greedy", "random"],
        title="Working case 1: impostor identified through a contradicted alibi (seen elsewhere)"),
    "case1b_smart_impostor_impossible_travel": dict(pred=case1b, kind="working", policies=["minimax"],
        title="Working case 1b: SMART impostor caught by an impossible-travel alibi"),
    "case2_impostor_vent_misdirection_win": dict(pred=case2, kind="working", policies=["minimax"],
        title="Working case 2: impostor wins via vent escape + misdirection"),
    "case3_innocent_wrongly_ejected": dict(pred=case3, kind="edge", policies=["minimax", "greedy"],
        title="Edge case 1: innocent voted out because of a bad alibi"),
    "case4_tie_no_elimination": dict(pred=case4, kind="edge", policies=["minimax", "greedy"],
        title="Edge case 2: tied vote -> no elimination"),
    "case5_impostor_trapped_dead_end": dict(pred=case5, kind="edge", policies=["greedy", "random", "minimax"],
        title="Edge case 3: impostor trapped in a vent-less dead end"),
    "case6_missing_crewmate_last_seen": dict(pred=case6, kind="edge", policies=["minimax", "greedy"],
        title="Edge case 4: a missing crewmate (body never found) - ejection driven by 'last seen with'"),
}


def find_cases(max_seeds: int = 1500, base: Optional[Config] = None, wanted=None, progress=True):
    """Sweep seeds for every case.  Returns {case: (seed, policy, meeting_index)}."""
    base = base or Config()
    wanted = list(wanted or CASES)
    found: Dict[str, tuple] = {}
    for name in wanted:
        spec = CASES[name]
        for pol in spec["policies"]:
            for seed in range(max_seeds):
                cfg = base.copy(seed=seed, impostor_policy=pol, compare_baselines=False)
                r = run_game(cfg)
                idx = spec["pred"](r)
                if idx is not None:
                    found[name] = (seed, pol, idx)
                    if progress:
                        print(f"  found {name}: seed={seed} policy={pol}")
                    break
            if name in found:
                break
        else:
            if progress:
                print(f"  NOT FOUND within {max_seeds} seeds: {name}")
    return found


def explain_meeting(r: GameResult, idx: int) -> List[str]:
    m = r.meetings[idx]
    imp = m["impostor"]
    out = [f"Meeting #{idx + 1} (round {m['round']}, tick {m['tick']}, trigger={m['reason']})"]
    if m["body"]:
        b = m["body"]
        out.append(f"  body of {b['victim']} found by {b['finder']} in {b['room']} at tick {b['found_tick']}")
    if m["killed"]:
        out.append(f"  killed this round: {m['killed']}")
    if m.get("missing"):
        out.append(f"  roster check: missing (body not found) = {m['missing']};  last-seen-with scores = "
                   + ", ".join(f"{k}={v:.2f}" for k, v in sorted(m['last_with'].items(), key=lambda kv: -kv[1])))
    if m.get("frame"):
        tf, rf, z = m["frame"]
        out.append(f"  [ground truth] impostor planted a false sighting: '{z} was in {rf} at tick {tf}'")
    out.append("  alibi conflicts raised:")
    for c in m["conflicts"]:
        truth = "claimant is the IMPOSTOR" if c["claimant"] == imp else "claimant is an innocent"
        out.append(f"    - [{c['kind']}] {c['detail']}  ({truth})")
    if not m["conflicts"]:
        out.append("    (none)")
    out.append("  suspicion breakdown averaged over crew observers (log-score terms):")
    names = sorted(m["agg"], key=lambda n: -m["agg"][n])
    for n in names[:4]:
        bds = [m["breakdown"][o][n] for o in m["breakdown"] if n in m["breakdown"][o]]
        if not bds:
            continue
        avg = {k: sum(b[k] for b in bds) / len(bds) for k in bds[0]}
        tag = " <== IMPOSTOR" if n == imp else ""
        out.append(f"    {n:<7} prior={avg['prior']:.2f} prox={avg['prox']:.2f} scene={avg['scene']:.2f} "
                   f"alibi={avg['alibi']:.2f} false={avg['false_testimony']:.2f} vent={avg['vent']:.2f} "
                   f"group={avg['group']:.2f} last_seen={avg.get('last_seen', 0):.2f} -> posterior={avg['posterior']:.2f}{tag}")
    out.append("  votes: " + ", ".join(f"{k}->{v or 'SKIP'}" for k, v in m["votes"].items()))
    out.append(f"  tally={m['tally']}  outcome={'EJECT ' + m['ejected'] if m['ejected'] else 'NO ELIMINATION'} "
               f"({m['outcome_reason']})  ejected_role={m.get('ejected_role', '-')}")
    return out


def write_case_logs(found: Dict[str, tuple], outdir="outputs/cases", base: Optional[Config] = None, gifs=False):
    base = base or Config()
    os.makedirs(outdir, exist_ok=True)
    summary = {}
    for name, (seed, pol, idx) in found.items():
        cfg = base.copy(seed=seed, impostor_policy=pol, record_frames=gifs, compare_baselines=True)
        r = run_game(cfg)
        spec = CASES[name]
        lines = [spec["title"], "=" * len(spec["title"]), "",
                 f"kind            : {spec['kind']} case",
                 f"replay          : python main.py run --seed {seed} --impostor {pol}",
                 f"impostor        : {r.impostor}   policy={pol}",
                 f"outcome         : {r.winner} wins - {r.reason}  (rounds={r.rounds}, ticks={r.ticks})",
                 f"stats           : {r.stats}", "",
                 "KEY MEETING", "-----------"] + explain_meeting(r, idx) + ["", "FULL EVENT LOG", "--------------"] + r.events
        sm = r.search_log.summary()
        if "astar_expanded_mean" in sm:
            lines += ["", "A* vs BFS on this game's path queries", "-------------------------------------",
                      f"queries={sm['queries']}  mean nodes expanded: A*={sm['astar_expanded_mean']:.2f} "
                      f"BFS={sm['bfs_expanded_mean']:.2f} Dijkstra={sm['dij_expanded_mean']:.2f}; "
                      f"mean path cost: A*={sm['astar_cost_mean']:.2f} BFS={sm['bfs_cost_mean']:.2f}"]
        if pol == "minimax" and r.minimax_log:
            lines += ["", "MINIMAX-STYLE DECISIONS (kill / hunt / lay-low), last 12", "-" * 56]
            for e in r.minimax_log[-12:]:
                t = e["chosen"]
                lines.append(f"t={e['tick']:03d} {e['decision']:<22} target={t['target']:<7} K={t['kill_room']:<10} "
                             f"value={t['value']:+.3f} worst_reply={t['worst_reply']} escape={t['escape']}")
        path = os.path.join(outdir, name + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        summary[name] = {"seed": seed, "policy": pol, "winner": r.winner, "reason": r.reason,
                         "title": spec["title"], "log": path}
        if gifs:
            from visualize import animate
            animate(r, reveal=True, save=os.path.join(outdir, name + ".gif"), show=False, interval=500)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_cases(max_seeds=1500, outdir="outputs/cases", gifs=False, base: Optional[Config] = None):
    print(f"Sweeping up to {max_seeds} seeds per case (deterministic, nothing scripted)...")
    found = find_cases(max_seeds, base)
    summary = write_case_logs(found, outdir, base, gifs)
    print(f"\nWrote {len(summary)} case logs to {outdir}/")
    for n, s in summary.items():
        print(f"  {n:<40} seed={s['seed']:<5} policy={s['policy']:<8} -> {s['winner']} ({s['reason']})")
    missing = [n for n in CASES if n not in found]
    if missing:
        print("Not found (increase --max-seeds):", ", ".join(missing))
    return summary


if __name__ == "__main__":
    run_cases()
