"""
suspicion.py - the crewmates' belief system.  Everything here is deliberately
simple, additive in log-space, and driven by the weights in Config.

BELIEF STATE
------------
Crewmate i keeps P_i(x) for every other living agent x:
    P_i(x) = probability (in i's mind) that x is the impostor,   sum_x P_i(x) = 1.
(i never suspects itself: a crewmate knows it is innocent.)  Start: uniform.

UPDATE RULE (once per meeting, after the testimonies are heard)
---------------------------------------------------------------
For every suspect x compute a log-score

    s_i(x) = (1-d)*ln P_i(x) + d*ln(1/n)                       # decay towards uniform
           + g_i * [  W_PROX  * prox(x)                         # near the body
                    + W_SCENE * scene(x)                        # standing at the body when found
                    + A_i(x)                                    # refuted alibis (see below)
                    + W_FALSE * F_i(x)                          # accusation refuted by corroborated alibi
                    + W_VENT  * vent(x)                         # seen using a vent
                    - W_GROUP * group(x) ]                      # seen in a group => alibi corroboration

and then the NEW belief is the softmax   P_i'(x) = exp(s_i(x)) / sum_y exp(s_i(y)).
Equivalently P' is proportional to  P * exp(weights): a naive-Bayes style update where each
rule contributes a likelihood ratio exp(W * feature).  g_i is the crewmate's personal
"gullibility" (evidence scale, ~N(1, sd)) - the only source of disagreement between
crewmates besides what they personally saw and which sightings they forgot.

After the vote, a separate rule applies:
    DISSENT:  s_i(x) += W_DISSENT  for every x that voted against the eventual majority.

FEATURES (all computed from PUBLIC TESTIMONY only - never from hidden game truth)
--------------------------------------------------------------------------------
prox(x)   in [0,1]: during the window (victim last seen alive, body found], how close to the body
          room did anyone place x?   prox = max_t 1/(1 + hops(pos_x(t), body_room)).
          pos_x(t) comes from x's own claims, overridden by other people's sightings of x.
          If nobody can place x at all in the window prox = PROX_UNKNOWN.
scene(x)  1 if x (other than the finder) was in the body room at the tick it was found, else 0.
          (The finder's own presence at discovery is ignored: that is how it found the body.)
A_i(x)    refuted-alibi penalty (the SHARP rule).  A "conflict" exists when
            (a) seen_elsewhere : x claims room A at tick t but reporter O says "I saw x in B at t";
            (b) absent         : x claims A at t, reporter O claims A at t too, yet O did not see x
                                 (counts only ABSENT_FACTOR - O might simply have forgotten);
            (c) impossible     : x's own consecutive claims need more ticks than elapsed;
            (d) self_incons    : O reports seeing someone in a room O itself claims not to be in.
          Who is blamed for (a)/(b)?  If a third witness corroborates x's claim -> the reporter is
          blamed (W_FALSE).  Else if a third witness corroborates the sighting -> x is blamed
          (W_ALIBI).  Else it is a he-said/she-said: the penalty is SPLIT between x and O in
          proportion to the observer's current beliefs P(x) : P(O).  (c)/(d) blame only the speaker.
          Every penalty is multiplied by trust_i(reporter) = 1 - TRUST_SLOPE * P_i(reporter)
          (1 for i's own first-hand data) and the total per suspect is capped at ALIBI_CAP.
vent(x)   number of witnesses who saw x use a vent (capped at 1).
group(x)  fraction (capped at 1) of GROUP_FULL_TICKS ticks, inside the window, in which x was
          reported by *someone else* in a room with at least 2 other agents.
"""
import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Testimony:
    who: str
    claims: Dict[int, str]                       # tick -> room ("I was in room R at tick t")
    sightings: List[Tuple[int, str, str]]        # (tick, room, other) "I saw `other` in `room`"
    vent_sightings: List[Tuple[int, str, str]] = field(default_factory=list)   # (tick, room, who vented)


@dataclass
class Body:
    room: str
    victim: str
    found_tick: int
    finder: str


@dataclass
class Conflict:
    claimant: str
    reporter: Optional[str]
    tick: int
    kind: str                       # seen_elsewhere | absent | impossible | self_incons
    factor: float
    claim_corroborated: bool = False
    sighting_corroborated: bool = False
    detail: str = ""


@dataclass
class Evidence:
    prox: Dict[str, float] = field(default_factory=dict)
    scene: Dict[str, float] = field(default_factory=dict)
    conflicts: List[Conflict] = field(default_factory=list)
    vent: Dict[str, List[str]] = field(default_factory=dict)    # who -> reporters
    group: Dict[str, int] = field(default_factory=dict)
    window: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
def build_evidence(tests: Dict[str, Testimony], body: Optional[Body], smap, cfg,
                   round_start: int, round_end: int, suspects: List[str]) -> Evidence:
    ev = Evidence()
    sight = defaultdict(list)            # (tick, who) -> [(reporter, room)]
    claim_at = defaultdict(dict)         # tick -> {agent: claimed room}
    for rep, T in tests.items():
        for (t, room, who) in T.sightings:
            sight[(t, who)].append((rep, room))
        for t, room in T.claims.items():
            claim_at[t][rep] = room

    # ---- (a),(b) alibi conflicts --------------------------------------------------
    for z, T in tests.items():
        for t, A in T.claims.items():
            seen_by = sight.get((t, z), [])
            for (o, B) in seen_by:
                if o == z or B == A:
                    continue
                c_claim = any(w not in (z, o) and r == A for w, r in seen_by)
                c_sight = any(w not in (z, o) and r == B for w, r in seen_by)
                ev.conflicts.append(Conflict(z, o, t, "seen_elsewhere", 1.0, c_claim, c_sight,
                                             f"{z} claims {A}@{t}, {o} saw {z} in {B}"))
            reporters = {w for w, _ in seen_by}
            for o, ro in claim_at[t].items():
                if o == z or ro != A or o in reporters:
                    continue
                c_claim = any(w not in (z, o) and r == A for w, r in seen_by)
                c_sight = any(w2 not in (z, o) and r2 == A and w2 not in reporters
                              for w2, r2 in claim_at[t].items())
                ev.conflicts.append(Conflict(z, o, t, "absent", cfg.absent_factor, c_claim, c_sight,
                                             f"{z} claims {A}@{t}, {o} was there and did not see {z}"))
    # ---- (c) impossible travel ---------------------------------------------------------
    for z, T in tests.items():
        ts = sorted(T.claims)
        for t1, t2 in zip(ts, ts[1:]):
            a, b = T.claims[t1], T.claims[t2]
            if smap.tick_dist(a, b) > t2 - t1:
                ev.conflicts.append(Conflict(z, None, t2, "impossible", 1.0, False, True,
                                             f"{z}: {a}@{t1} -> {b}@{t2} is too fast"))
    # ---- (d) reporter placed itself in two rooms ------------------------------------------
    for o, T in tests.items():
        for (t, room, who) in T.sightings:
            mine = T.claims.get(t)
            if mine is not None and mine != room:
                ev.conflicts.append(Conflict(o, None, t, "self_incons", 1.0, False, True,
                                             f"{o} claims {mine}@{t} but reports seeing {who} in {room}"))

    # ---- proximity / scene (only when a body was reported) -----------------------------
    w0 = round_start
    if body is not None:
        last_seen = [t for (t, who), lst in sight.items() if who == body.victim]
        w0 = max(last_seen) if last_seen else round_start
        w1 = body.found_tick
        ev.window = (w0, w1)
        for x in suspects:
            pos = {}
            if x in tests:
                pos.update({t: r for t, r in tests[x].claims.items() if w0 < t <= w1})
            for (t, who), lst in sight.items():
                if who == x and w0 < t <= w1:
                    pos[t] = lst[0][1]                         # other people's sightings override claims
            # The finder is standing at the body only BECAUSE it found it, so that moment is
            # not evidence against it: ignore the discovery tick when scoring the finder.
            scored = {t: r for t, r in pos.items() if not (x == body.finder and t == w1)}
            if not scored:
                ev.prox[x] = cfg.prox_unknown
            else:
                ev.prox[x] = max(1.0 / (1.0 + smap.hops(r, body.room)) for r in scored.values())
            ev.scene[x] = 1.0 if (x != body.finder and pos.get(w1) == body.room) else 0.0
    else:
        ev.window = (round_start, round_end)

    # ---- vent sightings -----------------------------------------------------------------
    for rep, T in tests.items():
        for (t, room, who) in T.vent_sightings:
            ev.vent.setdefault(who, []).append(rep)

    # ---- group (alibi corroboration) ---------------------------------------------------
    lo, hi = ev.window
    groups = defaultdict(set)            # (tick, room) -> agents known to be together
    seen_by_other = defaultdict(set)     # x -> ticks x was reported by somebody else
    for rep, T in tests.items():
        for (t, room, who) in T.sightings:
            if lo < t <= hi:
                groups[(t, room)].update((rep, who))
                seen_by_other[who].add(t)
    for x in suspects:
        n = 0
        for t in seen_by_other.get(x, ()):
            for (tt, room), members in groups.items():
                if tt == t and x in members and len(members) >= 3:
                    n += 1
                    break
        ev.group[x] = n
    return ev


# ---------------------------------------------------------------------------
def normalize(p: Dict[str, float]) -> Dict[str, float]:
    s = sum(p.values())
    if s <= 0:
        return {k: 1.0 / len(p) for k in p}
    return {k: v / s for k, v in p.items()}


def update_belief(observer: str, belief: Dict[str, float], ev: Evidence, cfg,
                  gullibility: float) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Apply the meeting update to one crewmate's belief.  Returns (new_belief, breakdown)
    where breakdown[x] lists every term added to x's log-score (for the viva / the log)."""
    P = normalize(belief)
    suspects = list(P)
    n = len(suspects)

    def trust(rep: Optional[str]) -> float:
        if rep is None or rep == observer:
            return 1.0
        return max(0.05, 1.0 - cfg.trust_slope * P.get(rep, 0.0))

    alibi = {x: 0.0 for x in suspects}
    false = {x: 0.0 for x in suspects}
    for c in ev.conflicts:
        z, o = c.claimant, c.reporter
        base = cfg.w_alibi * c.factor
        if c.kind in ("impossible", "self_incons"):
            if z in alibi:
                alibi[z] += base
            continue
        if z == observer:
            continue
        # observer is itself the reporter -> first-hand, certain
        first_hand = (o == observer)
        if first_hand or (c.sighting_corroborated and not c.claim_corroborated):
            if z in alibi:
                alibi[z] += base * trust(o)
        elif c.claim_corroborated and not c.sighting_corroborated:
            if o in false:
                false[o] += cfg.w_false * c.factor * trust(z)
        else:                                  # unresolved: split blame by current beliefs
            pz, po = P.get(z, 0.0), P.get(o, 0.0)
            share_z = pz / (pz + po) if (pz + po) > 0 else 0.5
            if z in alibi:
                alibi[z] += base * trust(o) * share_z
            if o in false:
                false[o] += base * trust(z) * (1.0 - share_z)
    for x in suspects:
        alibi[x] = min(alibi[x], cfg.alibi_cap)

    d = cfg.belief_decay
    score, breakdown = {}, {}
    for x in suspects:
        t_prox = cfg.w_prox * ev.prox.get(x, 0.0)
        t_scene = cfg.w_scene * ev.scene.get(x, 0.0)
        t_vent = cfg.w_vent * min(1, len(ev.vent.get(x, [])))
        t_group = -cfg.w_group * min(1.0, ev.group.get(x, 0) / max(1, cfg.group_full_ticks))
        ev_sum = t_prox + t_scene + alibi[x] + false[x] + t_vent + t_group
        score[x] = ((1 - d) * math.log(max(P[x], 1e-12)) + d * math.log(1.0 / n)
                    + gullibility * ev_sum)
        breakdown[x] = {"prior": round(P[x], 4), "prox": round(t_prox, 3), "scene": round(t_scene, 3),
                        "alibi": round(alibi[x], 3), "false_testimony": round(false[x], 3),
                        "vent": round(t_vent, 3), "group": round(t_group, 3)}
    m = max(score.values())
    e = {x: math.exp(s - m) for x, s in score.items()}
    new = normalize(e)
    for x in suspects:
        breakdown[x]["posterior"] = round(new[x], 4)
    return new, breakdown


def apply_dissent(belief: Dict[str, float], dissenters: List[str], cfg) -> Dict[str, float]:
    """After the vote: voting against the eventual majority slightly raises suspicion."""
    if not dissenters:
        return belief
    sc = {x: math.log(max(p, 1e-12)) + (cfg.w_dissent if x in dissenters else 0.0)
          for x, p in belief.items()}
    m = max(sc.values())
    return normalize({x: math.exp(s - m) for x, s in sc.items()})


def remove_agents(belief: Dict[str, float], gone: List[str]) -> Dict[str, float]:
    """Dead / ejected agents cannot be the impostor any more (impostor is alive until ejected)."""
    b = {x: p for x, p in belief.items() if x not in gone}
    return normalize(b) if b else b


# ------------------------------------------------------------------ voting
def decide_vote(belief: Dict[str, float], cfg) -> Optional[str]:
    """Vote for my argmax suspect, or SKIP (None) if even my top suspect is barely above uniform."""
    if not belief:
        return None
    x, p = max(sorted(belief.items()), key=lambda kv: kv[1])
    if p < cfg.skip_factor / len(belief):
        return None
    return x


def aggregate(beliefs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggregated suspicion = mean over observers of P_i(x) (observer excluded for itself)."""
    tot, cnt = defaultdict(float), defaultdict(int)
    for obs, b in beliefs.items():
        for x, p in b.items():
            tot[x] += p
            cnt[x] += 1
    return {x: tot[x] / cnt[x] for x in tot}


def resolve_votes(votes: Dict[str, Optional[str]], agg: Dict[str, float], cfg):
    """Return (ejected_or_None, reason, tally).  Reasons for NO elimination:
         all_skip | skip_plurality | tie | margin | near_tie | below_threshold."""
    tally = Counter(v for v in votes.values() if v)
    skips = sum(1 for v in votes.values() if v is None)
    if not tally:
        return None, "all_skip", dict(tally)
    ranked = tally.most_common()
    top, tv = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if skips >= tv:
        return None, "skip_plurality", dict(tally)
    if tv == second:
        return None, "tie", dict(tally)
    if tv - second < cfg.vote_margin:
        return None, "margin", dict(tally)
    a = sorted(agg.values(), reverse=True)
    if len(a) > 1 and (a[0] - a[1]) < cfg.tie_epsilon and (tv - second) < 2:
        return None, "near_tie", dict(tally)
    if agg.get(top, 0.0) < cfg.agg_floor:
        return None, "below_threshold", dict(tally)
    return top, "plurality", dict(tally)
