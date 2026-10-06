"""
impostor_ai.py - the Impostor's brain.

Three interchangeable policies (Config.impostor_policy):
    "minimax" : the SMART impostor.  Chooses whom to kill, when, and how to escape using the
                Minimax-style evaluation below, vents only when unobserved, lies about its
                alibi in a way that cannot be refuted, and plants a false sighting.
    "greedy"  : DUMB baseline #1.  Hunts the nearest crewmate (A*), kills whenever alone with
                someone, runs to the nearest vent immediately (even if watched), lies naively.
    "random"  : DUMB baseline #2.  Wanders randomly, kills if it happens to be alone with someone.

======================================================================================
 MINIMAX-STYLE EVALUATION  (read this before the viva)
======================================================================================
THIS IS *NOT* A CLASSICAL TWO-PLAYER ZERO-SUM BOARD-GAME MINIMAX.
  * There is no alternating turn-taking on a shared board, and the crew is not a single
    rational opponent with a mirror-image objective.
  * What we borrow from Minimax is only the DECISION RULE  "maximise my value assuming the
    other side answers in the way that is worst for me" (a worst-case / robust choice).
  * The "opponent" is a SET OF CREW REACTIONS (how the crew will turn the evidence into
    suspicion).  The "payoff" is a hand-built UTILITY over SUSPICION, not a game score:

        U(target, reply, escape) = progress(target)
                                   - mm_lambda   * suspicion(target, reply, escape)
                                   - mm_travel_w * approach_cost(target)

SEARCH TREE (depth <= 3, branching ~ 7 x 4 x 3):
    ply 1  MAX  (Impostor)  : candidate action = KILL(target t)   (or LAY_LOW, utility 0)
    ply 2  MIN  (Crewmates) : reply r  in {PROXIMITY_ACCUSE, ALIBI_CROSSCHECK,
                                           BYSTANDER_TESTIMONY, IGNORE}
                              the crew picks the reply that RAISES suspicion most (lowest U)
    ply 3  MAX  (Impostor)  : escape f in {VENT, WALK, BLEND}  (picks the lowest suspicion)
    leaf   utility U above.

    value(t) = min over r  of  max over f  of  U(t, r, f)
    choose   t* = argmax_t value(t);   kill only if value(t*) > U(LAY_LOW), where
             U(LAY_LOW) = mm_lay_low_value - mm_urgency * (crew task progress)  (waiting gets
             more expensive the closer the crew is to winning by tasks).

    minimax_depth = 3 : full tree above
                  = 2 : escape fixed to WALK        -> value = min_r U(t, r, WALK)
                  = 1 : no adversary (reply IGNORE) -> value = U(t, IGNORE, WALK)

STATE / FEATURES used by the leaf evaluation (all computable from the visible world):
    K          kill room (victim's current/destination room)
    T_disc     ticks until the nearest OTHER crewmate can reach K (when the body is found)
    e(f)       ticks the impostor needs to execute escape f  (vent: walk to nearest vent + hop)
    p_caught   logistic((e - T_disc)/tau) : chance somebody walks in while I'm still at K
    prox(f)    1/(1+hops(final room, K)): crew blame whoever ends up closest to the body
    trail      fraction of crew who saw me recently (they can refute a fake alibi)
    with_victim 1 if another crewmate recently saw me together with the victim (feeds the crew's
               'last seen with the missing player' rule if the body is not found)
    nearby     fraction of crew within 2 hops of K (bystander testimony)
    progress   1/(alive_crew-1): share of the remaining kills this kill accomplishes
Suspicion = w_caught*p_caught + (reply term) - (blend bonus), clipped at 0.
======================================================================================
"""
import math
from collections import Counter
from typing import Dict, List, Optional

from search import crowd_averse_cost
from suspicion import Testimony

INF = float("inf")
RESPONSES = ("PROXIMITY_ACCUSE", "ALIBI_CROSSCHECK", "BYSTANDER_TESTIMONY", "IGNORE")
FOLLOWUPS = ("VENT", "WALK", "BLEND")


class ImpostorBrain:
    def __init__(self, game, me):
        self.game, self.me, self.cfg = game, me, game.cfg
        self.policy = self.cfg.impostor_policy
        self.smart = self.policy == "minimax"
        self.vent_safe = self.smart and self.cfg.imp_vent_safe
        self.seen_by: Dict[str, int] = {}     # crew name -> last tick we shared a room
        self.mode = "normal"                  # "normal" | "escape"
        self.escape_kind: Optional[str] = None
        self.pending_escape: Optional[str] = None
        self.kill_room: Optional[str] = None
        self.target: Optional[str] = None
        self.last_eval_tick = -99
        self.fake_room: Optional[str] = None
        self.fake_timer = 0
        self.ready_tick = 0
        self.last_tree: Optional[dict] = None
        self.last_frame: Optional[tuple] = None
        self.lay_low_count = 0

    # --------------------------------------------------------------- helpers
    def begin_round(self):
        self.mode = "normal"
        self.escape_kind = None
        self.target = None
        self.fake_room = None
        self.ready_tick = self.game.tick + self.cfg.initial_cooldown
        self.last_frame = None

    def _crew(self):
        return self.game.alive_crew()

    def _watched(self, room: str) -> bool:
        """Is any crewmate in `room`, or about to arrive there next tick?"""
        return any(c.room == room or (c.room is None and c.edge[1] == room and c.remaining <= 1)
                   for c in self._crew())

    def _replan(self, goal, purpose, allow_vents):
        g = self.game
        cost_fn = crowd_averse_cost(g.crowd_counts(), self.cfg.crowd_penalty) if self.smart else None
        g.plan_path(self.me, goal, purpose, allow_vents, cost_fn)

    def _follow(self, goal, purpose, allow_vents=None):
        """Return the next action towards `goal`, (re)planning with A* when needed."""
        me = self.me
        if allow_vents is None:
            allow_vents = self.smart
        if me.goal != goal or not me.plan:
            self._replan(goal, purpose, allow_vents)
        if not me.plan:
            return ("wait",)
        step = me.plan[0]
        if step[2] and self.vent_safe and (self._watched(me.room) or self._watched(step[0])):
            self._replan(goal, purpose, False)     # never vent in front of a witness
            if not me.plan:
                return ("wait",)
            step = me.plan[0]
        return ("move", step)

    # ------------------------------------------------------------- main hook
    def decide(self):
        g, me = self.game, self.me
        crew_here = [c for c in self._crew() if c.room == me.room]
        ready = g.tick >= self.ready_tick
        if self.mode == "escape":
            act = self._escape_step()
            if act:
                return act
        if ready and len(crew_here) == 1 and self._want_kill(crew_here[0]):
            return ("kill", crew_here[0].name)
        if self.policy == "random":
            return self._wander()
        if ready:
            tgt = self._pick_target()
            if tgt is not None:
                return self._pursue(tgt)
        return self._fake_task()

    # ------------------------------------------------------------ kill logic
    def _lay_low_value(self) -> float:
        """Utility of the root action LAY_LOW ("do not kill now").  Waiting is not free: the crew
        wins when its tasks are finished, so the value falls as task progress rises:
            U(lay_low) = mm_lay_low_value - mm_urgency * (fraction of crew tasks done)"""
        crew = self._crew()
        total = sum(len(c.tasks) for c in crew) or 1
        frac = sum(c.tasks_done() for c in crew) / total
        return self.cfg.mm_lay_low_value - self.cfg.mm_urgency * frac

    def _want_kill(self, victim) -> bool:
        if not self.smart:
            self.pending_escape = "VENT" if self.policy == "greedy" else None
            return True
        tree = self.evaluate(victim, self.me.room, 0.0)
        self.last_tree = tree
        if tree["value"] > self._lay_low_value():
            self.pending_escape = tree["escape"]
            self.game.note_minimax(tree, decision="KILL")
            return True
        self.lay_low_count += 1
        self.game.note_minimax(tree, decision="LAY_LOW (kill rejected)")
        return False

    def on_kill(self, victim, room):
        g = self.game
        self.kill_room = room
        self.ready_tick = g.tick + self.cfg.kill_cooldown
        self.target = None
        if self.policy == "random":
            self.mode = "normal"
        else:
            self.mode = "escape"
            self.escape_kind = self.pending_escape or "VENT"
        self.me.plan, self.me.goal = [], None

    # ------------------------------------------------------- target choice
    def _pick_target(self):
        g, me, m = self.game, self.me, self.game.map
        crew = self._crew()
        if not crew:
            return None
        valid = self.target in {c.name for c in crew}
        if valid and g.tick - self.last_eval_tick < 3:
            return g.agents[self.target]
        self.last_eval_tick = g.tick
        if self.policy == "greedy":
            best = None
            for c in crew:
                r = g.search(me, c.loc, "greedy_target", False)
                if best is None or r.cost < best[0]:
                    best = (r.cost, c)
            self.target = best[1].name
            return best[1]
        # --- minimax: evaluate every candidate with the 3-ply tree -------------------
        cost_fn = crowd_averse_cost(g.crowd_counts(), self.cfg.crowd_penalty)
        trees = []
        for c in crew:
            r = g.search(me, c.loc, "minimax_approach", True, cost_fn)
            trees.append(self.evaluate(c, c.loc, r.cost if r.found else 99.0))
        best = max(trees, key=lambda t: t["value"])
        g.note_minimax(best, decision="HUNT" if best["value"] > self._lay_low_value() else "LAY_LOW",
                       all_trees=trees)
        if best["value"] <= self._lay_low_value():
            self.target = None
            self.lay_low_count += 1
            return None
        self.target = best["target"]
        return g.agents[self.target]

    def _pursue(self, tgt):
        me = self.me
        goal = tgt.loc
        if me.room == goal:
            return ("wait",)
        return self._follow(goal, "pursue_target")

    # ---------------------------------------------------- idle behaviours
    def _fake_task(self):
        g, me = self.game, self.me
        if self.fake_room is None or (me.room == self.fake_room and self.fake_timer <= 0):
            rooms = [t.room for c in self._crew() for t in c.tasks if not t.done] or list(g.map.rooms)
            self.fake_room = g.rng.choice(rooms)
            self.fake_timer = self.cfg.task_duration
        if me.room == self.fake_room:
            self.fake_timer -= 1
            return ("wait",)
        return self._follow(self.fake_room, "fake_task")

    def _wander(self):
        g, me = self.game, self.me
        if me.goal is None or me.room == me.goal or not me.plan:
            others = [r for r in g.map.rooms if r != me.room]
            me.goal = None
            goal = g.rng.choice(others)
        else:
            goal = me.goal
        return self._follow(goal, "random_wander", False)

    # ------------------------------------------------------------- escape
    def _escape_step(self):
        g, me, m = self.game, self.me, self.game.map
        K = self.kill_room
        kind = self.escape_kind
        crew = self._crew()
        if kind == "VENT":
            vrooms = m.vent_rooms()
            if not vrooms:
                kind = self.escape_kind = "WALK"
            elif me.room in vrooms:
                exits = sorted(m.vent_adj[me.room])
                if self.vent_safe:
                    exits = [r for r in exits if not self._watched(r)]
                if self.vent_safe and (self._watched(me.room) or not exits):
                    kind = self.escape_kind = "BLEND"          # someone is watching: don't vent
                else:
                    if self.vent_safe:
                        ex = max(exits, key=lambda r: (m.hops(r, K), r))
                    else:
                        ex = g.rng.choice(exits)
                    step = (ex, m.vent_cost, True)
                    me.plan, me.goal = [step], ex
                    self.mode = "normal"
                    self.fake_room = None
                    return ("vent", step)
            else:
                v = min(vrooms, key=lambda r: m.tick_dist(me.room, r))
                return self._follow(v, "escape_to_vent", False)
        if kind == "WALK":
            if me.room != K:
                self.mode = "normal"
                return None
            others = crew
            best = max(m.adj[K], key=lambda n: min((m.tick_dist(c.loc, n) for c in others), default=99))
            return self._follow(best, "escape_walk", False)
        if kind == "BLEND":
            counts = g.crowd_counts()
            cand = [r for r, k in counts.items() if k > 0 and r != K]
            if not cand:
                self.mode = "normal"
                return None
            B = max(cand, key=lambda r: (counts[r], -m.tick_dist(me.room, r)))
            if me.room == B:
                self.mode = "normal"
                self.fake_room, self.fake_timer = B, self.cfg.task_duration
                return ("wait",)
            return self._follow(B, "escape_to_group", False)
        self.mode = "normal"
        return None

    # ================================================================
    #  MINIMAX-STYLE EVALUATION OF "kill `victim` in room K"
    # ================================================================
    def evaluate(self, victim, K: str, approach_cost: float, depth: Optional[int] = None) -> dict:
        g, m, cfg = self.game, self.game.map, self.cfg
        depth = depth or cfg.minimax_depth
        others = [c for c in self._crew() if c is not victim]
        n_alive = len(self._crew())
        progress = 1.0 / max(1, n_alive - 1)

        # ticks until the nearest OTHER crewmate can stand in K (= body discovered)
        def eta(c):
            return (c.remaining if c.room is None else 0) + m.tick_dist(c.loc, K)
        T_disc = min((eta(c) for c in others), default=INF)

        def p_caught(e: float) -> float:
            if T_disc == INF:
                return 0.0
            if e == INF:
                return 1.0
            x = max(-50.0, min(50.0, (T_disc - e) / cfg.mm_tau))
            return 1.0 / (1.0 + math.exp(x))          # large slack -> ~0, negative slack -> ~1

        # ---- ply-3 options: how the impostor can leave the scene ----------------------
        fol: Dict[str, dict] = {}
        vrooms = m.vent_rooms()
        if vrooms:
            v = min(vrooms, key=lambda r: m.tick_dist(K, r))
            if m.tick_dist(K, v) < INF:
                wait = cfg.mm_vent_wait if any(c.room == v for c in others) else 0
                exit_ = max(sorted(m.vent_adj[v]), key=lambda r: m.hops(r, K))
                fol["VENT"] = {"e": m.tick_dist(K, v) + 1 + wait, "final": exit_, "bonus": 0.0, "via": v}
        nb = sorted(m.adj[K])
        n_star = max(nb, key=lambda n: min((m.tick_dist(c.loc, n) for c in others), default=99))
        fol["WALK"] = {"e": m.edge_ticks(m.adj[K][n_star]), "final": n_star, "bonus": 0.0, "via": n_star}
        counts = g.crowd_counts(exclude=victim)
        cand = [r for r, k in counts.items() if k > 0 and r != K]
        if cand:
            B = max(cand, key=lambda r: (counts[r], -m.tick_dist(K, r)))
            fol["BLEND"] = {"e": m.tick_dist(K, B), "final": B, "via": B,
                            "bonus": cfg.mm_w_blend_bonus * min(1.0, counts[B] / max(1, len(others)))}

        # ---- crew-reply features -----------------------------------------------------
        recent = [c for c in others if self.seen_by.get(c.name, -99) >= g.tick - cfg.mm_trail_window]
        trail = len(recent) / max(1, len(others))
        nearby = sum(1 for c in others if m.hops(c.loc, K) <= 2) / max(1, len(others))
        # mirrors the crew's 'last seen with the missing player' rule: did any OTHER living
        # crewmate see me together with this victim recently?  (impostor sees g.occ)
        alive_names = {c.name for c in others}
        with_victim = 0.0
        for t in range(g.tick - cfg.mm_trail_window, g.tick + 1):
            for room, names in g.occ.get(t, {}).items():
                if self.me.name in names and victim.name in names and alive_names.intersection(names):
                    with_victim = 1.0

        def suspicion(reply: str, f: dict) -> float:
            base = cfg.mm_w_caught * p_caught(f["e"])
            prox = 1.0 / (1.0 + m.hops(f["final"], K))
            if reply == "PROXIMITY_ACCUSE":
                extra = cfg.mm_w_prox_acc * prox
            elif reply == "ALIBI_CROSSCHECK":
                extra = cfg.mm_w_alibi_chk * trail + cfg.mm_w_last_seen * with_victim
            elif reply == "BYSTANDER_TESTIMONY":
                extra = cfg.mm_w_witness * nearby
            else:
                extra = 0.0
            return max(0.0, base + extra - f["bonus"])

        def utility(s: float) -> float:
            return progress - cfg.mm_lambda * s - cfg.mm_travel_w * approach_cost

        # ---- the tree --------------------------------------------------------------------
        replies = {}
        if depth >= 3:
            for r in RESPONSES:
                opts = {f: {"suspicion": round(suspicion(r, fol[f]), 4),
                            "U": round(utility(suspicion(r, fol[f])), 4)} for f in fol}
                bf = max(opts, key=lambda f: opts[f]["U"])           # ply 3: impostor maximises
                replies[r] = {"best_escape": bf, "U": opts[bf]["U"], "options": opts}
        elif depth == 2:
            for r in RESPONSES:
                s = suspicion(r, fol["WALK"])
                replies[r] = {"best_escape": "WALK", "U": round(utility(s), 4),
                              "options": {"WALK": {"suspicion": round(s, 4), "U": round(utility(s), 4)}}}
        else:
            s = suspicion("IGNORE", fol["WALK"])
            replies["IGNORE"] = {"best_escape": "WALK", "U": round(utility(s), 4),
                                 "options": {"WALK": {"suspicion": round(s, 4), "U": round(utility(s), 4)}}}
        worst = min(replies, key=lambda r: replies[r]["U"])           # ply 2: crew minimises
        return {"tick": g.tick, "target": victim.name, "kill_room": K, "depth": depth,
                "progress": round(progress, 4), "approach_cost": round(approach_cost, 3),
                "T_disc": T_disc, "trail": round(trail, 3), "nearby": round(nearby, 3), "with_victim": with_victim,
                "followups": {f: {"e": v["e"], "final": v["final"]} for f, v in fol.items()},
                "replies": replies, "worst_reply": worst,
                "escape": replies[worst]["best_escape"], "value": replies[worst]["U"]}

    # ================================================================
    #  Discussion behaviour
    # ================================================================
    def make_testimony(self) -> Testimony:
        g, me = self.game, self.me
        claims = dict(me.track)
        sightings = list(me.seen)
        kills = g.round_kills
        if kills and (self.cfg.imp_lie or not self.smart):
            replaced = self._lie_about_alibi(claims)
            sightings = [s for s in sightings if s[0] not in replaced]
            if self.smart and self.cfg.imp_frame and g.rng.random() < self.cfg.frame_probability:
                self._frame(claims, sightings)
        return Testimony(me.name, claims, sightings, [])

    def _window_ticks(self, claims) -> List[int]:
        """Ticks whose truth the impostor would rather not tell: just before/after every
        kill and around every vent hop (a vent hop would otherwise show up as impossible travel)."""
        g = self.game
        out = set()
        for (kt, K, v) in g.round_kills:
            out.update(t for t in claims if kt - self.cfg.alibi_lookback <= t <= kt + self.cfg.alibi_escape_window)
        for vt in g.round_vents:
            out.update(t for t in claims if vt - self.cfg.alibi_lookback <= t <= vt + self.cfg.alibi_escape_window)
        return sorted(out)

    def _lie_about_alibi(self, claims) -> set:
        g, m, me = self.game, self.game.map, self.me
        alive = {c.name for c in self._crew()}
        win = self._window_ticks(claims)
        if not self.smart:
            # naive liar: "I was in room X the whole time" for one random X, with NO check that X was
            # reachable in time, empty of crew, or consistent with what other people saw
            decoy = g.rng.choice(m.rooms)
            for t in win:
                claims[t] = decoy
            return set(win)
        # Smart liar.  Only lie about ticks nobody can vouch for (no living crew shared my room),
        # and pick the fake positions with a tiny dynamic programme over (tick, room):
        #  (1) a fake room must have been EMPTY of living crew at that tick (nobody can say
        #      "I was there and you were not"),
        #  (2) consecutive claims - including the honest claims on either side - must be reachable
        #      in time along corridors (no "impossible travel"),
        #  (3) prefer rooms far from the body, and prefer not to keep changing rooms.
        def observed(t):
            return any(n in alive for n in g.occ.get(t, {}).get(me.track[t], []) if n != me.name)
        rset = {t for t in win if not observed(t)}
        if g.round_kills:
            K0 = g.round_kills[0][1]
        elif g.round_vents:
            K0 = me.track.get(min(g.round_vents), m.hub)
        else:
            K0 = m.hub
        ticks = sorted(claims)
        runs, cur = [], []
        for t in ticks:
            if t in rset:
                cur.append(t)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        done = set()
        for run in runs:
            t0, t1 = run[0], run[-1]
            prev = [t for t in claims if t < t0 and t not in rset]
            prev_anchor = (max(prev), claims[max(prev)]) if prev else (g.round_start, m.hub)
            nxt = [t for t in claims if t > t1 and t not in rset]
            nxt_anchor = (min(nxt), claims[min(nxt)]) if nxt else None
            seq = self._decoy_sequence(run, prev_anchor, nxt_anchor, alive, K0)
            if seq:
                for t, R in zip(run, seq):
                    claims[t] = R
                done.update(run)
            else:
                g.stats["alibi_lie_failed"] += 1
        return done

    def _decoy_sequence(self, run, prev_anchor, nxt_anchor, alive, K0):
        """DP: best feasible list of fake rooms, one per tick in `run` (None if none exists)."""
        g, m, me = self.game, self.game.map, self.me
        rooms = m.rooms
        NEG = -1e9
        best = []                                  # best[i][R] = (score, back-pointer room)
        for i, t in enumerate(run):
            row = {}
            for R in rooms:
                if any(n in alive and n != me.name for n in g.occ.get(t, {}).get(R, [])):
                    continue
                gain = m.hops(R, K0)
                if i == 0:
                    if m.tick_dist(prev_anchor[1], R) <= t - prev_anchor[0]:
                        row[R] = (gain - (0 if R == prev_anchor[1] else 0.5), None)
                else:
                    dt = t - run[i - 1]
                    cand = [(best[i - 1][Q][0] + gain - (0 if Q == R else 0.5), Q)
                            for Q in best[i - 1] if m.tick_dist(Q, R) <= dt]
                    if cand:
                        row[R] = max(cand)
            best.append(row)
        last = {R: v for R, v in best[-1].items()
                if not nxt_anchor or m.tick_dist(R, nxt_anchor[1]) <= nxt_anchor[0] - run[-1]}
        if not last:
            return None
        R = max(last, key=lambda r: (last[r][0], r))
        seq = [R]
        for i in range(len(run) - 1, 0, -1):
            R = best[i][R][1]
            seq.append(R)
        return seq[::-1]

    def _frame(self, claims, sightings):
        """Misdirection: plant ONE false sighting of an innocent crewmate near the kill room,
        just before the kill, at a moment when nobody could contradict it."""
        g, m, me = self.game, self.game.map, self.me
        kt, K, _ = g.round_kills[0]
        alive = {c.name for c in self._crew()}
        ticks = [t for t in me.track if kt - 8 <= t < kt and claims.get(t) == me.track[t]]
        if not ticks:
            return
        tf = min(ticks, key=lambda t: (m.hops(me.track[t], K), -t))
        Rf = me.track[tf]
        scored = []
        for z in self._crew():
            zt = z.track.get(tf)
            if zt == Rf:
                continue
            if zt is None:
                score = 2                              # in a corridor: has no claim to contradict
            else:
                company = [n for n in g.occ.get(tf, {}).get(zt, []) if n != z.name and n in alive]
                score = 0 if company else 1            # alone in a room: nobody corroborates z
            if score:
                scored.append((score, g.rng.random(), z.name))
        if not scored:
            return
        score, _, zname = max(scored)
        sightings.append((tf, Rf, zname))
        self.last_frame = (tf, Rf, zname)
        g.stats["frames_planted"] += 1

    def choose_vote(self, votes: Dict[str, Optional[str]], agg: Dict[str, float]) -> Optional[str]:
        g, me = self.game, self.me
        others = [a.name for a in g.alive_agents() if a is not me]
        if not self.smart or not self.cfg.imp_bandwagon:
            return g.rng.choice(others)
        # bandwagon: vote with the leading crew vote (avoids the 'dissenter' penalty and
        # pushes the crew towards ejecting an innocent).  If the crew is about to pick me,
        # vote for the runner-up instead; never vote for myself.
        tally = Counter(v for v in votes.values() if v and v != me.name)
        if tally:
            return tally.most_common(1)[0][0]
        cand = sorted((x for x in agg if x != me.name), key=lambda x: -agg[x])
        return cand[0] if cand else g.rng.choice(others)
