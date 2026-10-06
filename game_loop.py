"""
game_loop.py - the simulation engine: rounds = TASK phase -> (DISCOVERY) -> MEETING -> RESOLUTION.

Time model: one tick = one step of every agent.  An agent is either IN a room or travelling
along a corridor/vent edge (edge of cost c takes ceil(c/edge_speed) ticks, a vent hop 1 tick).
Agents see each other only when they are in the same room at the end of a tick.

Phase 1 TASK/MOVEMENT : crewmates A*-path to their next task and work on it; the impostor
                        hunts/escapes/fakes tasks; it may kill when alone with a crewmate.
Phase 2 DISCOVERY     : a crewmate ending a tick in a room with an unreported body reports it
                        -> the phase ends early.  (Otherwise a meeting is forced after round_ticks.)
Phase 3 MEETING       : testimonies are public; every crewmate updates its belief (suspicion.py)
                        and votes.  Everyone is then moved back to the hub.
Phase 4 RESOLUTION    : plurality eject (or nobody), dissent update, win-condition check.

WIN CONDITIONS
  crew     : the impostor is ejected, OR every LIVING crewmate has finished all its tasks
             (tasks of dead crewmates no longer count - there are no ghosts in this model)
  impostor : living crew <= living impostors (1), i.e. the crew can no longer outvote it.
"""
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from agents import Agent, Task, PALETTE
from config import Config
from impostor_ai import ImpostorBrain
from search import a_star, bfs, dijkstra, SearchLog
from ship_map import build_map
import suspicion as sus


@dataclass
class GameResult:
    cfg: Config
    winner: Optional[str]                 # "crew" | "impostor" | None (timeout)
    reason: str
    rounds: int
    ticks: int
    impostor: str
    events: List[str]
    meetings: List[dict]
    move_log: List[dict]
    search_log: SearchLog
    round_times: List[dict]
    stats: dict
    minimax_log: List[dict]
    frames: List[dict]
    map: object
    agent_colors: Dict[str, str]
    agent_order: List[str]
    chat: List[tuple] = field(default_factory=list)


class Game:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.map = build_map(cfg)
        t0 = time.perf_counter()
        for av in (False, True):                       # pre-compute the distance tables once, so the
            self.map.dist(self.map.hub, self.map.hub, av)    # per-tick timings measure the simulation,
            self.map.hops(self.map.hub, self.map.hub, av)    # not a one-off Floyd-Warshall
            self.map.tick_dist(self.map.hub, self.map.hub, av)
        self.setup_s = time.perf_counter() - t0
        self.search_log = SearchLog()
        self.tick = 0
        self.round = 0
        self.round_start = 0
        self.events: List[str] = []
        self.chat: List[tuple] = []                 # (speaker, text) public chat, for the animation
        self._chatted: set = set()
        self.move_log: List[dict] = []
        self.meetings: List[dict] = []
        self.minimax_log: List[dict] = []
        self.frames: List[dict] = []
        self.round_times: List[dict] = []
        self.stats = defaultdict(int)
        self.winner: Optional[str] = None
        self.win_reason = ""
        self.occ: Dict[int, Dict[str, List[str]]] = {}
        self.round_kills: List[tuple] = []          # (tick, room, victim)
        self.round_vents: List[int] = []            # ticks at which the impostor hopped through a vent
        self.bodies: List[dict] = []                # {room, victim, tick, reported}
        self.vent_arrivals: List[str] = []
        self.round_roster: List[str] = []
        self.meeting_missing: List[str] = []
        self.last_beliefs: Optional[dict] = None
        self.last_agg: Optional[dict] = None
        self._setup_agents()
        self.brain = ImpostorBrain(self, self.impostor)

    # ------------------------------------------------------------------ setup
    def _setup_agents(self):
        cfg = self.cfg
        n = cfg.n_crew + 1
        palette = PALETTE[:n] if n <= len(PALETTE) else PALETTE + [(f"P{i}", "#555555") for i in range(n - len(PALETTE))]
        self.agents: Dict[str, Agent] = {}
        imp_idx = self.rng.randrange(n)
        for i, (name, color) in enumerate(palette):
            self.agents[name] = Agent(name, color, "impostor" if i == imp_idx else "crew")
        self.impostor = next(a for a in self.agents.values() if a.role == "impostor")
        rooms = self.map.rooms
        for a in self.agents.values():
            a.room = self.map.hub
            if a.role == "crew":
                last = None
                for _ in range(cfg.tasks_per_crew):
                    r = self.rng.choice([x for x in rooms if x != last])
                    a.tasks.append(Task(r, cfg.task_duration))
                    last = r
                a.gullibility = min(1.5, max(0.5, self.rng.gauss(1.0, cfg.gullibility_sd)))
        names = list(self.agents)
        for a in self.agents.values():
            if a.role == "crew":
                a.belief = {x: 1.0 / (n - 1) for x in names if x != a.name}

    # ---------------------------------------------------------------- queries
    def alive_agents(self):
        return [a for a in self.agents.values() if a.alive]

    def alive_crew(self):
        return [a for a in self.agents.values() if a.alive and a.role == "crew"]

    def crowd_counts(self, exclude=None) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for a in self.alive_crew():
            if a is not exclude and a.room is not None:
                c[a.room] += 1
        return dict(c)

    # ---------------------------------------------------------------- logging
    def log(self, msg: str):
        line = f"[t={self.tick:03d} R{self.round}] {msg}"
        self.events.append(line)
        if self.cfg.verbose:
            print(line)

    def log_move(self, a: Agent, ev: str, src, dst):
        self.move_log.append({"tick": self.tick, "round": self.round, "agent": a.name, "role": a.role,
                              "event": ev, "from": src, "to": dst})

    def note_minimax(self, tree: dict, decision: str, all_trees: Optional[List[dict]] = None):
        entry = {"tick": self.tick, "round": self.round, "decision": decision, "chosen": tree}
        if all_trees:
            entry["candidates"] = [{"target": t["target"], "value": t["value"], "worst_reply": t["worst_reply"],
                                    "escape": t["escape"]} for t in all_trees]
        self.minimax_log.append(entry)

    # ---------------------------------------------------------------- A* glue
    def search(self, agent, goal, purpose, allow_vents=False, cost_fn=None):
        start = agent.room
        res = a_star(self.map, start, goal, self.cfg.heuristic, cost_fn, allow_vents)
        b = d = None
        if self.cfg.compare_baselines:
            b = bfs(self.map, start, goal, cost_fn, allow_vents)
            d = dijkstra(self.map, start, goal, cost_fn, allow_vents)
        self.search_log.add(self.tick, self.round, agent.name, agent.role, purpose, allow_vents, res, b, d)
        return res

    def plan_path(self, agent, goal, purpose, allow_vents=False, cost_fn=None):
        res = self.search(agent, goal, purpose, allow_vents, cost_fn)
        agent.goal = goal
        agent.plan = list(res.steps)
        return res

    # ---------------------------------------------------------------- movement
    def depart(self, a: Agent, step):
        dst, cost, is_vent = step
        src = a.room
        ticks = self.map.edge_ticks(cost, is_vent)
        a.edge = (src, dst, is_vent, ticks)
        a.remaining = ticks
        a.room = None
        self.log_move(a, "vent" if is_vent else "depart", src, dst)
        if is_vent:
            self.stats["vent_hops"] += 1
            self.round_vents.append(self.tick)
            self._vent_witness(a, src, self.tick)
        self.advance(a)

    def advance(self, a: Agent):
        a.remaining -= 1
        if a.remaining <= 0:
            src, dst, is_vent, _ = a.edge
            a.room, a.edge, a.remaining = dst, None, 0
            self.log_move(a, "arrive", src, dst)
            if is_vent:
                self.vent_arrivals.append(a.name)

    def _vent_witness(self, imp: Agent, room: str, tick: int):
        """Crewmates standing in `room` when the impostor vents see it happen."""
        for c in self.alive_crew():
            if c.room == room:
                c.vent_seen.append((tick, room, imp.name))
                self.stats["vent_witnessed"] += 1
                self.log(f"{c.name} SAW {imp.name} use a vent in {room}")

    def agent_step(self, a: Agent):
        if a.remaining > 0:
            self.advance(a)
            return
        if a.role == "crew":
            self.crew_step(a)
        else:
            self.impostor_step(a)

    def crew_step(self, a: Agent):
        task = a.current_task()
        if task is None:
            return                                     # all tasks done: stand still
        if a.room == task.room:
            task.progress += 1
            if task.progress >= task.duration:
                task.done = True
                a.plan, a.goal = [], None
                self.log_move(a, "task_done", a.room, a.room)
                self.log(f"{a.name} finished a task in {a.room} ({a.tasks_done()}/{len(a.tasks)})")
            return
        if a.goal != task.room or not a.plan:
            self.plan_path(a, task.room, "task")
        if a.plan:
            self.depart(a, a.plan.pop(0))

    def impostor_step(self, a: Agent):
        act = self.brain.decide()
        kind = act[0]
        if kind == "kill":
            self.do_kill(a, self.agents[act[1]])
        elif kind in ("move", "vent"):
            step = act[1]
            if a.plan and a.plan[0] == step:
                a.plan.pop(0)
            self.depart(a, step)
        # "wait": do nothing this tick

    def do_kill(self, imp: Agent, victim: Agent):
        victim.alive = False
        victim.plan = []
        room = imp.room
        self.bodies.append({"room": room, "victim": victim.name, "tick": self.tick, "reported": False})
        self.round_kills.append((self.tick, room, victim.name))
        self.stats["kills"] += 1
        self.log_move(imp, "kill", room, room)
        self.log(f"KILL: {imp.name} (impostor) killed {victim.name} in {room}")
        self.brain.on_kill(victim, room)

    # ------------------------------------------------------------- observation
    def observe(self):
        occ: Dict[str, List[Agent]] = defaultdict(list)
        for a in self.alive_agents():
            if a.room is not None:
                occ[a.room].append(a)
        self.occ[self.tick] = {r: [x.name for x in g] for r, g in occ.items()}
        for room, group in occ.items():
            for a in group:
                a.track[self.tick] = room
                for b in group:
                    if b is a:
                        continue
                    if a.role == "impostor":
                        a.seen.append((self.tick, room, b.name))
                    elif self.rng.random() < self.cfg.memory_prob:
                        a.seen.append((self.tick, room, b.name))
                    if b.role == "impostor":
                        self.brain.seen_by[a.name] = self.tick
                    if a.role == "crew" and (a.name, b.name, self.round) not in self._chatted:
                        self._chatted.add((a.name, b.name, self.round))
                        self.chat.append((a.name, f"{b.name} is here in {room} with me (t{self.tick})."))
        # someone who emerged from a vent in a room full of people was seen
        for name in self.vent_arrivals:
            imp = self.agents[name]
            if imp.room is not None:
                for c in self.alive_crew():
                    if c.room == imp.room:
                        c.vent_seen.append((self.tick, imp.room, imp.name))
                        self.stats["vent_witnessed"] += 1
                        self.log(f"{c.name} SAW {imp.name} pop out of a vent in {imp.room}")
        self.vent_arrivals = []

    def check_discovery(self) -> Optional[dict]:
        for b in self.bodies:
            if b["reported"]:
                continue
            finders = [c for c in self.alive_crew() if c.room == b["room"]]
            if finders:
                f = self.rng.choice(finders)
                b["reported"] = True
                b["finder"] = f.name
                b["found_tick"] = self.tick
                return b
        return None

    # -------------------------------------------------------------- win checks
    def check_wins(self) -> bool:
        crew = self.alive_crew()
        if len(crew) <= 1:
            self.finish("impostor", "crew can no longer outvote the impostor")
            return True
        if all(c.current_task() is None for c in crew):
            self.finish("crew", "all living crewmates completed their tasks")
            return True
        return False

    def finish(self, winner, reason):
        self.winner, self.win_reason = winner, reason
        self.log(f"GAME OVER: {winner or 'nobody'} wins - {reason}")

    # ------------------------------------------------------------------- chat
    def say(self, speaker: str, text: str, frame: bool = True):
        """Public chat line.  Purely cosmetic: built from testimony, never touches the RNG."""
        self.chat.append((speaker, f"[R{self.round}] " + text if speaker == "" else text))
        if frame:
            self.snapshot("MEETING", "discussion")

    @staticmethod
    def _claim_story(claims: Dict[int, str], lo: int) -> str:
        runs = []
        for t in sorted(claims):
            if t < lo:
                continue
            if not runs or runs[-1][0] != claims[t]:
                runs.append([claims[t], t])
        runs = runs[-4:]
        return ", then ".join(f"{r} (t{t})" for r, t in runs) or "walking around"

    def _discuss(self, tests, body, ev, imp_name):
        rng_names = [a.name for a in self.alive_agents()]
        if body:
            self.say("", f"== MEETING: {body['victim']}'s body found in {body['room']} by {body['finder']} ==")
            self.say(body["finder"], f"I found {body['victim']}'s body in {body['room']}! Where was everyone?")
        else:
            self.say("", "== EMERGENCY MEETING (round timer ran out, no body) ==")
        for n in rng_names:
            T = tests[n]
            story = self._claim_story(T.claims, self.round_start)
            self.say(n, f"I was in: {story}.")
            for (t, room, who) in T.sightings[-2:]:
                self.say(n, f"I saw {who} in {room} at t{t}.", frame=False)
        for cf in ev.conflicts:
            if cf.kind == "seen_elsewhere":
                room = cf.detail.rsplit(" in ", 1)[1]
                self.say(cf.reporter, f"{cf.claimant} claims otherwise, but I saw {cf.claimant} in {room} at t{cf.tick}!")
            elif cf.kind == "absent":
                self.say(cf.reporter, f"{cf.claimant}? I was there at t{cf.tick} and did not see you.")
            elif cf.kind == "impossible":
                self.say("", f"{cf.detail} - impossible without a shortcut!")
            elif cf.kind == "self_incons":
                self.say("", cf.detail)
        if body:
            near = sorted(ev.prox.items(), key=lambda kv: -kv[1])[:2]
            for x, p in near:
                if x != body["finder"] and p >= 0.5:
                    self.say(body["finder"], f"{x} was close to {body['room']}...")

    # -------------------------------------------------------------------- frames
    def _xy(self, a: Agent):
        m = self.map
        if a.room is not None:
            return m.pos[a.room]
        src, dst, _, total = a.edge
        f = 1.0 - a.remaining / total
        (x1, y1), (x2, y2) = m.pos[src], m.pos[dst]
        return (x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)

    def snapshot(self, phase: str, label: str = ""):
        if not self.cfg.record_frames:
            return
        self.frames.append({
            "tick": self.tick, "round": self.round, "phase": phase, "label": label,
            "pos": {a.name: self._xy(a) for a in self.agents.values()},
            "alive": {a.name: a.alive for a in self.agents.values()},
            "bodies": [b["room"] for b in self.bodies],
            "log": self.events[-14:], "chat": list(self.chat[-14:]), "chat_n": len(self.chat), "log_n": len(self.events),
            "missing": list(self.meeting_missing) if phase == "MEETING" else [],
            "beliefs": self.last_beliefs, "agg": self.last_agg,
            "votes": None, "ejected": None,
        })

    # ------------------------------------------------------------------- rounds
    def begin_round(self):
        self.round += 1
        self.round_start = self.tick
        self.occ = {}
        self.round_kills = []
        self.round_vents = []
        # unreported bodies stay where they fell and can still be found in a later round
        self.bodies = [b for b in self.bodies if not b["reported"]]
        self.round_roster = [a.name for a in self.alive_agents()]   # who the crew saw at round start
        self.meeting_missing = []
        for a in self.alive_agents():
            a.room, a.edge, a.remaining = self.map.hub, None, 0
            a.clear_round_memory()
        self.brain.begin_round()
        self.log(f"--- ROUND {self.round} begins ({len(self.alive_crew())} crew alive) ---")

    def task_phase(self):
        for _ in range(self.cfg.round_ticks):
            self.tick += 1
            order = self.alive_agents()
            self.rng.shuffle(order)
            for a in order:
                if a.alive:
                    self.agent_step(a)
            self.observe()
            if self.check_wins():
                self.snapshot("TASK")
                return "end", None
            body = self.check_discovery()
            if body:
                self.log(f"BODY REPORTED: {body['finder']} found {body['victim']}'s body in {body['room']}")
                present = [n for n in self.occ[self.tick].get(body["room"], [])]
                if self.impostor.alive and self.impostor.room == body["room"]:
                    self.stats["caught_at_body"] += 1
                    if body["room"] in self.map.trap_rooms():
                        self.stats["trapped_in_dead_end"] += 1
                        body["trapped"] = True
                        self.log(f"!! impostor caught standing in dead-end room {body['room']} (no vent)")
                self.snapshot("TASK", "BODY REPORTED")
                return "body", body
            self.snapshot("TASK")
        self.log("Round timer expired -> emergency meeting")
        return "timer", None

    def meeting_phase(self, reason: str, body: Optional[dict]):
        cfg = self.cfg
        crew = self.alive_crew()
        imp = self.impostor
        dead_now = [v for (_, _, v) in self.round_kills]          # ground truth, analysis only
        self.snapshot("MEETING", "meeting called: " + ("body reported" if body else "timer"))

        # ---- roster check (public): who was at the start of the round but is not here now? -----
        # Absence is the only way to miss a meeting, so an absent player is dead and therefore
        # cannot be the impostor: their probability mass is removed EXPLICITLY, on its own frame.
        gone = [n for n in self.round_roster if not self.agents[n].alive]
        missing = [n for n in gone if not (body and n == body["victim"])]
        self.meeting_missing = missing
        for n in missing:
            self.log(f"ROSTER: {n} is missing (presumed dead, body not found)")
            self.say("", f"== Roster check: {n} did not show up - presumed dead ==")
        for c in crew:
            c.belief = sus.remove_agents(c.belief, gone)
        self.last_beliefs = {c.name: dict(c.belief) for c in crew}
        self.last_agg = sus.aggregate(self.last_beliefs)
        self.snapshot("MEETING", "roster check: " + (", ".join(missing) + " missing" if missing else "nobody missing"))

        # ---- testimonies (public) -------------------------------------------------
        tests: Dict[str, sus.Testimony] = {}
        for a in self.alive_agents():
            if a.role == "crew":
                tests[a.name] = sus.Testimony(a.name, dict(a.track), list(a.seen), list(a.vent_seen))
            else:
                tests[a.name] = self.brain.make_testimony()
        body_obj = sus.Body(body["room"], body["victim"], body["found_tick"], body["finder"]) if body else None
        suspects = [a.name for a in self.alive_agents()]
        ev = sus.build_evidence(tests, body_obj, self.map, cfg, self.round_start, self.tick, suspects, missing)

        self._discuss(tests, body, ev, imp.name)

        # ---- belief update ----------------------------------------------------------
        breakdowns = {}
        for c in crew:
            c.belief, breakdowns[c.name] = sus.update_belief(c.name, c.belief, ev, cfg, c.gullibility)
        beliefs = {c.name: dict(c.belief) for c in crew}
        agg = sus.aggregate(beliefs)
        self.last_beliefs, self.last_agg = beliefs, agg
        conf_by = defaultdict(int)
        for cf in ev.conflicts:
            conf_by[cf.claimant] += 1
        for cf in ev.conflicts:
            self.log(f"ALIBI CONFLICT [{cf.kind}] {cf.detail}")
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:3]
        self.log("aggregated suspicion: " + ", ".join(f"{n}={p:.2f}" for n, p in top))
        self.snapshot("MEETING", "testimonies heard, beliefs updated")

        # ---- voting ----------------------------------------------------------------------
        votes: Dict[str, Optional[str]] = {c.name: sus.decide_vote(c.belief, cfg) for c in crew}
        votes[imp.name] = self.brain.choose_vote(votes, agg)
        ejected, why, tally = sus.resolve_votes(votes, agg, cfg)
        for k, v in votes.items():
            self.say(k, f"I vote {v}." if v else "I skip.")
        self.say("", f"Result: {('EJECT ' + ejected) if ejected else 'no elimination (' + why + ')'}")
        self.log("votes: " + ", ".join(f"{k}->{v or 'SKIP'}" for k, v in votes.items()))
        self.log(f"tally: {tally}  => {('EJECT ' + ejected) if ejected else 'NO ELIMINATION'} ({why})")

        rec = {"round": self.round, "tick": self.tick, "reason": reason,
               "body": ({k: body[k] for k in ("room", "victim", "finder", "found_tick")} if body else None),
               "trapped": bool(body and body.get("trapped")),
               "conflicts": [asdict(c) for c in ev.conflicts],
               "conflicts_by_claimant": dict(conf_by),
               "prox": ev.prox, "scene": ev.scene, "group": ev.group, "vent": ev.vent,
               "breakdown": breakdowns, "beliefs": beliefs, "agg": agg, "votes": votes,
               "tally": tally, "ejected": ejected, "outcome_reason": why,
               "impostor": imp.name, "alive_before": [a.name for a in self.alive_agents()],
               "frame": self.brain.last_frame, "killed": dead_now, "missing": missing,
               "last_with": ev.last_with}

        # ---- resolution ----------------------------------------------------------------
        f_snap = {"votes": votes, "ejected": ejected}
        if ejected:
            e = self.agents[ejected]
            e.alive = False
            rec["ejected_role"] = e.role
            if e.role == "crew":
                self.stats["innocents_ejected"] += 1
            self.log(f"{ejected} was EJECTED ({'IMPOSTOR' if e.role == 'impostor' else 'innocent crewmate'})")
            dissenters = [k for k, v in votes.items() if v and v != ejected]
            for c in self.alive_crew():
                c.belief = sus.remove_agents(c.belief, [ejected])
                c.belief = sus.apply_dissent(c.belief, [d for d in dissenters if d != c.name], cfg)
            rec["dissenters"] = dissenters
        else:
            self.stats["no_elimination"] += 1
            if why in ("tie", "margin", "near_tie"):
                self.stats["tie_votes"] += 1
        self.meetings.append(rec)
        if cfg.record_frames:
            self.snapshot("MEETING", "VOTE RESULT: " + (f"{ejected} ejected" if ejected else f"no elimination ({why})"))
            self.frames[-1].update(f_snap)
            self.frames[-1]["beliefs"] = {c.name: dict(c.belief) for c in self.alive_crew()}
            self.frames[-1]["agg"] = sus.aggregate(self.frames[-1]["beliefs"]) if self.alive_crew() else {}
            self.last_beliefs, self.last_agg = self.frames[-1]["beliefs"], self.frames[-1]["agg"]
        if ejected and self.agents[ejected].role == "impostor":
            self.finish("crew", "impostor was voted out")
        elif len(self.alive_crew()) <= 1:
            self.finish("impostor", "crew can no longer outvote the impostor")

    # -------------------------------------------------------------------- run
    def run(self) -> GameResult:
        while self.winner is None and self.round < self.cfg.max_rounds:
            t0 = time.perf_counter()
            self.begin_round()
            t_start = self.tick
            kind, body = self.task_phase()
            t1 = time.perf_counter()
            if self.winner is None:
                self.meeting_phase(kind, body)
            t2 = time.perf_counter()
            self.round_times.append({
                "round": self.round, "ticks": self.tick - t_start, "task_s": t1 - t0, "meeting_s": t2 - t1,
                "total_s": t2 - t0, "agents": len(self.agents), "rooms": len(self.map.rooms),
                "branching": self.map.avg_branching(), "alive": len(self.alive_agents())})
        if self.winner is None:
            self.finish(None, f"no winner after {self.cfg.max_rounds} rounds")
        s = self.stats
        s["vent_escapes_unseen"] = max(0, s["vent_hops"] - s["vent_witnessed"])
        s["setup_ms"] = round(self.setup_s * 1000, 2)
        return GameResult(self.cfg, self.winner, self.win_reason, self.round, self.tick, self.impostor.name,
                          self.events, self.meetings, self.move_log, self.search_log, self.round_times,
                          dict(s), self.minimax_log, self.frames, self.map,
                          {a.name: a.color for a in self.agents.values()}, list(self.agents), self.chat)


def run_game(cfg: Config) -> GameResult:
    return Game(cfg).run()
