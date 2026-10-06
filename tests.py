"""
tests.py - sanity checks for the claims made in the report.   Run:  python tests.py
"""
import math
import unittest

from config import Config
from game_loop import run_game
from search import HEURISTICS, a_star, bfs, dijkstra, crowd_averse_cost
from ship_map import default_map, generate_map
import suspicion as sus

ADMISSIBLE = ["zero", "euclidean", "perfect"]


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.maps = [default_map(), generate_map(14, 1), generate_map(25, 2)]

    def test_astar_matches_dijkstra_for_admissible_heuristics(self):
        for g in self.maps:
            for av in (False, True):
                for a in g.rooms:
                    for b in g.rooms:
                        opt = dijkstra(g, a, b, None, av).cost
                        for h in ADMISSIBLE + ([] if av else ["euclidean_raw"]):
                            self.assertAlmostEqual(a_star(g, a, b, h, None, av).cost, opt, places=6, msg=(g.name, a, b, h, av))

    def test_heuristics_are_admissible_and_consistent(self):
        for g in self.maps:
            for av in (False, True):
                for h in ADMISSIBLE:
                    fn = HEURISTICS[h].fn
                    for a in g.rooms:
                        for b in g.rooms:
                            self.assertLessEqual(fn(g, a, b, av), g.dist(a, b, av) + 1e-9)
                        for n, c, _ in g.neighbors(a, av):
                            for goal in g.rooms:   # consistency: h(a) <= c(a,n) + h(n)
                                self.assertLessEqual(fn(g, a, goal, av), c + fn(g, n, goal, av) + 1e-9)

    def test_inadmissible_heuristic_can_be_suboptimal(self):
        g = self.maps[1]
        worse = 0
        for a in g.rooms:
            for b in g.rooms:
                opt = dijkstra(g, a, b).cost
                r = a_star(g, a, b, "weighted")
                self.assertGreaterEqual(r.cost, opt - 1e-9)
                worse += r.cost > opt + 1e-9
        self.assertGreaterEqual(worse, 0)

    def test_astar_never_expands_more_than_dijkstra_on_average(self):
        g = self.maps[2]
        ea = ed = 0
        for a in g.rooms:
            for b in g.rooms:
                ea += a_star(g, a, b).expanded
                ed += dijkstra(g, a, b).expanded
        self.assertLess(ea, ed)

    def test_crowd_cost_never_below_base(self):
        g = self.maps[0]
        fn = crowd_averse_cost({"Cafeteria": 3}, 1.5)
        r = a_star(g, "Reactor", "Navigation", cost_fn=fn)
        self.assertTrue(r.found)

    def test_map_properties(self):
        g = default_map()
        self.assertGreaterEqual(len(g.rooms), 8)
        self.assertTrue(g.dead_ends())
        self.assertTrue(g.trap_rooms())
        self.assertTrue(g.vent_rooms())
        for g in self.maps:
            for a in g.rooms:
                self.assertLess(g.dist(g.rooms[0], a), math.inf)          # connected


class TestExtraSearch(unittest.TestCase):
    def test_ida_optimal_and_greedy_not_better(self):
        from search import ida_star, greedy_best_first
        g = default_map()
        for av in (False, True):
            for a in g.rooms:
                for b in g.rooms:
                    opt = dijkstra(g, a, b, None, av).cost
                    self.assertAlmostEqual(ida_star(g, a, b, "euclidean", None, av).cost, opt, places=6)
                    self.assertGreaterEqual(greedy_best_first(g, a, b, "euclidean", None, av).cost, opt - 1e-9)


class TestSuspicion(unittest.TestCase):
    def test_belief_is_a_distribution(self):
        cfg = Config()
        belief = {x: 1 / 5 for x in "abcde"}
        ev = sus.Evidence(prox={x: i / 5 for i, x in enumerate("abcde")}, scene={}, group={"b": 3})
        ev.conflicts.append(sus.Conflict("c", "d", 3, "seen_elsewhere", 1.0, False, True, "x"))
        new, bd = sus.update_belief("z", belief, ev, cfg, 1.0)
        self.assertAlmostEqual(sum(new.values()), 1.0)
        self.assertGreater(new["c"], new["b"])        # refuted alibi raises suspicion
        self.assertGreater(new["e"], new["a"])        # more proximity -> more suspicion
        new2 = sus.apply_dissent(new, ["a"], cfg)
        self.assertGreater(new2["a"], new["a"] * 0.999)

    def test_tie_gives_no_elimination(self):
        cfg = Config()
        e, why, _ = sus.resolve_votes({"a": "x", "b": "x", "c": "y", "d": "y"}, {"x": .3, "y": .3}, cfg)
        self.assertIsNone(e)
        self.assertEqual(why, "tie")


class TestGame(unittest.TestCase):
    def test_deterministic(self):
        a = run_game(Config(seed=5))
        b = run_game(Config(seed=5, compare_baselines=False, record_frames=True))
        self.assertEqual(a.events, b.events)

    def test_every_game_terminates_with_valid_state(self):
        for pol in ("minimax", "greedy", "random"):
            for seed in range(8):
                r = run_game(Config(seed=seed, impostor_policy=pol, compare_baselines=False))
                self.assertIn(r.winner, ("crew", "impostor", None))
                for m in r.meetings:
                    for b in m["beliefs"].values():
                        self.assertAlmostEqual(sum(b.values()), 1.0, places=6)

    def test_scaled_configs_run(self):
        r = run_game(Config(n_rooms=20, n_crew=10, seed=3, compare_baselines=False))
        self.assertIsNotNone(r.winner)

    def test_logs_present(self):
        r = run_game(Config(seed=2))
        self.assertTrue(r.move_log)
        self.assertTrue(r.search_log.rows)
        row = r.search_log.rows[0]
        self.assertNotEqual(row["bfs_expanded"], "")


class TestMissing(unittest.TestCase):
    def test_last_with_weights(self):
        cfg = Config()
        g = default_map()
        tests = {n: sus.Testimony(n, {5: "Storage"}, [], []) for n in "abcd"}
        tests["a"].sightings.append((5, "Storage", "m"))
        ev = sus.build_evidence(tests, None, g, cfg, 0, 10, list("abcd"), ["m"])
        # a saw m in Storage at t5; companions = a,b,c,d (all claim Storage@5) -> 1/4 each
        self.assertAlmostEqual(ev.last_with["a"], 0.25)
        tests2 = {"a": sus.Testimony("a", {5: "Storage"}, [(5, "Storage", "m")], []),
                  "b": sus.Testimony("b", {5: "Admin"}, [], [])}
        ev2 = sus.build_evidence(tests2, None, g, cfg, 0, 10, ["a", "b"], ["m"])
        self.assertAlmostEqual(ev2.last_with["a"], 1.0)      # alone with the missing player
        self.assertNotIn("b", ev2.last_with)

    def test_missing_excludes_reported_victim_and_bodies_persist(self):
        seen_unreported = False
        for seed in range(60):
            r = run_game(Config(seed=seed, compare_baselines=False))
            for m in r.meetings:
                if m["body"]:
                    self.assertNotIn(m["body"]["victim"], m["missing"])
                if m["missing"]:
                    seen_unreported = True
                    self.assertTrue(set(m["missing"]) <= set(m["killed"]) | set(m["missing"]))
            if seen_unreported:
                break
        self.assertTrue(seen_unreported)


class TestImpostorRules(unittest.TestCase):
    def test_kill_only_when_alone_with_victim(self):
        import game_loop

        class Checked(game_loop.Game):
            kills = 0

            def do_kill(self, imp, victim):
                here = [c for c in self.alive_crew() if c.room == imp.room]
                assert here == [victim], "impostor killed with a witness present"
                Checked.kills += 1
                super().do_kill(imp, victim)
        for pol in ("minimax", "greedy", "random"):
            for seed in range(15):
                Checked(Config(seed=seed, impostor_policy=pol, compare_baselines=False)).run()
        self.assertGreater(Checked.kills, 0)

    def test_smart_impostor_never_vents_in_front_of_crew(self):
        import game_loop

        class Checked(game_loop.Game):
            hops = 0

            def depart(self, a, step):
                if step[2] and a.role == "impostor":
                    Checked.hops += 1
                    assert not any(c.room == a.room for c in self.alive_crew()), "vented with a witness"
                    assert not any(c.room == step[0] for c in self.alive_crew()), "vented into a watched room"
                super().depart(a, step)
        for seed in range(25):
            Checked(Config(seed=seed, compare_baselines=False)).run()
        self.assertGreater(Checked.hops, 0)


class TestEngineering(unittest.TestCase):
    def test_parallel_matches_serial(self):
        import experiments as E
        cfgs = [Config(seed=s, compare_baselines=False) for s in range(6)]
        a = E.run_batch(cfgs, 1)
        b = E.run_batch(cfgs, 2)
        for x, y in zip(a, b):
            for k in ("winner", "rounds", "ticks", "kills", "innocents"):
                self.assertEqual(x[k], y[k])

    def test_cautious_crew_beliefs_stay_distributions(self):
        for seed in range(10):
            r = run_game(Config(seed=seed, crew_policy="cautious", compare_baselines=False))
            for m in r.meetings:
                for b in m["beliefs"].values():
                    self.assertAlmostEqual(sum(b.values()), 1.0, places=6)

    def test_preset_loading(self):
        import json, os, tempfile
        d = tempfile.mkdtemp()
        good, bad = os.path.join(d, "g.json"), os.path.join(d, "b.json")
        with open(good, "w") as f:
            json.dump({"n_crew": 5, "crew_policy": "cautious"}, f)
        with open(bad, "w") as f:
            json.dump({"not_a_field": 1}, f)
        c = Config.from_json(good, seed=9)
        self.assertEqual((c.n_crew, c.crew_policy, c.seed), (5, "cautious", 9))
        with self.assertRaises(ValueError):
            Config.from_json(bad)

    def test_animation_renders_headless(self):
        import matplotlib
        matplotlib.use("Agg")
        from visualize import animate
        r = run_game(Config(seed=3, record_frames=True, compare_baselines=False))
        anim = animate(r, reveal=False, show=False)
        for i in (0, len(r.frames) // 2, len(r.frames) - 1):
            anim.draw_frame(i)


if __name__ == "__main__":
    unittest.main()
