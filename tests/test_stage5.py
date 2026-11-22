"""Stage 5: interacting scenarios, process isolation, presentation transforms and pass^k, leak scanner,
versioned configs and difficulty presets, JSON schemas."""

import json
import os
import unittest

from ledgertrap import config as config_mod
from ledgertrap.env import LedgerTrapEnv
from ledgertrap.generator import generate
from ledgertrap.grader import Grader
from ledgertrap.idmap import id_map, pass_hat_k, presentation, remap_findings, remap_truth, story_fingerprint
from ledgertrap.leakscan import ks_2samp, scan_records, stat_report
from ledgertrap.policies.agents import make_policy
from ledgertrap.policies.baselines import oracle_findings
from ledgertrap.reaudit import reaudit
from ledgertrap.runner import builtin_agent, run_episode, run_many, summarize
from ledgertrap.schemas import SCHEMAS, validate, validate_episode
from ledgertrap.timeutil import parse, parse_offset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = {"errors": {"mode": "exact", "exact": {"C1": 1, "C2": 1}}, "company": {"n_foreign_currencies": [1, 2]}}


def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


class TestScenarios(unittest.TestCase):
    def test_c1_c2_truth_and_reaudit(self):
        for seed in range(8):
            ep = generate(seed, SCEN)
            by = {}
            for g in ep.truth["issues"]:
                by.setdefault(g["scenario"], []).append((g["type"], g["variant"]))
            self.assertEqual(sorted(by["C1"]), [("fx_conversion", "wrong_date"), ("period_cutoff", "early")])
            self.assertEqual(sorted(by["C2"]), [("duplicate_invoice", "document"),
                                                ("payment_allocation", "wrong_invoice")])
            rep = reaudit(ep)
            self.assertTrue(rep["ok"], rep["evidence_problems"] or rep["unexplained"])
            c1 = [g for g in ep.truth["issues"] if g["scenario"] == "C1"]
            self.assertEqual(c1[0]["affected_ids"], c1[1]["affected_ids"])
            p2 = next(p for p in ep.records["payments"] if p["doc_id"] == c1[0]["affected_ids"][0])
            tz = parse_offset(ep.manifest["timezone"])
            self.assertEqual(parse(p2["timestamp"]).astimezone(tz).date().isoformat(), ep.manifest["period_end"][:10])
            self.assertLess(p2["timestamp"][:10], ep.manifest["period_end"][:10])  # the string shows the last day

    def test_random_mode_places_scenarios(self):
        seen = set()
        for seed in range(30):
            ep = generate(seed, {"errors": {"interacting": {"C1": 1.0, "C2": 1.0}, "count": [6, 8],
                                            "clean_probability": 0.0}})
            seen |= {g["scenario"] for g in ep.truth["issues"]}
            self.assertTrue(reaudit(ep)["ok"])
        self.assertTrue({"C1", "C2"} <= seen)


class TestIsolation(unittest.TestCase):
    def test_process_grader_matches_inproc(self):
        actions = [{"tool": "list_records", "args": {"table": "ledger"}},
                   {"tool": "get_record", "args": {"table": "fx", "id": "nope"}}]
        outs = []
        for grader in ("inproc", "process"):
            env = LedgerTrapEnv(mode="train", grader=grader)
            o = [json.dumps(env.reset(3, SCEN), sort_keys=True)]
            o += [json.dumps(env.step(a), sort_keys=True) for a in actions]
            o.append(json.dumps(env.step({"tool": "submit", "args": {"findings": env.oracle_findings()}}),
                                sort_keys=True))
            outs.append(o)
            env.close()
        self.assertEqual(outs[0], outs[1])

    def test_truth_not_in_env_process(self):
        env = LedgerTrapEnv(mode="eval", grader="process")
        env.reset(4, SCEN)
        with self.assertRaises(RuntimeError):
            env.privileged_episode()
        self.assertIsNone(getattr(env._episode, "truth", None))
        env.close()

    def test_bad_seed_through_process(self):
        env = LedgerTrapEnv(grader="process")
        with self.assertRaises(ValueError):
            env.reset(-5)
        env.close()

    def test_agent_subprocess_sees_no_secret_and_empty_cwd(self):
        err = os.path.join(ROOT, "build", "test_probe.err")
        os.makedirs(os.path.dirname(err), exist_ok=True)
        env = LedgerTrapEnv(mode="eval", grader="process", secret="top-secret-value")
        ag = builtin_agent("probe", stderr_path=err)
        traj, j = run_episode(env, ag, 5, None, "t", "probe")
        ag.close()
        env.close()
        with open(err, encoding="utf-8") as f:
            seen = json.loads(f.readline())
        self.assertEqual(seen["cwd_files"], [])
        self.assertEqual(set(seen["env_keys"]), {"LANG", "PATH", "PYTHONHASHSEED", "PYTHONPATH"})
        self.assertIsNot(seen["parent_environ_has_secret"], True)
        self.assertNotIn("top-secret-value", json.dumps(traj))
        self.assertEqual(j["reason"], "submitted")

    def test_isolated_agents_score_like_inproc(self):
        eps = [("t", s, None) for s in range(4)]
        a = summarize(run_many(["auditor", "naive", "random"], eps))
        b = summarize(run_many(["auditor", "naive", "random"], eps, isolate=True, grader="process"))
        self.assertEqual([(x["policy"], x["reward_mean"], x["steps_mean"]) for x in a],
                         [(x["policy"], x["reward_mean"], x["steps_mean"]) for x in b])

    def test_hung_agent_is_cut_at_the_episode_time_limit_and_restarted(self):
        import sys
        import time as _time
        from ledgertrap.runner import run_external
        hang = [sys.executable, "-c", "import sys, time\nfor line in sys.stdin:\n    time.sleep(30)\n"]
        cfg = {"limits": {"time_limit_s": 1}}
        t0 = _time.time()
        journals = run_external(hang, "hang", [("t", 1, cfg), ("t", 2, cfg)], mode="train", grader="inproc")
        self.assertLess(_time.time() - t0, 10)
        self.assertEqual([j["reason"] for j in journals], ["time_limit", "time_limit"])
        self.assertTrue(all(j["metrics"]["tp"] == 0 for j in journals))

    def test_crashed_agent_is_truncated(self):
        import sys
        from ledgertrap.runner import SubprocessPolicy
        env = LedgerTrapEnv(mode="train")
        ag = SubprocessPolicy([sys.executable, "-c", "import sys; sys.exit(3)"], "crash")
        traj, j = run_episode(env, ag, 6, None, "t", "crash")
        ag.close()
        self.assertEqual(j["reason"], "policy_error")
        self.assertTrue(traj[-1]["truncated"])


class TestPresentationAndPassK(unittest.TestCase):
    def test_id_and_order_change_keep_oracle_and_reward(self):
        for seed in range(6):
            a = generate(seed, SCEN)
            b = generate(seed, dict(SCEN, presentation=presentation(1 + seed % 4)))
            m = id_map(a, b)
            self.assertTrue(all(k != v for k, v in m.items()))
            self.assertEqual(story_fingerprint(a), story_fingerprint(b))
            self.assertEqual(remap_truth(a.truth, m), b.truth["issues"])
            self.assertEqual(Grader.for_episode(b).score(remap_findings(oracle_findings(a), m), 0).reward, 1.0)
            self.assertNotEqual(a.files["ledger.csv"], b.files["ledger.csv"])

    def test_pass_hat_k(self):
        self.assertEqual(pass_hat_k([5, 5], 5, 3), 1.0)
        self.assertEqual(pass_hat_k([0, 5], 5, 1), 0.5)
        self.assertAlmostEqual(pass_hat_k([3], 5, 2), 3 / 10)
        self.assertEqual(pass_hat_k([4], 5, 5), 0.0)

    def test_auditor_is_stable_across_presentations(self):
        for seed in range(3):
            res = []
            for i in range(4):
                env = LedgerTrapEnv(mode="train")
                traj, j = run_episode(env, make_policy("auditor"), seed, dict(SCEN, presentation=presentation(i)))
                res.append((j["metrics"]["tp"], j["metrics"]["fp"], j["metrics"]["fn"]))
            self.assertEqual(len(set(res)), 1)


class TestLeakScanner(unittest.TestCase):
    def test_detects_planted_leaks(self):
        ep = generate(1, SCEN)
        truth = {ep.episode_id: ep.truth}
        clean = {"episode_id": ep.episode_id, "step": 1, "observation": {"type": "tool_result"}, "reward": 0.0,
                 "mode": "eval"}
        self.assertEqual(scan_records([clean], truth), [])
        leaks = [dict(clean, observation={"issue_id": "x"}),
                 dict(clean, observation={"note": ep.truth["issues"][0]["issue_id"]}),
                 dict(clean, observation={"note": "see /home/user/grader/truth.json"}),
                 dict(clean, observation={"note": "s3cr3t"}), dict(clean, reward=1.0)]
        for rec in leaks:
            self.assertTrue(scan_records([rec], truth, secrets=["s3cr3t"]), rec)

    def test_real_trajectories_are_clean(self):
        for mode in ("train", "eval"):
            env = LedgerTrapEnv(mode=mode)
            for name in ("auditor", "naive", "random", "flag_all"):
                traj, j = run_episode(env, make_policy(name), 7, SCEN, "t", name)
                ep = env.privileged_episode()
                self.assertEqual(scan_records(traj, {ep.episode_id: ep.truth}, secrets=["ledgertrap-open-v1"]), [])

    def test_ks(self):
        d, p = ks_2samp(list(range(100)), list(range(100)))
        self.assertEqual(d, 0.0)
        d, p = ks_2samp(list(range(100)), list(range(50, 150)))
        self.assertLess(p, 0.01)

    def test_injected_records_look_like_clean_ones(self):
        rep = stat_report([generate(s) for s in range(80)])
        for name, r in rep.items():
            if not name.startswith("_"):
                self.assertTrue(r["ok"], (name, r))


class TestConfigs(unittest.TestCase):
    def test_splits_are_versioned_and_disjoint(self):
        sp = load("configs/splits.json")
        self.assertIn("splits_version", sp)
        train, dev = set(sp["train"]["seeds"]), set(sp["dev"]["seeds"])
        self.assertEqual((len(train), len(dev)), (10, 10))
        self.assertFalse(train & dev)
        ct, cd = load(sp["train"]["config"]), load(sp["dev"]["config"])
        self.assertNotEqual(config_mod.config_hash(config_mod.load(ct)), config_mod.config_hash(config_mod.load(cd)))
        for c in (ct, cd):
            self.assertEqual(c["config_version"], config_mod.CONFIG_VERSION)

    def test_presets_l1_to_l5(self):
        sizes = []
        for i in range(1, 6):
            cfg = load("configs/presets/L%d.json" % i)
            config_mod.load(cfg)
            ep = generate(i, cfg)
            self.assertTrue(reaudit(ep)["ok"])
            sizes.append(ep.manifest["counts"]["ledger"])
        self.assertLess(sizes[0], sizes[-1])
        l1 = generate(3, load("configs/presets/L1.json"))
        self.assertEqual(list(l1.manifest["currencies"]), [l1.manifest["base_currency"]])


class TestSchemas(unittest.TestCase):
    def test_schema_files_in_sync(self):
        for name, schema in SCHEMAS.items():
            path = os.path.join(ROOT, "schemas", "%s.schema.json" % name)
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            doc.pop("$schema")
            doc.pop("$id")
            self.assertEqual(doc, json.loads(json.dumps(schema)), name)

    def test_episodes_validate(self):
        for seed in range(6):
            for cfg in (None, SCEN, load("configs/presets/L5.json")):
                self.assertEqual(validate_episode(generate(seed, cfg)), [])

    def test_validator_rejects(self):
        self.assertTrue(validate({"tool": "rm"}, SCHEMAS["action"]))
        self.assertTrue(validate({"episode_id": "E", "findings": [{}]}, SCHEMAS["findings"]))


if __name__ == "__main__":
    unittest.main()
