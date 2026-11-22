"""Official FinTech checks (20.11.2026) through judge_adapter.py, plus negative controls: each block must FAIL
when the behaviour it guards is broken on purpose (otherwise a green run would prove nothing)."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import judge_adapter
from ledgertrap import official_checks as oc
from ledgertrap import store as store_mod
from ledgertrap.grader import matcher as matcher_mod
from ledgertrap.grader import normalize as normalize_mod


def _greedy_match(findings, issues, known_ids, tolerances=0):
    """The pre-20.11 greedy rule: first fitting finding takes the issue (not a maximum matching)."""
    taken, pairs = set(), []
    for f in findings:
        for g in sorted(issues, key=lambda g: g["issue_id"]):
            if g["issue_id"] not in taken and matcher_mod.is_tp(f, g, known_ids, tolerances):
                taken.add(g["issue_id"])
                pairs.append((f.index, g["issue_id"]))
                break
    tp = len(pairs)
    return matcher_mod.MatchResult(tp, len(findings) - tp, len(issues) - tp, pairs, [])


class TestOfficial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="ledgertrap-official-test-")
        cls.fx = oc.Fixtures(cls.work)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def _sub(self, name):
        d = os.path.join(self.work, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        return d

    def test_all_blocks_green(self):
        out = oc.run_official(verbose=False)
        self.assertEqual((out["block1"], out["block2"], out["block3"], out["examples"], out["block4"]),
                         (20, 20, 20, 12, 20), out["detail"])

    def test_adapter_outputs_follow_published_schemas(self):
        from ledgertrap.schemas import SCHEMAS, validate
        env = judge_adapter.make_env()
        obs, info = env.reset(0, {"mode": "closed", "episode_dir": self.fx.E, "team_config": {}})
        seen = [obs]
        f = oc.read_folder(self.fx.E)
        for act in ({"name": "list_records", "args": {"table": "fx", "cursor": None, "limit": 2}},
                    {"name": "get_record", "args": {"table": "ledger", "id": f["ledger"][0]["row_id"]}},
                    {"name": "get_document", "args": {"doc_id": f["payments"][0]["doc_id"]}},
                    {"name": "calculator", "args": {"expression": "1/3"}},
                    {"name": "get_record", "args": {"table": "ledger", "id": self.fx.U}},
                    {"name": "submit", "args": {"findings": oc._oracle(f["truth"], f["manifest"]["episode_id"])}},
                    {"name": "calculator", "args": {"expression": "1"}}):
            self.assertEqual(validate(act, SCHEMAS["judge_action"]), [])
            seen.append(env.step(act)[0])
        for o in seen:
            self.assertEqual(validate(o, SCHEMAS["judge_observation"]), [], o)
        good = judge_adapter.score_episode(self.fx.E, oc._oracle(f["truth"], f["manifest"]["episode_id"]), 4)
        bad = judge_adapter.score_episode(self.fx.E, {"episode_id": "x", "findings": "nope"}, 4)
        for r in (good, bad):
            self.assertEqual(validate(r, SCHEMAS["score_result"]), [], r)
        self.assertEqual(bad["return"], -1)
        self.assertEqual(judge_adapter.JOURNAL[-1]["invalid_reason"], "BAD_FIELD_TYPE")

    def test_flat_folder_and_train_mode(self):
        flat = self._sub("flat")
        for sub in ("public", "grader"):
            for name in os.listdir(os.path.join(self.fx.E, sub)):
                shutil.copy(os.path.join(self.fx.E, sub, name), flat)
        a = oc._run(self.fx.E, oc._script(self.fx.E, oc.read_folder(self.fx.E)))[0]
        b = oc._run(flat, oc._script(flat, oc.read_folder(flat)))[0]
        self.assertEqual(a, b)
        env = judge_adapter.make_env()
        obs, info = env.reset(7, {"mode": "train", "team_config": {"errors": {"mode": "clean"}}})
        self.assertEqual(obs["remaining_steps"], 100)
        self.assertIn("episode_id", info)
        o = env.step({"name": "submit", "args": {"findings": {"episode_id": obs["manifest"]["episode_id"],
                                                              "findings": []}}})
        self.assertEqual(o[1], 1.0)  # train mode: the terminal reward goes to the policy (clean + silence)
        self.assertIn("metrics", o[4])
        bare = judge_adapter.make_env().reset(7, {"errors": {"mode": "clean"}})[0]  # bare team config = train
        self.assertEqual(bare["manifest"], obs["manifest"])

    def test_block1_catches_contract_breaks(self):
        orig = judge_adapter.JudgeEnv.reset

        def leaky_reset(self, seed, config=None):
            obs, info = orig(self, seed, config)
            obs = dict(obs, seed=seed)
            return obs, info
        with mock.patch.object(judge_adapter.JudgeEnv, "reset", leaky_reset):
            res = oc.block1(self.fx)
        self.assertFalse(res["C02"])

    def test_block3_catches_relative_tolerance(self):
        with mock.patch.object(matcher_mod, "tolerance_for", lambda tol, t: 10):
            res, m = oc.block3(self.fx, self._sub("b3tol"))
        self.assertFalse(res["N08"])
        self.assertFalse(m["M07"])

    def test_block3_catches_greedy_matching(self):
        with mock.patch.object(matcher_mod, "match", _greedy_match), \
                mock.patch("ledgertrap.grader.grader.match", _greedy_match):
            res, m = oc.block3(self.fx, self._sub("b3greedy"))
        self.assertFalse(m["M10"])

    def test_block3_catches_case_folding(self):
        def folding(index, f):
            n = normalize_mod.NormFinding(index, f["type"].strip().lower(), frozenset(f["affected_ids"]),
                                          abs(f["claimed_amount_minor"]), f["currency"].strip().upper(),
                                          frozenset(f["evidence_ids"]))
            return n
        with mock.patch("ledgertrap.grader.grader.normalize_finding", folding):
            res, m = oc.block3(self.fx, self._sub("b3fold"))
        self.assertFalse(m["M06"])

    def test_block4_catches_case_insensitive_store(self):
        orig = store_mod.RecordStore.get_record

        def upper_lookup(self, table, rid):
            return orig(self, table, rid.strip().upper())
        _, _, dirs = oc.block2(self._sub("b4src"))
        with mock.patch.object(store_mod.RecordStore, "get_record", upper_lookup):
            ok, why = oc.pair_check(7, dirs["D08"], self._sub("b4"))   # ids replaced by lower-case ids
        self.assertFalse(ok)
        self.assertIn("evidence_reachable", why)

    def test_block2_catches_wrong_truth(self):
        _, _, dirs = oc.block2(self._sub("b2src"))
        import json
        d = dirs["D06"]
        p = os.path.join(d, "grader", "truth.json")
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        t["issues"][0]["expected_amount_minor"] += 1
        with open(p, "w", encoding="utf-8") as f:
            json.dump(t, f)
        from ledgertrap.episode_io import load_episode_dir
        from ledgertrap.jury_checks import b_checks
        c = b_checks(load_episode_dir(d))
        self.assertFalse(c["B10"])


if __name__ == "__main__":
    unittest.main()
