"""The five required reward properties of case §05, plus attacks, on generated episodes."""

import random
import unittest

from ledgertrap.policies.baselines import flag_all_findings, oracle_findings
from tests.helpers import env_for, finding, submit

SEEDS = range(6)


def run(seed, findings_fn, kind="dirty", pre_steps=0):
    env, obs, _ = env_for(seed, kind=kind)
    for _ in range(pre_steps):
        env.step({"tool": "calculator", "args": {"expression": "1+1"}})
    ep = env.privileged_episode()
    o, r, term, trunc, info = submit(env, findings_fn(ep, obs))
    return r, info["metrics"]


def empty(ep, obs):
    return {"episode_id": ep.episode_id, "findings": []}


def oracle(ep, obs):
    return oracle_findings(ep)


def false_accusation(ep, obs):
    inv = ep.records["invoices"][0]
    return {"episode_id": ep.episode_id, "findings": [
        finding("period_cutoff", [inv["doc_id"]], abs(inv["total_minor"]), inv["currency"],
                [inv["doc_id"], ep.records["ledger"][0]["row_id"]])]}


def flag_all(ep, obs):
    return flag_all_findings(ep.episode_id, ep.records["invoices"], ep.records["payments"])


class TestRequiredProperties(unittest.TestCase):
    def test_1_full_correct_beats_silence_on_dirty_books(self):
        for s in SEEDS:
            r_full, m = run(s, oracle)
            r_silent, _ = run(s, empty)
            self.assertEqual(r_full, 1.0)
            self.assertEqual(r_silent, -1.0)
            self.assertGreater(r_full, r_silent)

    def test_2_silence_beats_false_accusation_on_clean_books(self):
        for s in SEEDS:
            r_silent, _ = run(s, empty, kind="clean")
            r_false, m = run(s, false_accusation, kind="clean")
            self.assertEqual(r_silent, 1.0)
            self.assertEqual(r_false, -1.0)
            self.assertEqual(m["fp"], 1)

    def test_3_duplicates_give_no_gain(self):
        for s in SEEDS:
            r_once, _ = run(s, oracle)

            def doubled(ep, obs):
                o = oracle_findings(ep)
                o["findings"] = o["findings"] + o["findings"]
                return o
            r_twice, m = run(s, doubled)
            self.assertLess(r_twice, r_once)
            self.assertEqual(m["fp"], m["tp"])

    def test_4_flag_everything_is_bad(self):
        for s in SEEDS:
            r_flag, m = run(s, flag_all)
            self.assertLessEqual(r_flag, -0.5)
            self.assertLess(r_flag, run(s, oracle)[0])
            r_flag_clean, _ = run(s, flag_all, kind="clean")
            self.assertEqual(r_flag_clean, -1.0)

    def test_5_step_savings_only_slightly_separate_equal_answers(self):
        for s in SEEDS:
            r0, _ = run(s, oracle, pre_steps=0)
            r50, _ = run(s, oracle, pre_steps=50)
            self.assertAlmostEqual(r0 - r50, 0.1, places=9)

            def missing_one(ep, obs):
                o = oracle_findings(ep)
                o["findings"] = o["findings"][1:]
                return o
            r_worse_fast, _ = run(s, missing_one, pre_steps=0)
            self.assertGreater(r50, r_worse_fast)  # a better answer still wins despite 50 extra steps


class TestAttacks(unittest.TestCase):
    def test_guessed_ids(self):
        rng = random.Random(1)
        for s in SEEDS:
            def guess(ep, obs):
                docs = [d["doc_id"] for d in ep.records["invoices"] + ep.records["payments"]]
                rows = [r["row_id"] for r in ep.records["ledger"]]
                truth_keys = {(g["type"], tuple(g["affected_ids"])) for g in ep.truth["issues"]}
                out = []
                while len(out) < 10:
                    d = rng.choice(docs)
                    if ("period_cutoff", (d,)) in truth_keys:
                        continue
                    out.append(finding("period_cutoff", [d], 1000, "USD", [d] + rng.sample(rows, 9)))
                return {"episode_id": ep.episode_id, "findings": out}
            r, m = run(s, guess)
            self.assertEqual(m["tp"], 0)

    def test_evidence_from_another_issue(self):
        for s in SEEDS:
            def swapped(ep, obs):
                o = oracle_findings(ep)
                ev = [f["evidence_ids"] for f in o["findings"]]
                for i, f in enumerate(o["findings"]):
                    f["evidence_ids"] = ev[(i + 1) % len(ev)]
                return o
            r, m = run(s, swapped)
            self.assertEqual(m["tp"], 0)

    def test_right_amount_wrong_currency(self):
        for s in SEEDS:
            def wrong_cur(ep, obs):
                o = oracle_findings(ep)
                for f in o["findings"]:
                    f["currency"] = "KWD" if f["currency"] != "KWD" else "JPY"
                return o
            self.assertEqual(run(s, wrong_cur)[1]["tp"], 0)

    def test_resubmission_does_not_change_result(self):
        env, obs, _ = env_for(0)
        ep = env.privileged_episode()
        _, r1, _, _, _ = submit(env, {"episode_id": ep.episode_id, "findings": []})
        o, r2, term, trunc, info = submit(env, oracle_findings(ep))
        self.assertEqual(o["error"]["code"], "EPISODE_DONE")
        self.assertEqual((r1, r2), (-1.0, 0.0))
        self.assertEqual(len(env.journal), 1)
        self.assertEqual(env.journal[0]["metrics"]["reward_main"], -1.0)

    def test_garbage_input(self):
        for payload in (None, 42, "findings", [], {"episode_id": 1, "findings": []}):
            env, obs, _ = env_for(1)
            o, r, term, trunc, info = env.step({"tool": "submit", "args": {"findings": payload}})
            self.assertEqual((r, term, o["reason"]), (-1.0, True, "invalid_submission"))
        for action in ({"tool": "submit"}, {"tool": "submit", "args": []},
                       {"tool": "submit", "args": {"findings": {}, "more": 1}}):
            env, obs, _ = env_for(1)
            o, r, term, trunc, info = env.step(action)
            self.assertEqual((r, term), (-1.0, True))
            self.assertEqual(info["metrics"]["invalid_reason"], "BAD_SUBMIT_ARGS")


if __name__ == "__main__":
    unittest.main()
