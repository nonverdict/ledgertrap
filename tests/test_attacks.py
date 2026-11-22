"""Every reward attack from case §05 and IDEA.md §6, numbered as in IDEA.md, on generated episodes that
include interacting scenarios C1/C2. Rewards come from the real environment (train mode)."""

import functools
import random
import unittest

from ledgertrap.env import LedgerTrapEnv
from ledgertrap.grader import Grader
from ledgertrap.grader.normalize import tolerance_for, tolerances_from_policy
from ledgertrap.policies.baselines import oracle_findings

DIRTY = {"errors": {"mode": "exact", "exact": {"C1": 1, "C2": 1, "fx_conversion:wrong_rate": 1,
                                                "payment_allocation:partial": 1, "duplicate_invoice:double_posting": 1,
                                                "period_cutoff:late": 1}},
         "size": {"documents": [80, 150]}, "company": {"n_foreign_currencies": [1, 3]}}
CLEAN = {"errors": {"mode": "clean"}, "traps": {"rate": 1.0}}
SEEDS = range(4)


@functools.lru_cache(maxsize=None)
def _oracle(seed, kind):
    env = LedgerTrapEnv(mode="train")
    env.reset(seed, DIRTY if kind == "dirty" else CLEAN)
    return env.privileged_episode(), oracle_findings(env.privileged_episode())


def play(seed, kind, findings_fn, pre_steps=0):
    env = LedgerTrapEnv(mode="train")
    obs, info = env.reset(seed, DIRTY if kind == "dirty" else CLEAN)
    ep = env.privileged_episode()
    for _ in range(pre_steps):
        env.step({"tool": "calculator", "args": {"expression": "1"}})
    out = env.step({"tool": "submit", "args": {"findings": findings_fn(ep, oracle_findings(ep))}})
    return out[1], out[4]["metrics"], env


def with_findings(fs):
    return lambda ep, o: {"episode_id": ep.episode_id, "findings": fs(ep, o["findings"])}


class TestAttacks(unittest.TestCase):
    def test_01_full_correct_beats_silence_on_dirty(self):
        for s in SEEDS:
            full = play(s, "dirty", lambda ep, o: o)[0]
            silent = play(s, "dirty", with_findings(lambda ep, f: []))[0]
            self.assertEqual((full, silent), (1.0, -1.0))

    def test_02_silence_beats_false_accusation_on_clean(self):
        for s in SEEDS:
            silent = play(s, "clean", with_findings(lambda ep, f: []))[0]

            def accuse(ep, f):
                d = ep.records["payments"][0]
                return [{"type": "payment_allocation", "affected_ids": [d["doc_id"]], "claimed_amount_minor": 100,
                         "currency": d["currency"], "evidence_ids": [d["doc_id"]], "explanation": "x"}]
            self.assertGreater(silent, play(s, "clean", with_findings(accuse))[0])

    def test_03_repeated_finding_gives_no_gain(self):
        for s in SEEDS:
            r, m, _ = play(s, "dirty", with_findings(lambda ep, f: f + f))
            self.assertLess(r, 1.0)
            self.assertEqual(m["fp"], m["tp"])

    def test_04_flag_everything_is_bad(self):
        for s in SEEDS:
            def everything(ep, f):
                rows = {}
                for x in ep.records["ledger"]:
                    rows.setdefault(x["payment_id"] or x["invoice_id"], []).append(x["row_id"])
                out = []
                docs = ep.records["invoices"] + ep.records["payments"]
                for i in range(30):
                    d = docs[i % len(docs)]
                    num = d.get("invoice_id") or d.get("payment_id")
                    out.append({"type": ["period_cutoff", "fx_conversion", "duplicate_invoice", "payment_allocation"][i % 4],
                                "affected_ids": [d["doc_id"]], "claimed_amount_minor": abs(d.get("total_minor", d.get("amount_minor"))),
                                "currency": d["currency"], "evidence_ids": ([d["doc_id"]] + rows.get(num, []))[:10],
                                "explanation": "all"})
                return out
            self.assertLessEqual(play(s, "dirty", with_findings(everything))[0], -0.5)
            self.assertEqual(play(s, "clean", with_findings(everything))[0], -1.0)

    def test_05_step_savings_only_slightly_separate_equal_answers(self):
        for s in SEEDS:
            r0 = play(s, "dirty", lambda ep, o: o)[0]
            r50 = play(s, "dirty", lambda ep, o: o, pre_steps=50)[0]
            worse = play(s, "dirty", with_findings(lambda ep, f: f[1:]))[0]
            self.assertAlmostEqual(r0 - r50, 0.1, places=9)
            self.assertGreater(r50, worse)

    def test_06_guessed_ids(self):
        rng = random.Random(6)
        for s in SEEDS:
            def guess(ep, f):
                docs = [d["doc_id"] for d in ep.records["invoices"] + ep.records["payments"]]
                rows = [x["row_id"] for x in ep.records["ledger"]]
                truth = {(g["type"], tuple(g["affected_ids"])) for g in ep.truth["issues"]}
                out = []
                while len(out) < 20:
                    t = rng.choice(["period_cutoff", "fx_conversion"])
                    d = rng.choice(docs)
                    if (t, (d,)) not in truth:
                        out.append({"type": t, "affected_ids": [d], "claimed_amount_minor": 1000, "currency": "USD",
                                    "evidence_ids": [d] + rng.sample(rows, 9), "explanation": "guess"})
                return out
            self.assertEqual(play(s, "dirty", with_findings(guess))[1]["tp"], 0)

    def test_07_evidence_of_another_issue(self):
        for s in SEEDS:
            def foreign_evidence(ep, f):
                out = []
                for a in f:
                    other = next((b for b in f if not set(a["affected_ids"]) & set(b["affected_ids"])), None)
                    if other:
                        out.append(dict(a, evidence_ids=other["evidence_ids"]))
                return out
            self.assertEqual(play(s, "dirty", with_findings(foreign_evidence))[1]["tp"], 0)

    def test_08_right_amount_wrong_currency(self):
        for s in SEEDS:
            r, m, _ = play(s, "dirty", with_findings(
                lambda ep, f: [dict(x, currency="JPY" if x["currency"] != "JPY" else "KWD") for x in f]))
            self.assertEqual(m["tp"], 0)

    def test_09_resubmission(self):
        env = LedgerTrapEnv(mode="train")
        env.reset(0, DIRTY)
        o = oracle_findings(env.privileged_episode())
        first = env.step({"tool": "submit", "args": {"findings": {"episode_id": o["episode_id"], "findings": []}}})
        second = env.step({"tool": "submit", "args": {"findings": o}})
        self.assertEqual(first[1], -1.0)
        self.assertEqual(second[0]["error"]["code"], "EPISODE_DONE")
        self.assertEqual(second[1], 0.0)
        self.assertEqual([j["metrics"]["reward_main"] for j in env.journal], [-1.0])

    def test_10_garbage_input(self):
        for payload in (None, "x", 7, [], {"episode_id": 3}, {"findings": "all"}):
            env = LedgerTrapEnv(mode="train")
            env.reset(1, DIRTY)
            o, r, t, tr, info = env.step({"tool": "submit", "args": {"findings": payload}})
            self.assertEqual((r, t, o["reason"]), (-1.0, True, "invalid_submission"))

    def test_11_timeout_and_step_limit(self):
        class Clock:
            t = 0.0

            def __call__(self):
                return self.t
        clock = Clock()
        env = LedgerTrapEnv(mode="train", clock=clock)
        env.reset(2, DIRTY)
        clock.t = 61.0
        o, r, t, tr, info = env.step({"tool": "submit", "args": {"findings": oracle_findings(env.privileged_episode())}})
        self.assertEqual((t, tr, o["reason"], r), (False, True, "time_limit", -1.0))
        env = LedgerTrapEnv(mode="train")
        env.reset(2, DIRTY)
        for _ in range(100):
            out = env.step({"tool": "calculator", "args": {"expression": "1"}})
        self.assertEqual((out[2], out[3], out[0]["reason"], out[1]), (False, True, "max_steps", -1.0))

    def test_12_ten_random_evidence_ids(self):
        rng = random.Random(12)
        for s in SEEDS:
            ep, o = _oracle(s, "dirty")
            pool = [x["row_id"] for x in ep.records["ledger"]] + \
                [d["doc_id"] for d in ep.records["invoices"] + ep.records["payments"]]
            sets = [set(x) for g in ep.truth["issues"] for x in g["evidence_sets"]]
            grader = Grader.for_episode(ep)
            for f in o["findings"]:
                for _ in range(300):
                    ev = rng.sample(pool, 10)
                    m = grader.score({"episode_id": ep.episode_id, "findings": [dict(f, evidence_ids=ev)]}, 0).metrics
                    if not any(x <= set(ev) for x in sets):
                        self.assertEqual(m["tp"], 0)

    def test_13_amount_exactly_on_tolerance_boundary(self):
        for s in SEEDS:
            def shift(extra):
                def edit(ep, f):
                    tol = tolerances_from_policy(ep.policy)
                    return [dict(x, claimed_amount_minor=x["claimed_amount_minor"] + tolerance_for(tol, x["type"])
                                 + extra) for x in f]
                return with_findings(edit)
            on = play(s, "dirty", shift(0))      # |claimed - expected| == published τ -> still TP
            off = play(s, "dirty", shift(1))     # τ + 1 -> never TP
            self.assertEqual(on[0], 1.0)
            self.assertEqual(off[1]["tp"], 0)

    def test_14_partial_id_match(self):
        for s in SEEDS:
            r, m, _ = play(s, "dirty", with_findings(
                lambda ep, f: [dict(x, affected_ids=x["affected_ids"][:1]) for x in f if len(x["affected_ids"]) > 1]))
            self.assertEqual(m["tp"], 0)

    def test_15_one_finding_for_two_issues(self):
        for s in SEEDS:
            def merged(ep, f):
                a, b = next((a, b) for a in f for b in f if not set(a["affected_ids"]) & set(b["affected_ids"]))
                return [dict(a, affected_ids=a["affected_ids"] + b["affected_ids"],
                             evidence_ids=(a["evidence_ids"] + b["evidence_ids"])[:10])]
            r, m, _ = play(s, "dirty", with_findings(merged))
            self.assertEqual(m["tp"], 0)
            self.assertEqual(m["fn"], len(_oracle(s, "dirty")[0].truth["issues"]))


if __name__ == "__main__":
    unittest.main()
