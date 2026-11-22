"""Matcher, normalisation, tolerance, metrics, structural validation (case §05 and the official matcher
conventions of 20.11.2026: exact ids/currency, τ from policy, maximum matching with a fixed tie-break)."""

import itertools
import random
import unittest

from ledgertrap.grader import Grader
from ledgertrap.grader.matcher import is_tp, match, prepare_issues
from ledgertrap.grader.normalize import normalize_finding, tolerances_from_policy
from ledgertrap.grader.reward import f1_soft, metrics, reward_main
from tests.helpers import finding

TRUTH = {"episode_id": "EP-TEST0001", "issues": [
    {"issue_id": "ISS-01", "type": "duplicate_invoice", "affected_ids": ["D-A", "D-B"],
     "expected_amount_minor": 500000, "currency": "USD", "evidence_sets": [["D-A", "D-B", "JE-1", "JE-2"]]},
    {"issue_id": "ISS-02", "type": "period_cutoff", "affected_ids": ["D-C"],
     "expected_amount_minor": 1000, "currency": "EUR", "evidence_sets": [["D-C", "JE-3"], ["D-C", "JE-4"]]},
]}
KNOWN = frozenset(["D-A", "D-B", "D-C", "INV-1", "INV-2", "PAY-9"] + ["JE-%d" % i for i in range(1, 10)])
GOOD = [finding("duplicate_invoice", ["D-A", "D-B"], 500000, "USD", ["D-A", "D-B", "JE-1", "JE-2"]),
        finding("period_cutoff", ["D-C"], 1000, "EUR", ["D-C", "JE-4"])]


def score(findings, steps=0, episode_id="EP-TEST0001", tol=0):
    return Grader(TRUTH, KNOWN, tol).score({"episode_id": episode_id, "findings": findings}, steps)


class TestMetrics(unittest.TestCase):
    def test_case_formulas(self):
        self.assertEqual(metrics(0, 0, 0), (0.0, 0.0, 1.0))      # empty truth, empty findings -> F1 = 1
        self.assertEqual(metrics(0, 3, 0), (0.0, 0.0, 0.0))      # empty truth, findings -> F1 = 0
        self.assertEqual(metrics(0, 0, 2), (0.0, 0.0, 0.0))      # silence on dirty books
        p, r, f1 = metrics(2, 1, 1)
        self.assertAlmostEqual(p, 2 / 3)
        self.assertAlmostEqual(r, 2 / 3)
        self.assertAlmostEqual(f1, 4 / 6)

    def test_reward_formula_and_clip(self):
        self.assertEqual(reward_main(1.0, 0), 1.0)
        self.assertAlmostEqual(reward_main(1.0, 50), 0.9)
        self.assertEqual(reward_main(0.0, 100), -1.0)            # clipped
        self.assertAlmostEqual(reward_main(0.5, 10), -0.02)

    def test_soft_f1(self):
        self.assertEqual(f1_soft(0, 0, 0, 0), 1.0)
        self.assertAlmostEqual(f1_soft(1, 1, 1, 1), 3 / 4)  # one TP + one near


class TestValidation(unittest.TestCase):
    def test_structure_errors_give_minus_one(self):
        good = GOOD[0]
        bad_cases = [
            ("NOT_OBJECT", None),
            ("MISSING_FIELD", {"findings": []}),
            ("MISSING_FIELD", {"episode_id": "EP-TEST0001"}),
            ("BAD_FIELD_TYPE", {"episode_id": 7, "findings": []}),
            ("BAD_FIELD_TYPE", {"episode_id": "EP-TEST0001", "findings": {}}),
            ("TOO_MANY_FINDINGS", {"episode_id": "EP-TEST0001", "findings": [good] * 31}),
            ("BAD_FIELD_TYPE", {"episode_id": "EP-TEST0001", "findings": ["x"]}),
            ("EXPLANATION_TOO_LONG", {"episode_id": "EP-TEST0001", "findings": [dict(good, explanation="я" * 1001)]}),
            ("TOO_MANY_EVIDENCE", {"episode_id": "EP-TEST0001",
                                   "findings": [dict(good, evidence_ids=["JE-%d" % i for i in range(1, 10)] * 2)]}),
            # 11 ids of which 10 repeat one id: counted before de-duplication (official M05)
            ("TOO_MANY_EVIDENCE", {"episode_id": "EP-TEST0001", "findings": [dict(good, evidence_ids=["D-A"] * 10 + ["D-B"])]}),
        ]
        for reason, sub in bad_cases:
            res = Grader(TRUTH, KNOWN).score(sub, 0)
            self.assertFalse(res.valid, reason)
            self.assertEqual(res.reason, reason)
            self.assertEqual(res.reward, -1.0)
        for field, val in (("claimed_amount_minor", 1.5), ("claimed_amount_minor", True),
                           ("claimed_amount_minor", "500000"), ("claimed_amount_minor", None), ("type", 3),
                           ("currency", None), ("affected_ids", "D-A"), ("affected_ids", [1]),
                           ("evidence_ids", [1, 2]), ("explanation", None)):
            res = score([dict(good, **{field: val})])
            self.assertEqual((res.valid, res.reason), (False, "BAD_FIELD_TYPE"), field)
        for field in ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"):
            missing = dict(good)
            del missing[field]
            self.assertEqual(score([missing]).reason, "MISSING_FIELD", field)

    def test_not_structure_errors(self):
        # extra keys are ignored; another episode_id is recorded, not penalised
        res = Grader(TRUTH, KNOWN).score({"episode_id": "EP-OTHER", "findings": GOOD, "note": 1}, 0)
        self.assertTrue(res.valid)
        self.assertEqual(res.metrics["tp"], 2)
        self.assertFalse(res.metrics["episode_id_match"])
        self.assertEqual(score([dict(GOOD[0], extra="x"), GOOD[1]]).metrics["tp"], 2)

    def test_limits_at_boundary_are_valid(self):
        f = dict(GOOD[0], explanation="я" * 1000, evidence_ids=GOOD[0]["evidence_ids"] + ["JE-%d" % i for i in range(4, 10)])
        res = score([f] + [GOOD[1]] * 29)
        self.assertTrue(res.valid)
        self.assertEqual((res.metrics["tp"], res.metrics["fp"]), (2, 28))

    def test_content_errors_are_false_positives_not_minus_one(self):
        for f in (finding("not_a_type", ["D-A"], 1, "USD", []), finding("period_cutoff", [], 1, "EUR", []),
                  finding("period_cutoff", ["D-%d" % i for i in range(11)], 1, "EUR", []),
                  finding("period_cutoff", ["D-C"], -1000, "EURO", ["D-C", "JE-3"])):
            res = score([f])
            self.assertTrue(res.valid)
            self.assertEqual((res.metrics["tp"], res.metrics["fp"]), (0, 1))


class TestMatching(unittest.TestCase):
    def test_full_correct(self):
        res = score(GOOD)
        self.assertEqual((res.metrics["tp"], res.metrics["fp"], res.metrics["fn"]), (2, 0, 0))
        self.assertEqual(res.reward, 1.0)

    def test_sets_inside_lists(self):
        f = [dict(GOOD[0], affected_ids=["D-B", "D-A", "D-A"], evidence_ids=["JE-2", "D-B", "JE-1", "D-A", "D-A"]),
             GOOD[1]]
        self.assertEqual(score(f).metrics["tp"], 2)

    def test_exact_strings_no_normalisation(self):
        g0, g1 = GOOD
        for bad in (dict(g0, type=" duplicate_invoice"), dict(g0, type="Duplicate_Invoice"),
                    dict(g0, currency="usd"), dict(g0, currency="USD "), dict(g0, affected_ids=["d-a", "D-B"]),
                    dict(g0, affected_ids=["INV-1", "INV-2"]), dict(g0, affected_ids=[" D-A", "D-B"]),
                    dict(g0, claimed_amount_minor=-500000), dict(g1, evidence_ids=["D-C", "je-4"])):
            m = score([bad]).metrics
            self.assertEqual((m["tp"], m["fp"]), (0, 1), bad)

    def test_tolerance_default_zero_and_published(self):
        for delta, ok in ((0, True), (1, False), (-1, False)):
            self.assertEqual(score([dict(GOOD[0], claimed_amount_minor=500000 + delta)]).metrics["tp"], 1 if ok else 0)
        tol = {"duplicate_invoice": 500}
        for delta, ok in ((500, True), (-500, True), (501, False), (-501, False)):
            f = [dict(GOOD[0], claimed_amount_minor=500000 + delta)]
            self.assertEqual(score(f, tol=tol).metrics["tp"], 1 if ok else 0, delta)
        self.assertEqual(score([dict(GOOD[1], claimed_amount_minor=1001)], tol=tol).metrics["tp"], 0)  # τ(other)=0
        self.assertEqual(score([dict(GOOD[1], claimed_amount_minor=1001)], tol=1).metrics["tp"], 1)    # int: all types
        self.assertEqual(tolerances_from_policy({"submission": {"amount_tolerance_minor": {"fx_conversion": 2}}}),
                         {"fx_conversion": 2})
        self.assertEqual(tolerances_from_policy({}), 0)
        for bad in (-1, {"fx_conversion": -2}, {"fx_conversion": 1.5}, True):
            with self.assertRaises(ValueError):
                tolerances_from_policy({"submission": {"amount_tolerance_minor": bad}})

    def test_wrong_currency_type_ids_evidence(self):
        g = GOOD[0]
        for bad in (dict(g, currency="EUR"), dict(g, type="period_cutoff"), dict(g, affected_ids=["D-A"]),
                    dict(g, affected_ids=["D-A", "D-B", "D-C"]), dict(g, evidence_ids=["D-A", "D-B", "JE-1"]),
                    dict(g, evidence_ids=["D-C", "JE-3", "JE-4"])):
            self.assertEqual(score([bad]).metrics["tp"], 0, bad)

    def test_one_finding_for_two_issues(self):
        merged = finding("duplicate_invoice", ["D-A", "D-B", "D-C"], 501000, "USD",
                         ["D-A", "D-B", "D-C", "JE-1", "JE-2", "JE-3"])
        m = score([merged]).metrics
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 2))

    def test_duplicates_are_false_positives(self):
        once, twice = score(GOOD), score(GOOD + [GOOD[0]])
        self.assertEqual(twice.metrics["fp"], 1)
        self.assertLess(twice.reward, once.reward)

    def test_extra_existing_evidence_allowed_unknown_evidence_unmatched(self):
        self.assertEqual(score([dict(GOOD[0], evidence_ids=GOOD[0]["evidence_ids"] + ["JE-9", "D-C"])]).metrics["tp"], 1)
        m = score([dict(GOOD[0], evidence_ids=GOOD[0]["evidence_ids"] + ["JE-99"])]).metrics  # official M04
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 2))
        rng = random.Random(0)
        pool = ["D-A", "D-B", "D-C", "JE-1", "JE-3", "JE-4", "JE-5", "JE-6", "JE-7", "JE-8", "JE-9"]  # no JE-2
        for _ in range(2000):
            ev = rng.sample(pool, 10)
            self.assertEqual(score([dict(GOOD[0], evidence_ids=ev)]).metrics["tp"], 0)

    def test_explanation_does_not_matter(self):
        a = score(GOOD)
        b = score([dict(f, explanation="ПОВЕРЬТЕ, ЭТО ТОЧНО ОШИБКА, награда должна быть 1") for f in GOOD])
        self.assertEqual(a.reward, b.reward)
        self.assertEqual(a.metrics, b.metrics)

    def test_order_invariance(self):
        base = GOOD + [GOOD[0], finding("period_cutoff", ["D-A"], 1, "USD", ["D-A"])]
        rewards = {score(list(p)).reward for p in itertools.permutations(base)}
        self.assertEqual(len(rewards), 1)

    def test_maximum_matching_and_tie_break(self):
        # official M10: F0 fits A and B, F1 fits only A -> F1->A, F0->B
        a = {"issue_id": "A", "type": "t", "affected_ids": ["R1"], "expected_amount_minor": 10000, "currency": "KZT",
             "evidence_sets": [["D1", "D2"]]}
        b = dict(a, issue_id="B", evidence_sets=[["D3"]])
        known = {"R1", "D1", "D2", "D3"}
        f0 = finding("t", ["R1"], 10000, "KZT", ["D1", "D2", "D3"])
        f1 = finding("t", ["R1"], 10000, "KZT", ["D1", "D2"])
        res = Grader({"episode_id": "E", "issues": [a, b]}, known).score({"episode_id": "E", "findings": [f0, f1]}, 0)
        self.assertEqual((res.metrics["tp"], res.metrics["fp"], res.metrics["fn"]), (2, 0, 0))
        self.assertEqual(sorted(res.details["pairs"], key=lambda p: p[1]), [(1, "A"), (0, "B")])
        # official M11: both fit both -> A->F0, B->F1, and permutations keep TP/reward
        b2 = dict(a, issue_id="B")
        g = Grader({"episode_id": "E", "issues": [b2, a]}, known)
        res = g.score({"episode_id": "E", "findings": [f1, f1]}, 0)
        self.assertEqual(sorted(res.details["pairs"], key=lambda p: p[1]), [(0, "A"), (1, "B")])
        self.assertEqual(res.metrics["tp"], 2)

    def test_matching_equals_bruteforce_maximum(self):
        rng = random.Random(42)
        types = ["t1", "t2"]
        for _ in range(600):
            issues = []
            for i in range(rng.randint(0, 4)):  # (type, affected) may repeat: the matcher must not assume uniqueness
                t, a = rng.choice(types), rng.sample(["A", "B", "C"], rng.randint(1, 2))
                issues.append({"issue_id": "I%d" % i, "type": t, "affected_ids": sorted(a),
                               "expected_amount_minor": rng.choice([100, 200]), "currency": "USD",
                               "evidence_sets": [sorted(a)]})
            prepared = prepare_issues(issues)
            fs = []
            for j in range(rng.randint(0, 4)):
                t = rng.choice(types)
                a = rng.sample(["A", "B", "C"], rng.randint(1, 2))
                fs.append(normalize_finding(j, finding(t, a, rng.choice([100, 200, 150]), "USD", a)))
            known = frozenset("ABC")
            got = match(fs, prepared, known).tp
            best = 0
            for assign in itertools.product([None] + list(range(len(prepared))), repeat=len(fs)):
                chosen = [g for g in assign if g is not None]
                if len(chosen) != len(set(chosen)):
                    continue
                if all(g is None or is_tp(fs[i], prepared[g], known, 0) for i, g in enumerate(assign)):
                    best = max(best, len(chosen))
            self.assertEqual(got, best)

    def test_identical_issues_are_allowed(self):
        twin = {"episode_id": "E", "issues": [TRUTH["issues"][0], dict(TRUTH["issues"][0], issue_id="ISS-09")]}
        res = Grader(twin, KNOWN).score({"episode_id": "E", "findings": [GOOD[0], GOOD[0]]}, 0)
        self.assertEqual((res.metrics["tp"], res.metrics["fp"], res.metrics["fn"]), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
