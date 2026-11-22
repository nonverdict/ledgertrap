"""Stage 4: payment_allocation and fx_conversion, multi-currency, clean companies and traps,
independent re-auditor (incl. sabotage), baseline policies and trajectories."""

import copy
import json
import unittest
from datetime import date
from decimal import Decimal

from ledgertrap.config import TRAPS
from ledgertrap.env import LedgerTrapEnv
from ledgertrap.generator import generate
from ledgertrap.money import conv
from ledgertrap.policies.agents import make_policy
from ledgertrap.reaudit import reaudit, verify_evidence_sets
from ledgertrap.runner import run_episode, run_many, summarize
from ledgertrap.timeutil import last_business_on_or_before, parse, parse_offset
from tests.helpers import ALL_VARIANTS, episode

TRAJ_FIELDS = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}


def _index(ep):
    docs = {d["doc_id"]: d for d in ep.records["invoices"] + ep.records["payments"]}
    rows = {r["row_id"]: r for r in ep.records["ledger"]}
    fx = {(f["date"], f["source_currency"]): Decimal(f["rate"]) for f in ep.records["fx"]}
    return docs, rows, fx


class TestNewErrorTypes(unittest.TestCase):
    def test_all_nine_variants_placed(self):
        for seed in range(6):
            ep = episode(seed, "all")
            got = sorted("%s:%s" % (g["type"], g["variant"]) for g in ep.truth["issues"])
            self.assertEqual(got, sorted(ALL_VARIANTS["errors"]["exact"]))

    def test_allocation_semantics(self):
        for seed in range(6):
            ep = episode(seed, "all")
            docs, rows, fx = _index(ep)
            for g in ep.truth["issues"]:
                if g["type"] != "payment_allocation":
                    continue
                pay = next(docs[x] for x in g["affected_ids"] if "payment_id" in docs[x])
                inv = next(docs[x] for x in g["affected_ids"] if "invoice_id" in docs[x])
                self.assertEqual(len(pay["invoice_refs"]), 1)
                self.assertEqual(pay["invoice_refs"][0]["invoice_id"], inv["invoice_id"])
                self.assertEqual(g["currency"], pay["currency"])
                prow = [r for r in ep.records["ledger"] if r["payment_id"] == pay["payment_id"]]
                credited = [r for r in prow if r["invoice_id"] == inv["invoice_id"]]
                if g["variant"] == "missing":
                    self.assertEqual(credited, [])
                    self.assertEqual(g["expected_amount_minor"], pay["invoice_refs"][0]["amount_minor"])
                elif g["variant"] == "partial":
                    self.assertEqual(len(credited), 1)
                    self.assertLess(g["expected_amount_minor"], pay["invoice_refs"][0]["amount_minor"])
                else:
                    stray = [r for r in prow if r["invoice_id"] not in (None, inv["invoice_id"])]
                    self.assertEqual(len(stray), 1)
                    self.assertEqual(credited, [])
                for s in g["evidence_sets"]:
                    has_fx = any(x.count(":") == 2 for x in s)
                    if g["variant"] != "wrong_invoice":
                        self.assertEqual(has_fx, pay["currency"] != ep.manifest["base_currency"])

    def test_fx_semantics(self):
        for seed in range(6):
            ep = episode(seed, "all")
            docs, rows, fx = _index(ep)
            tz = parse_offset(ep.manifest["timezone"])
            base = ep.manifest["base_currency"]
            for g in ep.truth["issues"]:
                if g["type"] != "fx_conversion":
                    continue
                d = docs[g["affected_ids"][0]]
                self.assertNotEqual(d["currency"], base)
                self.assertEqual(g["currency"], base)
                when = parse(d.get("issued_at") or d["timestamp"])
                rd = last_business_on_or_before(when.astimezone(tz).date()).isoformat()
                rate = fx[(rd, d["currency"])]
                amount = d.get("total_minor", d.get("amount_minor"))
                right = conv(amount, d["currency"], base, rate, ep.manifest["rounding"])
                ev = g["evidence_sets"][0]
                self.assertIn("%s:%s:%s" % (rd, d["currency"], base), ev)
                row = rows[[x for x in ev if x in rows][0]]
                posted = row["debit_minor"] + row["credit_minor"]
                self.assertEqual(abs(posted - right), g["expected_amount_minor"])
                self.assertGreaterEqual(g["expected_amount_minor"], 100)


class TestMultiCurrency(unittest.TestCase):
    CFG = {"company": {"base_currency": "KZT", "foreign_currencies": ["JPY", "KWD", "USD"],
                       "n_foreign_currencies": [3, 3]}, "errors": {"count": [3, 8]}}

    def test_exponents_and_business_day_rates(self):
        for seed in range(5):
            ep = generate(seed, self.CFG)
            self.assertEqual(ep.manifest["currencies"], {"JPY": 0, "KWD": 3, "KZT": 2, "USD": 2})
            for f in ep.records["fx"]:
                self.assertLess(date.fromisoformat(f["date"]).weekday(), 5)
                self.assertEqual(len(f["rate"].replace(".", "").lstrip("0")), 6)
            self.assertTrue(reaudit(ep)["ok"])

    def test_every_foreign_document_has_its_rate(self):
        for seed in range(5):
            ep = generate(seed, self.CFG)
            docs, rows, fx = _index(ep)
            tz = parse_offset(ep.manifest["timezone"])
            for d in docs.values():
                if d["currency"] == "KZT":
                    continue
                when = parse(d.get("issued_at") or d["timestamp"])
                rd = last_business_on_or_before(when.astimezone(tz).date()).isoformat()
                self.assertIn((rd, d["currency"]), fx)

    def test_rounding_mode_from_policy(self):
        for mode in ("HALF_UP", "HALF_EVEN"):
            ep = generate(3, {"company": {"rounding": mode}})
            self.assertEqual(ep.policy["rounding"], mode)
            self.assertTrue(reaudit(ep)["ok"])


class TestCleanAndTraps(unittest.TestCase):
    def test_clean_companies_have_all_traps_and_no_findings(self):
        planted = set()
        for seed in range(12):
            ep = generate(seed, {"errors": {"mode": "clean"}, "traps": {"rate": 1.0}})
            self.assertEqual(ep.truth["issues"], [])
            planted |= set(ep.provenance["traps_planted"])
            rep = reaudit(ep)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["fp"], 0)
        self.assertEqual(planted, set(TRAPS))


class TestReauditor(unittest.TestCase):
    def test_exact_on_random_and_all_variant_episodes(self):
        for seed in range(30):
            rep = reaudit(generate(seed))
            self.assertTrue(rep["ok"], (seed, rep["tp"], rep["fp"], rep["fn"], rep["evidence_problems"]))
        for seed in range(6):
            rep = reaudit(episode(seed, "all"))
            self.assertTrue(rep["ok"], (seed, rep["evidence_problems"]))
            self.assertEqual(rep["tp"], 9)
            self.assertEqual(len(rep["certificates"]), 9)

    def _tamper(self, ep, pick, edit):
        for r in ep.records["ledger"]:
            if pick(r):
                edit(r)
                return True
        return False

    def test_sabotage_undeclared_discrepancy_is_detected(self):
        base_ep = generate(5, {"errors": {"mode": "clean"}, "company": {"n_foreign_currencies": [2, 2]}})
        self.assertTrue(reaudit(base_ep)["ok"])
        base = base_ep.manifest["base_currency"]
        nums = {d["invoice_id"]: d for d in base_ep.records["invoices"] if d["kind"] == "invoice"}
        for want_foreign in (False, True):
            ep = copy.deepcopy(base_ep)
            target = next(n for n, d in nums.items() if (d["currency"] != base) == want_foreign)

            def pick(r, target=target):
                return r["invoice_id"] == target and r["payment_id"] is None and r["debit_minor"] > 0

            def edit(r):
                r["debit_minor"] += 12345   # an undeclared error in the books
            self.assertTrue(self._tamper(ep, pick, edit))
            rep = reaudit(ep)
            self.assertFalse(rep["ok"], want_foreign)
            self.assertTrue(rep["fp"] > 0 or rep["unexplained"], want_foreign)
        # an expense row nobody would flag as one of the four types
        ep = copy.deepcopy(base_ep)
        self.assertTrue(self._tamper(ep, lambda r: r["invoice_id"] is None and r["payment_id"] is None,
                                     lambda r: r.update(debit_minor=r["debit_minor"] + 1)
                                     if r["debit_minor"] else r.update(credit_minor=r["credit_minor"] + 1)))
        self.assertFalse(reaudit(ep)["ok"])
        # a timestamp moved across the period end without being declared
        ep = copy.deepcopy(base_ep)
        pe = ep.manifest["period_end"]
        self.assertTrue(self._tamper(ep, lambda r: r["payment_id"] is not None and r["timestamp"] < pe,
                                     lambda r: r.update(timestamp=pe)))
        self.assertFalse(reaudit(ep)["ok"])

    def test_sabotage_missing_truth_issue_is_detected(self):
        ep = copy.deepcopy(episode(1, "all"))
        ep.truth["issues"] = ep.truth["issues"][1:]
        rep = reaudit(ep)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["fp"], 1)

    def test_sabotage_insufficient_or_bloated_evidence_is_detected(self):
        ep = copy.deepcopy(episode(2, "all"))
        ep.truth["issues"][0]["evidence_sets"] = [s[:-1] for s in ep.truth["issues"][0]["evidence_sets"]]
        self.assertTrue(any(p[1] == "insufficient" for p in verify_evidence_sets(ep)))
        ep = copy.deepcopy(episode(2, "all"))
        extra = ep.records["ledger"][0]["row_id"]
        ep.truth["issues"][0]["evidence_sets"] = [s + [extra] for s in ep.truth["issues"][0]["evidence_sets"]]
        self.assertTrue(any(p[1] == "not_minimal_without" for p in verify_evidence_sets(ep)))

    def test_reauditor_never_reads_truth(self):
        ep = copy.deepcopy(episode(3, "all"))
        truth = ep.truth
        ep.truth = {"episode_id": truth["episode_id"], "issues": []}
        rep = reaudit(ep)  # same findings, now all counted as FP -> shows they came from records
        self.assertEqual(rep["fp"], len(truth["issues"]))


class TestPolicies(unittest.TestCase):
    def test_oracle_auditor_naive_flagall(self):
        eps = [("x", s, None) for s in range(12)] + [("x", s, ALL_VARIANTS) for s in range(4)]
        rows = {r["policy"]: r for r in summarize(run_many(["oracle", "auditor", "naive", "flag_all"], eps))}
        self.assertEqual(rows["oracle"]["reward_min"], 1.0)
        self.assertEqual(rows["auditor"]["f1_micro"], 1.0)
        self.assertGreater(rows["auditor"]["reward_mean"], 0.85)
        self.assertGreater(rows["auditor"]["reward_mean"], rows["naive"]["reward_mean"] + 0.5)
        self.assertLessEqual(rows["flag_all"]["reward_mean"], -0.5)

    def test_random_explorer_is_deterministic(self):
        def go():
            env = LedgerTrapEnv(mode="train")
            traj, journal = run_episode(env, make_policy("random", seed=7), 4, None, "t", "random")
            return json.dumps(traj, sort_keys=True)
        self.assertEqual(go(), go())

    def test_trajectory_format_and_eval_silence(self):
        env = LedgerTrapEnv(mode="eval")
        traj, journal = run_episode(env, make_policy("auditor"), 6, None, "t", "auditor")
        self.assertEqual(traj[0]["step"], 0)
        self.assertIsNone(traj[0]["action"])
        for i, rec in enumerate(traj):
            self.assertTrue(TRAJ_FIELDS <= set(rec))
            self.assertEqual(rec["step"], i)
            self.assertEqual(rec["reward"], 0.0)
            json.dumps(rec)
        self.assertTrue(traj[-1]["terminated"])
        self.assertGreater(journal["metrics"]["reward_main"], 0.8)

    def test_auditor_stays_within_budget_on_largest_books(self):
        cfg = {"size": {"ledger_rows": [2000, 2000], "documents": [190, 200]}, "errors": {"count": [8, 12]}}
        for seed in range(3):
            env = LedgerTrapEnv(mode="train")
            traj, journal = run_episode(env, make_policy("auditor"), seed, cfg, "t", "auditor")
            self.assertEqual(journal["reason"], "submitted")
            self.assertLess(journal["steps"], 100)


if __name__ == "__main__":
    unittest.main()
