"""Generator: determinism, schemas/formats, invariants, truth consistency, presentation streams."""

import csv
import io
import json
import re
import unittest

from ledgertrap import invariants
from ledgertrap.generator import LEDGER_COLUMNS, generate
from ledgertrap.money import CURRENCY_EXP
from ledgertrap.timeutil import ISO_RE, parse
from tests.helpers import CLEAN, DIRTY, episode

RATE_RE = re.compile(r"^\d+(\.\d+)?$")


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_bytes(self):
        for seed in (0, 1, 1000, 2**40):
            a, b = generate(seed), generate(seed)
            self.assertEqual(a.files, b.files)
            self.assertEqual(a.grader_files(), b.grader_files())

    def test_secret_changes_episode(self):
        self.assertNotEqual(generate(5).files, generate(5, secret="jury-secret").files)
        self.assertEqual(generate(5, secret="s").files, generate(5, secret="s").files)

    def test_bad_seed(self):
        for bad in (-1, 2**63, 1.0, "1", True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                generate(bad)

    def test_presentation_changes_ids_not_story(self):
        a = generate(11, DIRTY)
        b = generate(11, dict(DIRTY, presentation={"id_salt": 1, "order_seed": 1, "order": "shuffle"}))
        self.assertNotEqual(a.files["ledger.csv"], b.files["ledger.csv"])

        def story(ep):
            rows = sorted((r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"])
                          for r in ep.records["ledger"])
            issues = sorted((g["type"], g["variant"], g["expected_amount_minor"], g["currency"])
                            for g in ep.truth["issues"])
            return rows, issues, ep.manifest["counts"]
        self.assertEqual(story(a), story(b))
        ids_a = {r["row_id"] for r in a.records["ledger"]}
        ids_b = {r["row_id"] for r in b.records["ledger"]}
        self.assertFalse(ids_a & ids_b)
        self.assertNotEqual(a.episode_id, b.episode_id)


class TestFormats(unittest.TestCase):
    def check_episode(self, ep):
        m = ep.manifest
        for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                  "counts", "rounding", "currencies", "export_start", "export_end"):
            self.assertIn(k, m)
        self.assertNotIn("config_hash", m)
        self.assertEqual(m["counts"]["ledger"], len(ep.records["ledger"]))
        self.assertEqual(m["counts"]["documents"], len(ep.records["invoices"]) + len(ep.records["payments"]))
        self.assertTrue(200 <= m["counts"]["ledger"] <= 2000)
        self.assertTrue(20 <= m["counts"]["documents"] <= 200)
        self.assertLessEqual(len(ep.truth["issues"]), 12)
        # ledger.csv header and row semantics
        rows = list(csv.reader(io.StringIO(ep.files["ledger.csv"].decode("utf-8"))))
        self.assertEqual(rows[0], LEDGER_COLUMNS)
        for r in ep.records["ledger"]:
            self.assertTrue(ISO_RE.match(r["timestamp"]))
            self.assertIsInstance(r["debit_minor"], int)
            self.assertIsInstance(r["credit_minor"], int)
            self.assertTrue((r["debit_minor"] > 0) != (r["credit_minor"] > 0))
            self.assertEqual(r["currency"], m["base_currency"])
        for d in ep.records["invoices"]:
            self.assertTrue(ISO_RE.match(d["issued_at"]) and ISO_RE.match(d["due_at"]))
            self.assertEqual(sum(x["amount_minor"] for x in d["line_items"]), d["total_minor"])
            for x in d["line_items"]:
                self.assertEqual(x["quantity"] * x["unit_price_minor"], x["amount_minor"])
            self.assertIn(d["currency"], m["currencies"])
        for p in ep.records["payments"]:
            self.assertTrue(ISO_RE.match(p["timestamp"]))
            self.assertTrue(p["invoice_refs"] is None or isinstance(p["invoice_refs"], list))
        for f in ep.records["fx"]:
            self.assertTrue(RATE_RE.match(f["rate"]), f["rate"])
            self.assertEqual(f["target_currency"], m["base_currency"])
        for cur, exp in m["currencies"].items():
            self.assertEqual(CURRENCY_EXP[cur], exp)
        # jsonl lines parse; no truth markers in public files
        for name in ("invoices.jsonl", "payments.jsonl"):
            for line in ep.files[name].decode("utf-8").splitlines():
                json.loads(line)
        blob = b"".join(ep.files.values()).decode("utf-8")
        for marker in ("ISS-", "CERT-", "is_clean", "evidence_sets", "expected_amount_minor", "variant\""):
            self.assertNotIn(marker, blob)

    def test_formats_many_seeds(self):
        for seed in range(40):
            self.check_episode(generate(seed))
        for seed in range(5):
            self.check_episode(episode(seed, "dirty"))
            self.check_episode(episode(seed, "clean"))

    def test_csv_null_is_empty_cell(self):
        ep = generate(3)
        text = ep.files["ledger.csv"].decode("utf-8")
        self.assertNotIn("None", text)
        self.assertNotIn("null", text)
        self.assertEqual(ep.manifest["csv_null"], "")


class TestTruth(unittest.TestCase):
    def test_clean_mode_has_no_issues(self):
        for seed in range(5):
            ep = episode(seed, "clean")
            self.assertEqual(ep.truth["issues"], [])
            self.assertTrue(ep.truth["is_clean"])

    def test_exact_plan_is_placed(self):
        for seed in range(8):
            ep = episode(seed, "dirty")
            got = sorted("%s:%s" % (g["type"], g["variant"]) for g in ep.truth["issues"])
            self.assertEqual(got, sorted(DIRTY["errors"]["exact"]))

    def test_truth_references_public_records(self):
        for seed in range(8):
            ep = episode(seed, "dirty")
            docs = {d["doc_id"]: d for d in ep.records["invoices"] + ep.records["payments"]}
            rows = {r["row_id"]: r for r in ep.records["ledger"]}
            fx = {"%s:%s:%s" % (f["date"], f["source_currency"], f["target_currency"]) for f in ep.records["fx"]}
            keys = set()
            for g in ep.truth["issues"]:
                key = (g["type"], tuple(g["affected_ids"]))
                self.assertNotIn(key, keys)
                keys.add(key)
                for a in g["affected_ids"]:
                    self.assertIn(a, docs)
                self.assertTrue(g["evidence_sets"])
                for s in g["evidence_sets"]:
                    for x in s:
                        self.assertTrue(x in docs or x in rows or x in fx, x)
                self.assertGreaterEqual(g["expected_amount_minor"], 100)

    def test_issue_semantics_visible_in_records(self):
        """Each declared issue is visible from public records alone (independent spot check)."""
        for seed in range(8):
            ep = episode(seed, "dirty")
            m = ep.manifest
            ps, pe = parse(m["period_start"]), parse(m["period_end"])
            docs = {d["doc_id"]: d for d in ep.records["invoices"] + ep.records["payments"]}
            rows = {r["row_id"]: r for r in ep.records["ledger"]}
            for g in ep.truth["issues"]:
                ev = g["evidence_sets"][0]
                if g["type"] == "period_cutoff":
                    d = docs[g["affected_ids"][0]]
                    when = parse(d.get("issued_at") or d["timestamp"])
                    r = rows[[x for x in ev if x in rows][0]]
                    in_period_doc = ps <= when < pe
                    in_period_row = ps <= parse(r["timestamp"]) < pe
                    self.assertNotEqual(in_period_doc, in_period_row)
                    self.assertEqual(g["variant"], "late" if in_period_doc else "early")
                    amt = abs(d.get("total_minor", d.get("amount_minor")))
                    self.assertEqual(g["expected_amount_minor"], amt)
                elif g["variant"] == "document":
                    d1, d2 = (docs[x] for x in g["affected_ids"])
                    self.assertEqual(d1["entity_id"], d2["entity_id"])
                    self.assertEqual(d1["line_items"], d2["line_items"])
                    self.assertEqual(g["expected_amount_minor"], d2["total_minor"])
                else:
                    x = docs[g["affected_ids"][0]]
                    own = [r for r in ep.records["ledger"]
                           if r["invoice_id"] == x["invoice_id"] and r["payment_id"] is None]
                    self.assertEqual(len(own), 4)


class TestInvariants(unittest.TestCase):
    def test_clean_world_has_no_violations_many_configs(self):
        # generate() raises if clean books violate I1-I11 or if post-mutation discrepancy is undeclared.
        configs = [None, CLEAN, DIRTY,
                   {"company": {"base_currency": "USD", "timezone": "-05:00", "chart_template": "alpha_codes",
                                "period_kind": "quarter", "foreign_currencies": ["JPY", "KWD"],
                                "n_foreign_currencies": [2, 2]}},
                   {"company": {"timezone": "+08:00", "rounding": "HALF_EVEN"},
                    "size": {"ledger_rows": [1500, 2000], "documents": [150, 200]}},
                   {"size": {"ledger_rows": [200, 200], "documents": [20, 20]}}]
        for cfg in configs:
            for seed in range(12):
                generate(seed, cfg)

    def test_mutation_detected_by_invariants(self):
        from ledgertrap import world as W
        from ledgertrap.story import build_story
        from ledgertrap.config import load
        from ledgertrap.rng import DetRng
        cfg = load(None)
        w = W.World()
        rng = DetRng(b"t")
        W.build_company(rng.fork("c"), cfg, w)
        W.build_period(rng.fork("p"), cfg, w)
        W.build_fx(rng.fork("f"), w)
        W.build_customers(rng.fork("u"), cfg, w)
        build_story(rng.fork("s"), cfg, w)
        self.assertEqual(invariants.check(w), [])
        row = next(r for r in w.rows.values() if r["role"] == "revenue")
        row["credit"] += 1
        codes = {c for c, _ in invariants.check(w)}
        self.assertIn("I1", codes)
        self.assertIn("I5", codes)


if __name__ == "__main__":
    unittest.main()
