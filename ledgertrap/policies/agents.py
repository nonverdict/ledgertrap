"""Policy adapters (case §06/§09): random explorer, naive reconciler, flag-all, oracle, and our auditor.

Interface: policy.reset(observation, info); policy.act(observation) -> action dict.
Agents see only observations. The oracle runs on the grader side and receives the episode from
the trusted runner.
"""

import hashlib
from collections import defaultdict
from decimal import Decimal

from ..audit_rules import Books, audit, duplicate_candidates, public_findings
from ..money import conv
from ..rng import DetRng
from .baselines import ISSUE_TYPES

TABLES = ("fx", "invoices", "payments", "ledger")


def submit(episode_id, findings):
    return {"tool": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings[:30]}}}


class _Reader:
    """Pages through tables, then lets the subclass decide. Keeps one step for submit."""

    tables = TABLES

    def reset(self, obs, info=None):
        self.manifest = obs["manifest"]
        self.policy = obs["policy"]
        self.episode_id = self.manifest["episode_id"]
        self.data = {t: [] for t in TABLES}
        self.queue = list(self.tables)
        self.cursor = None
        self.steps_left = obs["budget"]["steps_left"]

    def _absorb(self, obs):
        if obs.get("type") == "tool_result":
            self.steps_left = obs["steps_left"]
            if obs.get("ok") and obs["tool"] == "list_records":
                self.data[self.queue[0]].extend(obs["result"]["rows"])
                self.cursor = obs["result"]["next_cursor"]
                if self.cursor is None:
                    self.queue.pop(0)
            elif obs.get("tool") == "list_records" and not obs.get("ok"):
                self.queue.pop(0)
                self.cursor = None
            return True
        return False

    def read_action(self):
        if self.queue and self.steps_left > 1:
            return {"tool": "list_records", "args": {"table": self.queue[0], "cursor": self.cursor, "limit": 50}}
        return None


AUDITOR_LATEST = 2
AUDITOR_VERSIONS = {
    0: "reads all four tables page by page, then get_document for every duplicate candidate",
    1: "v0 without the get_document calls: list_records already returns invoices with line_items",
    2: "v1, and the fx table is not read when the manifest lists only the base currency",
}


class AuditorPolicy(_Reader):
    """Our strong scripted auditor: reads the tables, then applies every policy rule (timezones, weekend
    rates, line items, cumulative AR settlement). Versions are the harness-engineering iterations
    (stage 7, level 1); see AUDITOR_VERSIONS and reports/diagnostics.md, section 9."""

    def __init__(self, version=AUDITOR_LATEST):
        if version not in AUDITOR_VERSIONS:
            raise ValueError("unknown auditor version %r" % version)
        self.version = version
        self.name = "auditor" if version == AUDITOR_LATEST else "auditor_v%d" % version

    def reset(self, obs, info=None):
        super().reset(obs, info)
        if self.version >= 2 and set(self.manifest.get("currencies", {})) <= {self.manifest["base_currency"]}:
            self.queue = [t for t in self.queue if t != "fx"]
        self.lines = {}
        self.doc_queue = None

    def act(self, obs):
        self._absorb(obs)
        if obs.get("tool") == "get_document" and obs.get("ok"):
            d = obs["result"]["document"]
            self.lines[d["doc_id"]] = d["line_items"]
        a = self.read_action()
        if a is not None:
            return a
        if self.doc_queue is None:
            b = self._books()
            self.doc_queue = [d["doc_id"] for g in duplicate_candidates(b) for d in g
                              if self.version == 0 or "line_items" not in d]
        while self.doc_queue and self.steps_left > 1:
            doc_id = self.doc_queue.pop(0)
            if doc_id not in self.lines:
                return {"tool": "get_document", "args": {"doc_id": doc_id}}
        return submit(self.episode_id, public_findings(audit(self._books())))

    def _books(self):
        d = self.data
        return Books(self.manifest, self.policy, d["ledger"], d["invoices"], d["payments"], d["fx"], self.lines)


class NaivePolicy(_Reader):
    """Simple reconciliation, deliberately naive: dates are read from the timestamp STRING (no timezone),
    FX is taken on the POSTING date with no weekend fallback, duplicates are matched by entity and total
    (line items and service periods are ignored), any advances next to invoice_refs look like a short
    allocation."""

    name = "naive"
    tables = ("fx", "invoices", "payments", "ledger")

    def act(self, obs):
        self._absorb(obs)
        a = self.read_action()
        if a is not None:
            return a
        return submit(self.episode_id, self._findings())

    def _findings(self):
        m, d = self.manifest, self.data
        base = m["base_currency"]
        p_start, p_end = m["period_start"][:10], m["period_end"][:10]
        fx = {(f["date"], f["source_currency"]): f["rate"] for f in d["fx"]}
        own, pay_rows = defaultdict(list), defaultdict(list)
        for r in d["ledger"]:
            if r["payment_id"]:
                pay_rows[r["payment_id"]].append(r)
            elif r["invoice_id"]:
                own[r["invoice_id"]].append(r)
        acc = self.policy["accounts"]
        out = []

        def add(t, affected, amount, cur, ev, why):
            out.append({"type": t, "affected_ids": affected, "claimed_amount_minor": abs(int(amount)),
                        "currency": cur, "evidence_ids": ev[:10], "explanation": why})

        def in_period(s):
            return p_start <= s[:10] < p_end

        # duplicates: same entity and total
        seen = {}
        for inv in d["invoices"]:
            if inv.get("kind") != "invoice":
                continue
            k = (inv["entity_id"], inv["total_minor"], inv["currency"])
            if k in seen and own.get(inv["invoice_id"]) and own.get(seen[k]["invoice_id"]):
                a = seen[k]
                add("duplicate_invoice", [a["doc_id"], inv["doc_id"]], inv["total_minor"], inv["currency"],
                    [a["doc_id"], inv["doc_id"], own[a["invoice_id"]][0]["row_id"], own[inv["invoice_id"]][0]["row_id"]],
                    "same entity and total")
            seen.setdefault(k, inv)
        docs = [(x, own.get(x["invoice_id"], []), x["issued_at"], x["total_minor"]) for x in d["invoices"]]
        docs += [(x, pay_rows.get(x["payment_id"], []), x["timestamp"], x["amount_minor"]) for x in d["payments"]]
        for doc, rows, ts, amount in docs:
            if not rows:
                continue
            # cutoff by date strings
            if in_period(ts) != in_period(rows[0]["timestamp"]):
                add("period_cutoff", [doc["doc_id"]], amount, doc["currency"], [doc["doc_id"], rows[0]["row_id"]],
                    "document date vs posting date")
            # fx by posting date, exact date only
            if doc["currency"] != base:
                key_acc = acc["bank"] if "payment_id" in doc else acc["ar"]
                key_rows = [r for r in rows if r["account"] == key_acc]
                pdate = rows[0]["timestamp"][:10]
                rate = fx.get((pdate, doc["currency"]))  # posting date, no weekend fallback
                if key_rows and rate is not None:
                    want = conv(amount, doc["currency"], base, Decimal(rate), m["rounding"])
                    got = key_rows[0]["debit_minor"] - key_rows[0]["credit_minor"]
                    if got != want:
                        add("fx_conversion", [doc["doc_id"]], abs(got - want), base,
                            [doc["doc_id"], key_rows[0]["row_id"], "%s:%s:%s" % (pdate, doc["currency"], base)],
                            "rate of posting date")
        # allocation: refs without a matching AR row, or advances next to refs
        for p in d["payments"]:
            refs = p["invoice_refs"] or []
            rows = pay_rows.get(p["payment_id"], [])
            ar_inv = {r["invoice_id"] for r in rows if r["account"] == acc["ar"]}
            adv = [r for r in rows if r["account"] == acc["advances"]]
            inv_by_num = {x["invoice_id"]: x for x in d["invoices"]}
            for ref in refs:
                x = inv_by_num.get(ref["invoice_id"])
                if x is None:
                    continue
                if ref["invoice_id"] not in ar_inv and adv:
                    add("payment_allocation", [p["doc_id"], x["doc_id"]], ref["amount_minor"], p["currency"],
                        [p["doc_id"], adv[0]["row_id"]], "no AR credit for referenced invoice")
                elif adv:
                    add("payment_allocation", [p["doc_id"], x["doc_id"]],
                        p["amount_minor"] - sum(r["amount_minor"] for r in refs), p["currency"],
                        [p["doc_id"], adv[0]["row_id"]], "advances next to invoice_refs")
        return out


class FlagAllPolicy(_Reader):
    """Attack: lists documents and the ledger, then flags up to 30 documents with every type, citing the
    document and all of its rows as evidence."""

    name = "flag_all"

    def act(self, obs):
        self._absorb(obs)
        a = self.read_action()
        if a is not None:
            return a
        rows_of = defaultdict(list)
        for r in self.data["ledger"]:
            rows_of[r["payment_id"] or r["invoice_id"]].append(r["row_id"])
        docs = [(x["doc_id"], x["invoice_id"], abs(x["total_minor"]), x["currency"]) for x in self.data["invoices"]]
        docs += [(x["doc_id"], x["payment_id"], abs(x["amount_minor"]), x["currency"]) for x in self.data["payments"]]
        out = []
        for i in range(min(30, 4 * len(docs))):
            doc_id, num, amount, cur = docs[i % len(docs)]
            t = ISSUE_TYPES[(i // max(1, len(docs)) + i) % 4]
            out.append({"type": t, "affected_ids": [doc_id], "claimed_amount_minor": amount, "currency": cur,
                        "evidence_ids": ([doc_id] + rows_of.get(num, []))[:10], "explanation": "flag all"})
        return submit(self.episode_id, out)


class RandomPolicy:
    """Random explorer with a fixed seed (the stream also depends on the episode id)."""

    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, obs, info=None):
        self.episode_id = obs["manifest"]["episode_id"]
        key = hashlib.sha256(("random|%d|%s" % (self.seed, self.episode_id)).encode()).digest()
        self.rng = DetRng(key)
        self.submit_at = self.rng.randint(20, 99)
        self.step = 0
        self.ids = {"doc": [], "row": [], "fx": []}
        self.docs = {}
        self.cursors = {t: None for t in TABLES}

    def _remember(self, obs):
        res = obs.get("result") or {}
        rows = res.get("rows") or ([res["record"]] if "record" in res else []) + (
            [res["document"]] if "document" in res else [])
        for r in rows:
            if "row_id" in r:
                self.ids["row"].append(r["row_id"])
            if "doc_id" in r:
                self.ids["doc"].append(r["doc_id"])
                self.docs[r["doc_id"]] = r
            if "source_currency" in r:
                self.ids["fx"].append("%s:%s:%s" % (r["date"], r["source_currency"], r["target_currency"]))

    def act(self, obs):
        self.step += 1
        self._remember(obs)
        r = self.rng
        if self.step >= self.submit_at:
            findings = []
            for _ in range(r.randint(0, 3)):
                if not self.ids["doc"]:
                    break
                doc = self.docs[r.choice(self.ids["doc"])]
                amount = abs(doc.get("total_minor", doc.get("amount_minor", 0)))
                ev = [doc["doc_id"]] + [r.choice(self.ids["row"]) for _ in range(r.randint(0, 3)) if self.ids["row"]]
                findings.append({"type": r.choice(ISSUE_TYPES), "affected_ids": [doc["doc_id"]],
                                 "claimed_amount_minor": amount, "currency": doc["currency"], "evidence_ids": ev,
                                 "explanation": "random"})
            return submit(self.episode_id, findings)
        tool = r.choice(["list_records", "list_records", "get_record", "get_document", "calculator"])
        if tool == "list_records":
            t = r.choice(TABLES)
            return {"tool": "list_records", "args": {"table": t, "cursor": None, "limit": r.randint(1, 50)}}
        if tool == "get_document" and self.ids["doc"]:
            return {"tool": "get_document", "args": {"doc_id": r.choice(self.ids["doc"])}}
        if tool == "get_record" and self.ids["row"]:
            return {"tool": "get_record", "args": {"table": "ledger", "id": r.choice(self.ids["row"])}}
        return {"tool": "calculator", "args": {"expression": "%d*%d/100" % (r.randint(1, 10**6), r.randint(1, 999))}}


class SilentPolicy:
    name = "silent"

    def reset(self, obs, info=None):
        self.episode_id = obs["manifest"]["episode_id"]

    def act(self, obs):
        return submit(self.episode_id, [])


class OraclePolicy:
    """Grader-side only: the trusted runner hands it the grader's oracle answer."""

    name = "oracle"

    def __init__(self, findings_provider):
        self.provider = findings_provider

    def reset(self, obs, info=None):
        self.findings = self.provider()

    def act(self, obs):
        return {"tool": "submit", "args": {"findings": self.findings}}


def make_policy(name, env=None, seed=0):
    if name == "auditor":
        return AuditorPolicy()
    if name.startswith("auditor_v") and name[9:].isdigit():
        return AuditorPolicy(int(name[9:]))
    if name == "naive":
        return NaivePolicy()
    if name == "flag_all":
        return FlagAllPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "silent":
        return SilentPolicy()
    if name == "oracle":
        return OraclePolicy(env.oracle_findings)
    raise ValueError("unknown policy %r" % name)
