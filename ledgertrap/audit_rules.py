"""Audit rules over PUBLIC records only (manifest, policy, ledger, invoices, payments, fx).

Used by the independent re-auditor (full files) and by our scripted auditor policy (records
gathered through tools). Never imports the generator, truth or provenance. Shares only the
published money/time helpers.
"""

from collections import defaultdict
from decimal import Decimal

from .money import CURRENCY_EXP, conv, format_minor, format_rate, unconv
from .timeutil import last_business_on_or_before, parse, parse_offset

TYPE_DUP, TYPE_ALLOC, TYPE_FX, TYPE_CUTOFF = ("duplicate_invoice", "payment_allocation", "fx_conversion",
                                              "period_cutoff")


def _signed_debit(r):
    return r["debit_minor"] - r["credit_minor"]


def _signed_credit(r):
    return r["credit_minor"] - r["debit_minor"]


class Books:
    """Indexes over public records. `lines` maps doc_id -> line_items when known."""

    def __init__(self, manifest, policy, ledger, invoices, payments, fx, lines=None):
        self.manifest, self.policy = manifest, policy
        self.tz = parse_offset(manifest["timezone"])
        self.ps, self.pe = parse(manifest["period_start"]), parse(manifest["period_end"])
        self.base = manifest["base_currency"]
        self.rounding = policy.get("rounding", manifest.get("rounding"))
        acc = policy["accounts"]
        self.role_of = {}
        for role, code in acc.items():
            for c in (code if isinstance(code, list) else [code]):
                self.role_of[c] = role
        self.fx = {}
        for f in fx:
            self.fx[(f["date"], f["source_currency"])] = Decimal(f["rate"])
        self.fx_ids = {"%s:%s:%s" % (f["date"], f["source_currency"], f["target_currency"]) for f in fx}
        self.invoices = list(invoices)
        self.payments = list(payments)
        self.inv_by_num = {d["invoice_id"]: d for d in self.invoices}
        self.pay_by_num = {p["payment_id"]: p for p in self.payments}
        self.lines = dict(lines or {})
        for d in self.invoices:
            if "line_items" in d:
                self.lines[d["doc_id"]] = d["line_items"]
        self.own_rows = defaultdict(list)
        self.pay_rows = defaultdict(list)
        for r in ledger:
            r = dict(r)
            r["role"] = self.role_of.get(r["account"], "unknown")
            if r["payment_id"] is not None:
                self.pay_rows[r["payment_id"]].append(r)
            elif r["invoice_id"] is not None:
                self.own_rows[r["invoice_id"]].append(r)
        for k in self.own_rows:
            self.own_rows[k].sort(key=lambda r: r["row_id"])
        for k in self.pay_rows:
            self.pay_rows[k].sort(key=lambda r: r["row_id"])
        self.refs_to = defaultdict(list)
        for p in self.payments:
            for ref in p["invoice_refs"] or []:
                self.refs_to[ref["invoice_id"]].append((p, ref["amount_minor"]))

    # --- policy rules -----------------------------------------------------------------
    @staticmethod
    def moment(doc):
        return parse(doc["issued_at"] if "issued_at" in doc else doc["timestamp"])

    def interval(self, dt):
        if dt < self.ps:
            return "pre"
        return "period" if dt < self.pe else "window"

    def rate_date(self, doc):
        return last_business_on_or_before(self.moment(doc).astimezone(self.tz).date()).isoformat()

    def fx_id(self, doc):
        return "%s:%s:%s" % (self.rate_date(doc), doc["currency"], self.base)

    def rate(self, doc):
        """None for base currency; raises KeyError if the policy rate is not available."""
        if doc["currency"] == self.base:
            return None
        return self.fx[(self.rate_date(doc), doc["currency"])]

    def conv(self, amount, doc):
        return conv(amount, doc["currency"], self.base, self.rate(doc), self.rounding)

    def ar_credit(self, inv, pay, a):
        pb = sum(x for p, x in self.refs_to[inv["invoice_id"]] if self.moment(p) < self.moment(pay))
        r = self.rate(inv)
        return (conv(pb + a, inv["currency"], self.base, r, self.rounding)
                - conv(pb, inv["currency"], self.base, r, self.rounding))

    def is_opening(self, inv):
        return any(r["role"] == "opening_equity" for r in self.own_rows.get(inv["invoice_id"], []))

    def fmt(self, amount, cur):
        return format_minor(amount, cur)


def _finding(t, affected, amount, cur, evidence, cert, variant):
    return {"type": t, "affected_ids": sorted(affected), "claimed_amount_minor": abs(amount), "currency": cur,
            "evidence_ids": list(evidence), "explanation": cert[:1000], "_variant": variant, "_certificate": cert}


# ------------------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------------------

def duplicate_candidates(b):
    """Groups of posted invoices that could be duplicates by header (entity, currency, total, lines)."""
    groups = defaultdict(list)
    for d in b.invoices:
        if d.get("kind", "invoice") != "invoice" or not b.own_rows.get(d["invoice_id"]):
            continue
        n = len(d["line_items"]) if "line_items" in d else d.get("line_count")
        groups[(d["entity_id"], d["currency"], d["total_minor"], n)].append(d)
    return [sorted(g, key=lambda d: (b.moment(d), d["doc_id"])) for g in groups.values() if len(g) >= 2]


def rule_duplicate_document(b):
    out = []
    for group in duplicate_candidates(b):
        known = [d for d in group if d["doc_id"] in b.lines]
        seen = []
        for d in known:
            twin = next((s for s in seen if b.lines[s["doc_id"]] == b.lines[d["doc_id"]]), None)
            if twin is None:
                seen.append(d)
                continue
            r1 = _key_row(b.own_rows[twin["invoice_id"]])
            r2 = _key_row(b.own_rows[d["invoice_id"]])
            if r1 is None or r2 is None:
                continue
            cert = ("Счета %s и %s: один клиент %s, валюта %s, позиции совпадают полностью (включая период услуги), "
                    "оба проведены (%s, %s). Сумма дубля %s." % (
                        twin["invoice_id"], d["invoice_id"], d["entity_id"], d["currency"], r1["row_id"],
                        r2["row_id"], b.fmt(d["total_minor"], d["currency"])))
            out.append(_finding(TYPE_DUP, [twin["doc_id"], d["doc_id"]], d["total_minor"], d["currency"],
                                [twin["doc_id"], d["doc_id"], r1["row_id"], r2["row_id"]], cert, "document"))
    return out


def _key_row(rows):
    """AR row of an invoice's own posting, else its revenue/returns row."""
    for role in ("ar", "revenue", "returns"):
        rs = [r for r in rows if r["role"] == role]
        if rs:
            return rs[0]
    return None


def rule_double_posting(b):
    out = []
    for d in b.invoices:
        own = b.own_rows.get(d["invoice_id"], [])
        for role in ("ar", "revenue"):
            rs = [r for r in own if r["role"] == role]
            if len(rs) >= 2:
                cert = ("Счёт %s (%s) проведён %d раза: строки роли %s %s; ожидается одна проводка. Сумма счёта %s." % (
                    d["invoice_id"], d["doc_id"], len(rs), role, ", ".join(r["row_id"] for r in rs),
                    b.fmt(d["total_minor"], d["currency"])))
                out.append(_finding(TYPE_DUP, [d["doc_id"]], d["total_minor"], d["currency"],
                                    [d["doc_id"], rs[0]["row_id"], rs[1]["row_id"]], cert, "double_posting"))
                break
    return out


def rule_cutoff(b):
    out = []
    docs = [(d, b.own_rows.get(d["invoice_id"], [])) for d in b.invoices if not b.is_opening(d)]
    docs += [(p, b.pay_rows.get(p["payment_id"], [])) for p in b.payments]
    for d, rows in docs:
        if not rows:
            continue
        di = b.interval(b.moment(d))
        ri = b.interval(parse(rows[0]["timestamp"]))
        if di == ri or di == "pre":
            continue
        amount = abs(d["total_minor"] if "total_minor" in d else d["amount_minor"])
        num = d.get("invoice_id") or d.get("payment_id")
        cert = ("Документ %s (%s): метка %s, по времени компании %s — интервал «%s» (период [%s, %s)). Проводка %s "
                "датирована %s — интервал «%s». Сумма %s." % (
                    num, d["doc_id"], b.moment(d).isoformat(), b.moment(d).astimezone(b.tz).strftime("%Y-%m-%d %H:%M"),
                    di, b.ps.isoformat(), b.pe.isoformat(), rows[0]["row_id"], rows[0]["timestamp"], ri,
                    b.fmt(amount, d["currency"])))
        out.append(_finding(TYPE_CUTOFF, [d["doc_id"]], amount, d["currency"], [d["doc_id"], rows[0]["row_id"]],
                            cert, "late" if di == "period" else "early"))
    return out


def rule_fx(b):
    out = []
    for d in b.invoices + b.payments:
        if d["currency"] == b.base:
            continue
        try:
            rate = b.rate(d)
        except KeyError:
            continue
        if "invoice_id" in d:
            own = b.own_rows.get(d["invoice_id"], [])
            rows = [r for r in own if r["role"] == "ar"]
            sign = _signed_debit
            if not rows:  # evidence may cite the revenue row instead
                rows = [r for r in own if r["role"] in ("revenue", "returns")]
                sign = _signed_credit if (rows and rows[0]["role"] == "revenue") else _signed_debit
                if rows and rows[0]["role"] == "returns":
                    sign = lambda r: -_signed_debit(r)  # noqa: E731  returns: Dr |conv(T)|, T < 0
            amount = d["total_minor"]
        else:
            rows = [r for r in b.pay_rows.get(d["payment_id"], []) if r["role"] == "bank"]
            sign = _signed_debit
            amount = d["amount_minor"]
        if not rows:
            continue
        right = conv(amount, d["currency"], b.base, rate, b.rounding)
        posted = sign(rows[0])
        if posted == right or any(sign(r) != posted for r in rows):
            continue
        diff = abs(posted - right)
        num = d.get("invoice_id") or d.get("payment_id")
        cert = ("Документ %s (%s) на %s, дата по времени компании %s; курс по политике %s (%s), правильно %s. "
                "Строка %s: %s. Расхождение %s." % (
                    num, d["doc_id"], b.fmt(amount, d["currency"]),
                    b.moment(d).astimezone(b.tz).date().isoformat(), format_rate(rate), b.fx_id(d),
                    b.fmt(right, b.base), rows[0]["row_id"], b.fmt(posted, b.base), b.fmt(diff, b.base)))
        out.append(_finding(TYPE_FX, [d["doc_id"]], diff, b.base, [d["doc_id"], rows[0]["row_id"], b.fx_id(d)],
                            cert, "fx"))
    return out


def rule_allocation(b):
    out = []
    for p in b.payments:
        refs = p["invoice_refs"]
        if not refs:
            continue
        rows = b.pay_rows.get(p["payment_id"], [])
        ar_rows = [r for r in rows if r["role"] == "ar"]
        adv_rows = [r for r in rows if r["role"] == "advances"]
        ref_nums = {x["invoice_id"] for x in refs}
        stray = [r for r in ar_rows if r["invoice_id"] not in ref_nums]
        u = p["amount_minor"] - sum(x["amount_minor"] for x in refs)
        for ref in refs:
            inv = b.inv_by_num.get(ref["invoice_id"])
            # Without the invoice record the canonical doc_id of X is unknown; the finding then names X by its
            # invoice_id and cannot match (affected ids are compared exactly), so X belongs to the evidence set.
            target = inv["doc_id"] if inv is not None else ref["invoice_id"]
            a = ref["amount_minor"]
            got = sum(_signed_credit(r) for r in ar_rows if r["invoice_id"] == ref["invoice_id"])
            if inv is not None:
                try:
                    exp = b.ar_credit(inv, p, a)
                except KeyError:  # rate of X not in view (evidence-only books): decide from the rows alone
                    exp = None
                if exp is not None and got == exp:
                    continue
            if got == 0 and stray:
                r = stray[0]
                cert = ("Платёж %s (%s) по invoice_refs распределён на %s (%s), но кредит дебиторки %s проведён по "
                        "счёту %s. Не зачтено: %s." % (p["payment_id"], p["doc_id"], ref["invoice_id"],
                                                       b.fmt(a, p["currency"]), r["row_id"], r["invoice_id"],
                                                       b.fmt(a, p["currency"])))
                out.append(_finding(TYPE_ALLOC, [p["doc_id"], target], a, p["currency"],
                                    [p["doc_id"], target, r["row_id"]], cert, "wrong_invoice"))
                continue
            if not adv_rows:
                continue
            adv = sum(_signed_credit(r) for r in adv_rows)
            try:
                rate = b.rate(p)
            except KeyError:
                continue
            m = unconv(adv, p["currency"], b.base, rate) - u
            if m <= 0:
                continue
            ev = [p["doc_id"], target, adv_rows[0]["row_id"]] + ([b.fx_id(p)] if p["currency"] != b.base else [])
            if inv is None and m <= 0:
                continue
            cert = ("Платёж %s (%s) на %s, распределение на %s — %s, остаток по документу %s. Авансы %s: %s, то есть "
                    "%s в валюте платежа; не зачтено в счёт %s." % (
                        p["payment_id"], p["doc_id"], b.fmt(p["amount_minor"], p["currency"]), ref["invoice_id"],
                        b.fmt(a, p["currency"]), b.fmt(u, p["currency"]), adv_rows[0]["row_id"],
                        b.fmt(adv, b.base), b.fmt(u + m, p["currency"]), b.fmt(m, p["currency"])))
            out.append(_finding(TYPE_ALLOC, [p["doc_id"], target], m, p["currency"], ev, cert,
                                "missing" if got == 0 else "partial"))
    return out


RULES = (rule_duplicate_document, rule_double_posting, rule_allocation, rule_fx, rule_cutoff)


def audit(b):
    findings = []
    for rule in RULES:
        findings.extend(rule(b))
    findings.sort(key=lambda f: (f["type"], f["affected_ids"]))
    return findings


def public_findings(findings):
    """Strip private helper fields before submit."""
    keys = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")
    return [{k: f[k] for k in keys} for f in findings]


def exp_of(cur):
    return CURRENCY_EXP[cur]
