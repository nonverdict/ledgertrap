"""Accounting equalities I1-I11 (DESIGN §9.2) over the internal world.

`check(world)` returns a sorted list of violations `(code, subject)`. Clean books must
return []. After mutations the violations must equal exactly the union of the
violations each declared issue is expected to cause (`declared discrepancy`).
"""

from collections import defaultdict

from .money import conv
from .posting import ar_credit


def _net(rows, role, sign_debit=True):
    s = 0
    for r in rows:
        if r["role"] == role:
            s += (r["debit"] - r["credit"]) if sign_debit else (r["credit"] - r["debit"])
    return s


def check(world):
    v = set()
    rows = list(world.rows.values())
    docs = world.docs
    base, rnd = world.base, world.rounding

    # I2: exactly one positive side, non-negative ints.
    for r in rows:
        ok = (isinstance(r["debit"], int) and isinstance(r["credit"], int) and r["debit"] >= 0
              and r["credit"] >= 0 and ((r["debit"] > 0) != (r["credit"] > 0)))
        if not ok:
            v.add(("I2", r["key"]))

    # I1: balance overall and per group.
    if sum(r["debit"] for r in rows) != sum(r["credit"] for r in rows):
        v.add(("I1", "*"))
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    for g, rs in groups.items():
        if sum(r["debit"] for r in rs) != sum(r["credit"] for r in rs):
            v.add(("I1", g))

    by_invoice = defaultdict(list)
    by_payment = defaultdict(list)
    for r in rows:
        if r["invoice"] is not None:
            by_invoice[r["invoice"]].append(r)
        if r["payment"] is not None:
            by_payment[r["payment"]].append(r)

    # I10: references.
    for r in rows:
        if r["invoice"] is not None and (r["invoice"] not in docs or docs[r["invoice"]]["kind"] == "payment"):
            v.add(("I10", r["key"]))
        if r["payment"] is not None and (r["payment"] not in docs or docs[r["payment"]]["kind"] != "payment"):
            v.add(("I10", r["key"]))
    for p in world.payments():
        if p["refs"] is None:
            continue
        total = 0
        for k, a in p["refs"]:
            inv = docs.get(k)
            if inv is None or inv["kind"] == "payment" or inv["currency"] != p["currency"]:
                v.add(("I10", p["key"]))
            if (a > 0) != (p["amount"] > 0) or a == 0:
                v.add(("I10", p["key"]))
            total += abs(a)
        if total > abs(p["amount"]):
            v.add(("I10", p["key"]))
    for d in docs.values():
        if d["kind"] == "credit_note" and (d["original"] not in docs):
            v.add(("I10", d["key"]))

    # I11: FX coverage.
    for d in docs.values():
        if d["currency"] != base and (world.rate_date(d), d["currency"]) not in world.fx:
            v.add(("I11", d["key"]))

    # I3 / I5 / I9b per invoice-like document.
    for d in world.invoices():
        k = d["key"]
        rate = world.rate_for(d)
        amt = conv(d["total"], d["currency"], base, rate, rnd)
        expected_ar = amt - sum(ar_credit(world, k, p, a) for p, a in world.payments_for(k))
        if _net(by_invoice[k], "ar") != expected_ar:
            v.add(("I3", k))
        own = [r for r in by_invoice[k] if r["payment"] is None]
        if d["kind"] == "credit_note":
            if _net(own, "returns") != -amt:
                v.add(("I5", k))
            roles = sorted(r["role"] for r in own)
            if roles != ["ar", "returns"]:
                v.add(("I9b", k))
        elif d["opening"]:
            roles = sorted(r["role"] for r in own)
            if roles != ["ar", "opening_equity"]:
                v.add(("I9b", k))
        else:
            if _net(own, "revenue", sign_debit=False) != amt:
                v.add(("I5", k))
            roles = sorted(r["role"] for r in own)
            if roles != ["ar", "revenue"]:
                v.add(("I9b", k))

    # I4 bank, I6 advances, I7 fx diff per payment.
    expected_bank = 0
    for p in world.payments():
        rate = world.rate_for(p)
        bank = conv(p["amount"], p["currency"], base, rate, rnd)
        expected_bank += bank
        rs = by_payment[p["key"]]
        refs = p["refs"] or []
        u = p["amount"] - sum(a for _, a in refs)
        adv = conv(u, p["currency"], base, rate, rnd)
        if _net(rs, "advances", sign_debit=False) != adv:
            v.add(("I6", p["key"]))
        ar_rows = [r for r in rs if r["role"] == "ar"]
        expected_ar = {}
        for k, a in refs:
            expected_ar[k] = expected_ar.get(k, 0) + ar_credit(world, k, p, a)
        got_ar = defaultdict(int)
        for r in ar_rows:
            got_ar[r["invoice"]] += r["credit"] - r["debit"]
        if dict((k, x) for k, x in got_ar.items() if x != 0) != dict((k, x) for k, x in expected_ar.items() if x != 0):
            v.add(("I3p", p["key"]))
        if _net(rs, "bank") != bank:
            v.add(("I4p", p["key"]))
        fx = _net(rs, "fx_gain", sign_debit=False) - _net(rs, "fx_loss")
        if fx != bank - sum(expected_ar.values()) - adv:
            v.add(("I7", p["key"]))
    expense_bank = sum(r["credit"] - r["debit"] for r in rows if r["role"] == "bank" and r["payment"] is None)
    if _net(rows, "bank") != expected_bank - expense_bank:
        v.add(("I4", "*"))

    # I8 intervals.
    for d in docs.values():
        k = d["key"]
        rs = by_payment[k] if d["kind"] == "payment" else [r for r in by_invoice[k] if r["payment"] is None]
        if d["kind"] != "payment" and d.get("opening"):
            if any(r["ts"] != world.period_start for r in rs):
                v.add(("I8", k))
            continue
        iv = world.interval(world.doc_moment(d))
        if any(world.interval(r["ts"]) != iv for r in rs):
            v.add(("I8", k))
        if any(r["ts"] < world.export_start or r["ts"] >= world.export_end for r in rs):
            v.add(("I8", k))
    for d in docs.values():
        m = world.doc_moment(d)
        if not (world.export_start <= m < world.export_end):
            v.add(("I8", d["key"]))

    # I9a identical lines per entity; I9c distinct payment moments per invoice.
    seen = {}
    for d in sorted(world.invoices(), key=lambda x: x["key"]):
        sig = (d["entity"], d["currency"], tuple(sorted(
            (ln["sku"], ln["description"], ln["quantity"], ln["unit_price_minor"], ln["amount_minor"],
             ln["service_period_start"], ln["service_period_end"]) for ln in d["lines"])))
        if sig in seen:
            v.add(("I9a", tuple(sorted((seen[sig], d["key"])))))
        else:
            seen[sig] = d["key"]
        moments = [p["timestamp"] for p, _ in world.payments_for(d["key"])]
        if len(set(moments)) != len(moments):
            v.add(("I9c", d["key"]))
        if sum(ln["amount_minor"] for ln in d["lines"]) != d["total"] or any(
                ln["quantity"] * ln["unit_price_minor"] != ln["amount_minor"] for ln in d["lines"]):
            v.add(("I5l", d["key"]))

    return sorted(v, key=lambda x: (x[0], str(x[1])))
