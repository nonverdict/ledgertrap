"""duplicate_invoice: `document` (second identical invoice) and `double_posting`."""

from datetime import timedelta

from ..money import conv
from ..posting import add_row, posting_moment
from ..world import display_offset, new_invoice
from .common import has_payments, invoice_group, new_issue, snap_doc, snap_row


def eligible_document(world, used):
    out = []
    for d in world.invoices():
        if d["kind"] != "invoice" or d["opening"] or d["key"] in used:
            continue
        if world.interval(d["issued_at"]) != "period":
            continue
        if d["issued_at"] + timedelta(hours=2) >= world.period_end:
            continue
        out.append(d["key"])
    return sorted(out)


MAX_DOCUMENTS = 200  # case §03 upper bound; a duplicate document must not exceed it


def apply_document(rng, world, used, same_prob):
    if len(world.docs) >= MAX_DOCUMENTS:
        return None
    cands = eligible_document(world, used)
    if not cands:
        return None
    d1 = world.docs[rng.choice(cands)]
    room = world.period_end - d1["issued_at"]
    max_days = min(10, max(0, room.days - 1))
    for _ in range(20):
        delta = timedelta(days=rng.randint(0, max_days), hours=rng.randint(0, 10), minutes=rng.randint(1, 59))
        issued = d1["issued_at"] + delta
        if issued < world.period_end:
            break
    else:
        return None
    terms = (d1["due_at"] - d1["issued_at"]).days
    lines = [dict(ln) for ln in d1["lines"]]
    d2 = new_invoice(world, d1["entity"], d1["currency"],
                     issued.astimezone(display_offset(rng, world, same_prob)), lines, terms)
    ts = posting_moment(rng, world, d2["issued_at"])
    rate = world.rate_for(d2)
    amt = conv(d2["total"], d2["currency"], world.base, rate, world.rounding)
    r1 = add_row(world, ts, "ar", "debit", amt, d2["entity"], d2["key"], None, d2["key"])
    r2 = add_row(world, ts, "revenue", "credit", amt, d2["entity"], d2["key"], None, d2["key"])
    used.update({d1["key"], d2["key"]})
    g1 = invoice_group(world, d1["key"])
    g2 = [r1, r2]
    evidence = [[d1["key"], d2["key"], a["key"], b["key"]] for a in g1 for b in g2]
    cert = {"kind": "duplicate_document", "d1": d1["key"], "d2": d2["key"], "rows_d1": [r["key"] for r in g1],
            "rows_d2": [r1["key"], r2["key"]], "amount": d2["total"], "currency": d2["currency"]}
    return new_issue("duplicate_invoice", "document", [d1["key"], d2["key"]], d2["total"], d2["currency"],
                     evidence, "occurrence", [("I9a", tuple(sorted((d1["key"], d2["key"]))))],
                     [{"op": "add_doc", "target": d2["key"], "after": snap_doc(d2)},
                      {"op": "add_rows", "target": d2["key"], "after": [snap_row(r1), snap_row(r2)]}], cert)


def eligible_double(world, used):
    out = []
    for d in world.invoices():
        if d["kind"] != "invoice" or d["opening"] or d["key"] in used:
            continue
        if world.interval(d["issued_at"]) != "period" or has_payments(world, d["key"]):
            continue
        out.append(d["key"])
    return sorted(out)


def apply_double(rng, world, used):
    cands = eligible_double(world, used)
    if not cands:
        return None
    x = world.docs[rng.choice(cands)]
    ts = posting_moment(rng, world, x["issued_at"], max_lag=2)
    rate = world.rate_for(x)
    amt = conv(x["total"], x["currency"], world.base, rate, world.rounding)
    orig = invoice_group(world, x["key"])
    r1 = add_row(world, ts, "ar", "debit", amt, x["entity"], x["key"], None, x["key"])
    r2 = add_row(world, ts, "revenue", "credit", amt, x["entity"], x["key"], None, x["key"])
    used.add(x["key"])
    ar0 = [r for r in orig if r["role"] == "ar"][0]
    rev0 = [r for r in orig if r["role"] == "revenue"][0]
    evidence = [[x["key"], ar0["key"], r1["key"]], [x["key"], rev0["key"], r2["key"]]]
    cert = {"kind": "double_posting", "doc": x["key"], "ar": [ar0["key"], r1["key"]],
            "rev": [rev0["key"], r2["key"]], "amount": x["total"], "currency": x["currency"], "base_amount": amt}
    return new_issue("duplicate_invoice", "double_posting", [x["key"]], x["total"], x["currency"], evidence,
                     "occurrence", [("I3", x["key"]), ("I5", x["key"]), ("I9b", x["key"])],
                     [{"op": "add_rows", "target": x["key"], "after": [snap_row(r1), snap_row(r2)]}], cert)
