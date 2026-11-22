"""period_cutoff: `early` (window document posted in the period) and `late` (period document
posted after period end). All rows of the document's posting group move together."""

from datetime import timedelta

from ..posting import business_time
from .common import doc_group, new_issue, snap_row


def _amount(doc):
    return abs(doc["total"]) if doc["kind"] != "payment" else abs(doc["amount"])


def eligible(world, used, variant):
    want = "window" if variant == "early" else "period"
    out = []
    for d in world.docs.values():
        if d["key"] in used:
            continue
        if d["kind"] == "credit_note":
            continue
        if d["kind"] == "invoice" and d["opening"]:
            continue
        if d["kind"] == "payment" and d["amount"] <= 0:
            continue
        if world.interval(world.doc_moment(d)) != want:
            continue
        if not doc_group(world, d["key"]):
            continue
        out.append(d["key"])
    return sorted(out)


def apply(rng, world, used, variant, window_days):
    cands = eligible(world, used, variant)
    if not cands:
        return None
    doc = world.docs[rng.choice(cands)]
    end_d = world.period_end.astimezone(world.tz).date()
    if variant == "early":
        new_d = end_d - timedelta(days=rng.randint(1, 3))
    else:
        new_d = end_d + timedelta(days=rng.randint(0, min(3, window_days - 1)))
    ts = business_time(rng, world, new_d)
    group = doc_group(world, doc["key"])
    before = [snap_row(r) for r in group]
    for r in group:
        r["ts"] = ts
    after = [snap_row(r) for r in group]
    used.add(doc["key"])
    evidence = [[doc["key"], r["key"]] for r in group]
    cert = {"kind": "cutoff_" + variant, "doc": doc["key"], "rows": [r["key"] for r in group],
            "amount": _amount(doc), "currency": doc["currency"]}
    return new_issue("period_cutoff", variant, [doc["key"]], _amount(doc), doc["currency"], evidence, "cutoff",
                     [("I8", doc["key"])],
                     [{"op": "update_rows", "target": doc["key"], "before": before, "after": after}], cert)
