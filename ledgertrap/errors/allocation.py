"""payment_allocation: `missing`, `partial`, `wrong_invoice` (DESIGN §7.3).

Only payments with exactly one invoice_refs entry, inside the period. Only the payment's own
posting group is rewritten; the FX difference row is recomputed so the group stays balanced.
"""

from ..money import conv, unconv
from ..posting import ar_credit, rewrite_payment
from .common import fx_key, new_issue, snap_row

MIN_AMOUNT = 100


def eligible(world, used):
    out = []
    for p in world.payments():
        if p["key"] in used or p["amount"] <= 0 or not p["refs"] or len(p["refs"]) != 1:
            continue
        if world.interval(p["timestamp"]) != "period":
            continue
        x, a = p["refs"][0]
        if x in used or world.docs[x]["kind"] != "invoice" or a < MIN_AMOUNT:
            continue
        out.append(p["key"])
    return sorted(out)


def _open_balance(world, inv_key):
    inv = world.docs[inv_key]
    return inv["total"] - sum(a for _, a in world.payments_for(inv_key))


def _paid_before(world, inv_key, moment):
    return sum(a for p, a in world.payments_for(inv_key) if p["timestamp"] < moment)


def wrong_invoice_targets(world, used, p, x):
    out = []
    for y in world.invoices():
        if y["key"] in used or y["key"] == x or y["kind"] != "invoice":
            continue
        if y["entity"] != p["entity"] or y["currency"] != p["currency"]:
            continue
        if y["issued_at"] >= p["timestamp"]:
            continue
        if _open_balance(world, y["key"]) < p["refs"][0][1]:
            continue
        out.append(y["key"])
    return sorted(out)


def apply(rng, world, used, variant, scenario_y=None):
    cands = eligible(world, used)
    rng.shuffle(cands)
    for pk in cands[:40]:
        p = world.docs[pk]
        x, a = p["refs"][0]
        if scenario_y is not None and x != scenario_y[0]:
            continue
        rate = world.rate_for(p)
        cur, base, rnd = p["currency"], world.base, world.rounding
        bank = conv(p["amount"], cur, base, rate, rnd)
        u = p["amount"] - a
        if variant in ("missing", "partial"):
            m = a if variant == "missing" else a * rng.randint(20, 80) // 100
            # The published amount rule is unconv(advances) - u (tolerance 0). A partial amount is snapped to
            # the value that rule recovers; any target whose amount the rule cannot recover exactly is skipped.
            if variant == "partial":
                m = unconv(conv(u + m, cur, base, rate, rnd), cur, base, rate) - u
            if unconv(conv(u + m, cur, base, rate, rnd), cur, base, rate) - u != m:
                continue
            if m < MIN_AMOUNT or (variant == "partial" and m >= a):
                continue
            ars = [] if variant == "missing" else [(x, ar_credit(world, x, p, a - m))]
            adv = conv(u + m, cur, base, rate, rnd)
            before, after = rewrite_payment(world, p, bank, ars, adv)
            adv_row = [r for r in after if r["role"] == "advances"][0]
            # invoice X's document is part of the set: without it the canonical doc_id of X is not knowable
            ev = [pk, x, adv_row["key"]] + ([fx_key(world, p)] if cur != base else [])
            used.update({pk, x})
            cert = {"kind": "alloc_" + variant, "pay": pk, "inv": x, "a": a, "m": m, "u": u,
                    "adv_row": adv_row["key"], "adv": adv, "rate": rate, "currency": cur}
            return new_issue("payment_allocation", variant, [pk, x], m, cur, [ev], "accuracy_classification",
                             [("I3", x), ("I3p", pk)],
                             [{"op": "update_rows", "target": pk, "before": [snap_row(r) for r in before],
                               "after": [snap_row(r) for r in after]}], cert)
        # wrong_invoice
        ys = [scenario_y[1]] if scenario_y is not None else wrong_invoice_targets(world, used, p, x)
        if not ys:
            continue
        y = world.docs[rng.choice(ys)]
        ry = world.rate_for(y)
        pb = _paid_before(world, y["key"], p["timestamp"])
        credit_y = conv(pb + a, cur, base, ry, rnd) - conv(pb, cur, base, ry, rnd)
        adv = conv(u, cur, base, rate, rnd)
        before, after = rewrite_payment(world, p, bank, [(y["key"], credit_y)], adv)
        row_y = [r for r in after if r["role"] == "ar" and r["invoice"] == y["key"]][0]
        used.update({pk, x, y["key"]})
        cert = {"kind": "alloc_wrong_invoice", "pay": pk, "inv": x, "y": y["key"], "a": a,
                "row_y": row_y["key"], "currency": cur}
        return new_issue("payment_allocation", "wrong_invoice", [pk, x], a, cur, [[pk, x, row_y["key"]]],
                         "accuracy_classification", [("I3", x), ("I3", y["key"]), ("I3p", pk)],
                         [{"op": "update_rows", "target": pk, "before": [snap_row(r) for r in before],
                           "after": [snap_row(r) for r in after]}], cert, extra_docs=[y["key"]])
    return None
