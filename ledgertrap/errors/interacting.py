"""Interacting scenarios (DESIGN §8.1-8.2). Each creates its own documents, posts them correctly,
then applies the mutations, so the clean history stays consistent.

C1 "boundary + currency + partial payment": invoice I in a foreign currency paid in two parts.
   P1 (share s) is correct. P2 arrives just after period end in company time, but its timestamp is
   written in a western offset so the date string shows the last day of the period. The bookkeeper
   posted P2 on the last day at that day's rate: period_cutoff/early + fx_conversion/wrong_date on P2.
C2 "duplicate + payment to the wrong invoice": invoice I1 and its duplicate I2; payment P refers to I1
   but the AR credit is posted to I2: duplicate_invoice/document + payment_allocation/wrong_invoice.
"""

from datetime import datetime, timedelta

from ..money import conv
from ..posting import ar_credit, business_time, post_invoice, post_payment, posting_moment, rewrite_payment
from ..timeutil import is_business, offset_hours, offset_str, parse_offset
from ..world import display_offset, make_lines, month_bounds, new_invoice, new_payment, random_moment
from .common import fx_key, invoice_group, new_issue, payment_group, snap_doc, snap_row

MAX_DOCUMENTS = 200


def c1_feasible(world):
    last = (world.period_end - timedelta(days=1)).astimezone(world.tz).date()
    first = world.period_end.astimezone(world.tz).date()
    return is_business(last) and is_business(first) and bool(world.company["foreign_currencies"])


def apply_c1(rng, world, used, same_prob):
    if not c1_feasible(world) or len(world.docs) + 3 > MAX_DOCUMENTS:
        return None
    foreign = [c for c in world.customers if c["currency"] != world.base]
    if not foreign:
        return None
    tzh = offset_hours(world.tz)
    last = (world.period_end - timedelta(days=1)).astimezone(world.tz).date()
    first = world.period_end.astimezone(world.tz).date()
    for _ in range(30):
        cust = rng.choice(foreign)
        cur = cust["currency"]
        rate_last, rate_first = world.fx.get((last, cur)), world.fx.get((first, cur))
        if rate_last is None or rate_first is None or rate_last == rate_first:
            continue
        half = world.period_start + (world.period_end - world.period_start) / 2
        issued = random_moment(rng, world, world.period_start, half - timedelta(days=2))
        s, e = month_bounds(issued.astimezone(world.tz).date())
        lines = make_lines(rng, cur, s, e)
        total = sum(ln["amount_minor"] for ln in lines)
        share = rng.randint(20, 80)
        a1 = total * share // 100
        a2 = total - a1
        if a1 < 100 or a2 < 100:
            continue
        right = conv(a2, cur, world.base, rate_first, world.rounding)
        wrong = conv(a2, cur, world.base, rate_last, world.rounding)
        if abs(wrong - right) < max(100, -(-abs(right) // 1000)):
            continue
        hour = rng.randint(0, 2)
        p2_local = datetime(first.year, first.month, first.day, hour, rng.randint(5, 59), rng.randint(0, 59),
                            tzinfo=world.tz)
        hi_off = tzh - (hour + 1)
        if hi_off < -12:
            continue
        p2_off = parse_offset(offset_str(rng.randint(max(-12, tzh - 12), hi_off)))
        p1_moment = random_moment(rng, world, issued + timedelta(days=1), half + timedelta(days=5))
        if not (issued < p1_moment < world.period_end):
            continue
        inv = new_invoice(world, cust["key"], cur, issued.astimezone(display_offset(rng, world, same_prob)),
                          lines, 30)
        post_invoice(world, inv, posting_moment(rng, world, inv["issued_at"]))
        p1 = new_payment(world, cust["key"], cur, p1_moment.astimezone(display_offset(rng, world, same_prob)),
                         a1, [(inv["key"], a1)])
        post_payment(world, p1, posting_moment(rng, world, p1["timestamp"]))
        p2 = new_payment(world, cust["key"], cur, p2_local.astimezone(p2_off), a2, [(inv["key"], a2)])
        post_payment(world, p2, posting_moment(rng, world, p2["timestamp"]))
        # Mutation: bookkeeper posts P2 on the last day of the period at that day's rate.
        ars = [(inv["key"], ar_credit(world, inv["key"], p2, a2))]
        adv = 0
        before, after = rewrite_payment(world, p2, wrong, ars, adv)
        ts = business_time(rng, world, last)
        for r in after:
            r["ts"] = ts
        after_snap = [snap_row(r) for r in after]
        before_snap = [snap_row(r) for r in before]
        used.update({inv["key"], p1["key"], p2["key"]})
        bank = [r for r in after if r["role"] == "bank"][0]
        group = payment_group(world, p2["key"])
        muts = [{"op": "add_doc", "target": k, "after": snap_doc(world.docs[k])}
                for k in (inv["key"], p1["key"], p2["key"])]
        muts.append({"op": "update_rows", "target": p2["key"], "before": before_snap, "after": after_snap})
        cut_cert = {"kind": "cutoff_early", "doc": p2["key"], "rows": [r["key"] for r in group], "amount": a2,
                    "currency": cur}
        fx_cert = {"kind": "fx_wrong_date", "doc": p2["key"], "row": bank["key"], "posted": wrong, "right": right,
                   "wrong_rate": rate_last, "wrong_date": last, "rate": rate_first, "rate_date": first,
                   "amount_doc": a2}
        cut = new_issue("period_cutoff", "early", [p2["key"]], a2, cur, [[p2["key"], r["key"]] for r in group],
                        "cutoff", [("I8", p2["key"])], muts, cut_cert)
        fxi = new_issue("fx_conversion", "wrong_date", [p2["key"]], abs(wrong - right), world.base,
                        [[p2["key"], bank["key"], fx_key(world, p2)]], "accuracy_valuation", [("I4p", p2["key"])],
                        [], fx_cert, allowed_global=("I4",))
        for iss in (cut, fxi):
            iss["scenario"] = "C1"
        return [cut, fxi]
    return None


def apply_c2(rng, world, used, same_prob):
    if len(world.docs) + 3 > MAX_DOCUMENTS:
        return None
    for _ in range(30):
        cust = rng.choice(world.customers)
        cur = cust["currency"]
        span = world.period_end - world.period_start
        issued = random_moment(rng, world, world.period_start, world.period_start + span / 3)
        s, e = month_bounds(issued.astimezone(world.tz).date())
        lines = make_lines(rng, cur, s, e)
        total = sum(ln["amount_minor"] for ln in lines)
        if total < 100:
            continue
        i2_moment = issued + timedelta(days=rng.randint(0, 10), hours=rng.randint(0, 8), minutes=rng.randint(1, 59))
        pay_moment = random_moment(rng, world, i2_moment + timedelta(days=1), world.period_end)
        if not (i2_moment < pay_moment < world.period_end):
            continue
        i1 = new_invoice(world, cust["key"], cur, issued.astimezone(display_offset(rng, world, same_prob)), lines, 30)
        post_invoice(world, i1, posting_moment(rng, world, i1["issued_at"]))
        i2 = new_invoice(world, cust["key"], cur, i2_moment.astimezone(display_offset(rng, world, same_prob)),
                         [dict(x) for x in lines], 30)
        post_invoice(world, i2, posting_moment(rng, world, i2["issued_at"]))
        p = new_payment(world, cust["key"], cur, pay_moment.astimezone(display_offset(rng, world, same_prob)),
                        total, [(i1["key"], total)])
        post_payment(world, p, posting_moment(rng, world, p["timestamp"]))
        rate = world.rate_for(p)
        bank = conv(total, cur, world.base, rate, world.rounding)
        credit_i2 = ar_credit(world, i2["key"], p, total)  # i2 has no refs -> paid_before = 0
        before, after = rewrite_payment(world, p, bank, [(i2["key"], credit_i2)], 0)
        row_i2 = [r for r in after if r["role"] == "ar"][0]
        used.update({i1["key"], i2["key"], p["key"]})
        g1, g2 = invoice_group(world, i1["key"]), invoice_group(world, i2["key"])
        muts = [{"op": "add_doc", "target": k, "after": snap_doc(world.docs[k])}
                for k in (i1["key"], i2["key"], p["key"])]
        muts.append({"op": "update_rows", "target": p["key"], "before": [snap_row(r) for r in before],
                     "after": [snap_row(r) for r in after]})
        dup_cert = {"kind": "duplicate_document", "d1": i1["key"], "d2": i2["key"], "rows_d1": [r["key"] for r in g1],
                    "rows_d2": [r["key"] for r in g2], "amount": total, "currency": cur}
        alloc_cert = {"kind": "alloc_wrong_invoice", "pay": p["key"], "inv": i1["key"], "y": i2["key"], "a": total,
                      "row_y": row_i2["key"], "currency": cur}
        dup = new_issue("duplicate_invoice", "document", [i1["key"], i2["key"]], total, cur,
                        [[i1["key"], i2["key"], a["key"], b["key"]] for a in g1 for b in g2], "occurrence",
                        [("I9a", tuple(sorted((i1["key"], i2["key"]))))], muts, dup_cert, extra_docs=[p["key"]])
        alloc = new_issue("payment_allocation", "wrong_invoice", [p["key"], i1["key"]], total, cur,
                          [[p["key"], i1["key"], row_i2["key"]]], "accuracy_classification",
                          [("I3", i1["key"]), ("I3", i2["key"]), ("I3p", p["key"])], [], alloc_cert,
                          extra_docs=[i2["key"]])
        for iss in (dup, alloc):
            iss["scenario"] = "C2"
        return [dup, alloc]
    return None
