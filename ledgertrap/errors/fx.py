"""fx_conversion: `wrong_date` and `wrong_rate` on foreign-currency documents (DESIGN §7.4)."""

from datetime import timedelta
from decimal import Decimal

from ..money import conv
from ..posting import ar_credit, rewrite_payment
from ..timeutil import is_business, last_business_on_or_before
from ..world import _round_sig
from .common import fx_key, invoice_group, new_issue, snap_row


def eligible(world, used):
    out = []
    for d in world.docs.values():
        if d["key"] in used or d["currency"] == world.base:
            continue
        if d["kind"] == "invoice":
            if d["opening"] or world.payments_for(d["key"]):
                continue
            if world.interval(d["issued_at"]) != "period":
                continue
        elif d["kind"] == "payment":
            if d["amount"] <= 0 or world.interval(d["timestamp"]) != "period":
                continue
        else:
            continue
        out.append(d["key"])
    return sorted(out)


def _wrong_date_rates(rng, world, doc, posting_ts):
    rd = world.rate_date(doc)
    cur = doc["currency"]
    correct = world.fx[(rd, cur)]
    dates = []
    dates.append(last_business_on_or_before(posting_ts.astimezone(world.tz).date()))
    dates.append(last_business_on_or_before((world.period_end - timedelta(days=1)).astimezone(world.tz).date()))
    d = rd
    for _ in range(3):
        d -= timedelta(days=1)
        while not is_business(d):
            d -= timedelta(days=1)
        dates.append(d)
    d = rd
    for _ in range(3):
        d += timedelta(days=1)
        while not is_business(d):
            d += timedelta(days=1)
        dates.append(d)
    out = []
    for x in dates:
        r = world.fx.get((x, cur))
        if r is not None and r != correct and x != rd and (x, r) not in out:
            out.append((x, r))
    rng.shuffle(out)
    return out


def _wrong_rates(rng, world, doc):
    cur = doc["currency"]
    correct = world.fx[(world.rate_date(doc), cur)]
    existing = {r for (d, c), r in world.fx.items() if c == cur}
    out = []
    for _ in range(6):
        k = Decimal(rng.randint(5, 50)) / Decimal(1000)
        r = _round_sig(correct * (Decimal(1) + k if rng.below(2) else Decimal(1) - k))
        if r not in existing and r != correct:
            out.append((None, r))
    digits = list(format(correct, "f"))
    pos = [i for i in range(len(digits) - 1) if digits[i].isdigit() and digits[i + 1].isdigit()
           and digits[i] != digits[i + 1]]
    if pos:
        i = rng.choice(pos)
        digits[i], digits[i + 1] = digits[i + 1], digits[i]
        r = Decimal("".join(digits))
        if r > 0 and r not in existing:
            out.append((None, r))
    rng.shuffle(out)
    return out


def apply(rng, world, used, variant):
    cands = eligible(world, used)
    rng.shuffle(cands)
    base, rnd = world.base, world.rounding
    for key in cands[:40]:
        doc = world.docs[key]
        correct = world.rate_for(doc)
        if doc["kind"] == "invoice":
            group = invoice_group(world, key)
            ts = group[0]["ts"]
            amount_doc = doc["total"]
        else:
            group = sorted((r for r in world.rows.values() if r["payment"] == key), key=lambda r: r["key"])
            ts = group[0]["ts"]
            amount_doc = doc["amount"]
        right = conv(amount_doc, doc["currency"], base, correct, rnd)
        options = _wrong_date_rates(rng, world, doc, ts) if variant == "wrong_date" else _wrong_rates(rng, world, doc)
        for used_date, wrong in options:
            posted = conv(amount_doc, doc["currency"], base, wrong, rnd)
            diff = abs(posted - right)
            if diff < max(100, -(-abs(right) // 1000)):
                continue
            fxk = fx_key(world, doc)
            if doc["kind"] == "invoice":
                before = [snap_row(r) for r in group]
                for r in group:
                    if r["role"] == "ar":
                        r["debit"] = posted
                    elif r["role"] == "revenue":
                        r["credit"] = posted
                after = [snap_row(r) for r in group]
                ar = [r for r in group if r["role"] == "ar"][0]
                rev = [r for r in group if r["role"] == "revenue"][0]
                ev = [[key, ar["key"], fxk], [key, rev["key"], fxk]]
                required = [("I3", key), ("I5", key)]
                row_key, allowed_global = ar["key"], ()
            else:
                ars = [(k, ar_credit(world, k, doc, a)) for k, a in (doc["refs"] or [])]
                u = doc["amount"] - sum(a for _, a in (doc["refs"] or []))
                adv = conv(u, doc["currency"], base, correct, rnd)
                b, a_ = rewrite_payment(world, doc, posted, ars, adv)
                before, after = [snap_row(r) for r in b], [snap_row(r) for r in a_]
                bank = [r for r in a_ if r["role"] == "bank"][0]
                ev = [[key, bank["key"], fxk]]
                required = [("I4p", key)]
                row_key, allowed_global = bank["key"], ("I4",)
            used.add(key)
            cert = {"kind": "fx_" + variant, "doc": key, "row": row_key, "posted": posted, "right": right,
                    "wrong_rate": wrong, "wrong_date": used_date, "rate": correct,
                    "rate_date": world.rate_date(doc), "amount_doc": amount_doc}
            return new_issue("fx_conversion", variant, [key], diff, base, ev, "accuracy_valuation", required,
                             [{"op": "update_rows", "target": key, "before": before, "after": after}], cert,
                             allowed_global=allowed_global)
    return None
