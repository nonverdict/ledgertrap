"""Clean double-entry posting rules (DESIGN §5.3). Shared by the generator and mutations."""

from datetime import datetime, timedelta

from .money import conv


def ar_credit(world, inv_key, pay, a):
    """Cumulative AR settlement: conv(paid_before + a) - conv(paid_before).

    paid_before is taken from documents (invoice_refs of payments with an earlier
    moment), never from postings, so mutating one payment never changes another.
    """
    inv = world.docs[inv_key]
    rate = world.rate_for(inv)
    pb = 0
    for p, amt in world.payments_for(inv_key):
        if p["timestamp"] < pay["timestamp"]:
            pb += amt
    return (conv(pb + a, inv["currency"], world.base, rate, world.rounding)
            - conv(pb, inv["currency"], world.base, rate, world.rounding))


def add_row(world, ts, role, side, signed, entity, invoice, payment, group, account=None):
    """Add a row. Positive `signed` goes to `side`, negative to the opposite side. Zero -> no row."""
    if signed == 0:
        return None
    if signed < 0:
        side = "credit" if side == "debit" else "debit"
        signed = -signed
    acc = account if account is not None else world.company["accounts"][role]
    key = world.new_key("row")
    world.rows[key] = {
        "key": key, "ts": ts.astimezone(world.tz), "role": role, "account": acc,
        "debit": signed if side == "debit" else 0, "credit": signed if side == "credit" else 0,
        "entity": entity, "invoice": invoice, "payment": payment, "group": group,
    }
    return world.rows[key]


def interval_bounds(world, moment):
    iv = world.interval(moment)
    if iv == "period":
        return world.period_start, world.period_end
    if iv == "window":
        return world.period_end, world.export_end
    return world.export_start, world.period_start


def business_time(rng, world, d):
    return datetime(d.year, d.month, d.day, rng.randint(9, 17), rng.randint(0, 59), rng.randint(0, 59),
                    tzinfo=world.tz)


def posting_moment(rng, world, doc_moment, max_lag=3):
    """Posting time: local doc date + lag (0..max_lag) at 09:00-17:59:59, never before the
    document and always inside the document's interval (pre/period/window)."""
    lo, hi = interval_bounds(world, doc_moment)
    base_d = doc_moment.astimezone(world.tz).date()
    lags = list(range(0, max_lag + 1))
    rng.shuffle(lags)
    for lag in lags:
        t = business_time(rng, world, base_d + timedelta(days=lag))
        if doc_moment <= t < hi and t >= lo:
            return t
    t = doc_moment + timedelta(seconds=60)
    return t if t < hi else doc_moment


def post_invoice(world, d, ts):
    rate = world.rate_for(d)
    amt = conv(d["total"], d["currency"], world.base, rate, world.rounding)
    k, ent = d["key"], d["entity"]
    if d["kind"] == "credit_note":
        add_row(world, ts, "returns", "debit", -amt, ent, k, None, k)
        add_row(world, ts, "ar", "credit", -amt, ent, k, None, k)
    elif d["opening"]:
        add_row(world, world.period_start, "ar", "debit", amt, ent, k, None, k)
        add_row(world, world.period_start, "opening_equity", "credit", amt, ent, k, None, k)
    else:
        add_row(world, ts, "ar", "debit", amt, ent, k, None, k)
        add_row(world, ts, "revenue", "credit", amt, ent, k, None, k)


def payment_amounts(world, p, ar_override=None):
    """Return (bank, [(inv_key, ar_signed)], adv_signed, fx_diff) for a payment.

    ar_override lets mutations replace the AR allocation list."""
    rate = world.rate_for(p)
    bank = conv(p["amount"], p["currency"], world.base, rate, world.rounding)
    refs = p["refs"] or []
    if ar_override is None:
        ars = [(k, ar_credit(world, k, p, a)) for k, a in refs]
        u = p["amount"] - sum(a for _, a in refs)
    else:
        ars, u = ar_override
    adv = conv(u, p["currency"], world.base, rate, world.rounding)
    diff = bank - sum(v for _, v in ars) - adv
    return bank, ars, adv, diff


def post_payment(world, p, ts):
    bank, ars, adv, diff = payment_amounts(world, p)
    write_payment_rows(world, p, ts, bank, ars, adv, diff)


def write_payment_rows(world, p, ts, bank, ars, adv, diff):
    k, ent = p["key"], p["entity"]
    add_row(world, ts, "bank", "debit", bank, ent, None, k, k)
    for inv_key, v in ars:
        add_row(world, ts, "ar", "credit", v, ent, inv_key, k, k)
    add_row(world, ts, "advances", "credit", adv, ent, None, k, k)
    if diff > 0:
        add_row(world, ts, "fx_gain", "credit", diff, ent, None, k, k)
    elif diff < 0:
        add_row(world, ts, "fx_loss", "debit", -diff, ent, None, k, k)


def rewrite_payment(world, p, bank, ars, adv):
    """Replace a payment's posting group (same timestamp) with the given amounts.
    The FX difference row is recomputed so the group stays balanced. Returns (before, after) rows."""
    old = sorted((r for r in world.rows.values() if r["payment"] == p["key"]), key=lambda r: r["key"])
    ts = old[0]["ts"]
    for r in old:
        del world.rows[r["key"]]
    diff = bank - sum(v for _, v in ars) - adv
    n0 = len(world.rows)
    write_payment_rows(world, p, ts, bank, ars, adv, diff)
    new = sorted((r for r in world.rows.values() if r["payment"] == p["key"]), key=lambda r: r["key"])
    assert len(world.rows) - n0 == len(new)
    return old, new


def post_all_documents(rng, world):
    docs = sorted(world.docs.values(), key=lambda d: (world.doc_moment(d), d["key"]))
    for d in docs:
        r = rng.fork("post-" + d["key"])
        if d["kind"] == "payment":
            post_payment(world, d, posting_moment(r, world, d["timestamp"]))
        elif d.get("opening"):
            post_invoice(world, d, world.period_start)
        else:
            post_invoice(world, d, posting_moment(r, world, d["issued_at"]))


def post_expense(rng, world, ts, amount, account):
    key = world.new_key("exp")
    world.expense_groups.append(key)
    add_row(world, ts, "expense", "debit", amount, None, None, None, key, account=account)
    add_row(world, ts, "bank", "credit", amount, None, None, None, key)
