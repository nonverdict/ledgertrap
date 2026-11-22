"""Clean company history: invoices, payments, credit notes, refunds, expenses, traps."""

from datetime import datetime, timedelta
from decimal import Decimal

from .config import pick
from .money import CURRENCY_EXP
from .posting import business_time, post_all_documents, post_expense
from .timeutil import is_business, offset_hours, offset_str, parse_offset
from .world import (USD_VALUE, display_offset, line_signature, make_lines, month_bounds, new_invoice,
                    new_payment, random_moment)

MAX_NEW_DOCS_FROM_ERRORS = 12  # duplicate documents may add up to 12 docs
ROW_MARGIN_LOW = 12            # payment rewrites may remove up to one row per issue
ROW_MARGIN_HIGH = 64           # scenarios and duplicates add up to ~8 rows per issue pair


class _Story:
    def __init__(self, rng, cfg, world):
        self.rng = rng
        self.cfg = cfg
        self.w = world
        self.same_prob = cfg["company"]["doc_offset_same_prob"]
        self.signatures = set()
        self._n = 0

    def r(self, label):
        self._n += 1
        return self.rng.fork("%s-%d" % (label, self._n))

    def count(self):
        return len(self.w.docs)

    # --- primitives ---------------------------------------------------------------------
    def customer(self, r, currency=None):
        pool = [c for c in self.w.customers if currency is None or c["currency"] == currency]
        return r.choice(pool) if pool else None

    def unique_lines(self, r, entity, currency, svc_start, svc_end):
        for _ in range(20):
            lines = make_lines(r, currency, svc_start, svc_end)
            sig = (entity, currency, line_signature(lines))
            if sig not in self.signatures:
                self.signatures.add(sig)
                return lines
        raise RuntimeError("could not draw unique invoice lines")

    def register_lines(self, entity, currency, lines):
        sig = (entity, currency, line_signature(lines))
        if sig in self.signatures:
            return False
        self.signatures.add(sig)
        return True

    def invoice(self, r, moment=None, interval="period", cust=None, lines=None, offset=None):
        w = self.w
        if cust is None:
            cust = self.customer(r)
        if moment is None:
            if interval == "period":
                moment = random_moment(r, w, w.period_start, w.period_end)
            else:
                moment = random_moment(r, w, w.period_end, w.export_end)
        tz = offset if offset is not None else display_offset(r, w, self.same_prob)
        if lines is None:
            s, e = month_bounds(moment.astimezone(w.tz).date())
            lines = self.unique_lines(r, cust["key"], cust["currency"], s, e)
        return new_invoice(w, cust["key"], cust["currency"], moment.astimezone(tz), lines,
                           r.choice([14, 30]))

    def pay_moments(self, r, inv, k, weekend=False):
        w = self.w
        lo = inv["issued_at"] + timedelta(days=1)
        if inv.get("opening"):
            lo = max(lo, w.period_start + timedelta(hours=9))
        hi = min(lo + timedelta(days=35), w.export_end)
        if lo >= hi:
            return None
        if weekend:
            d = lo.astimezone(w.tz).date()
            days = []
            while datetime(d.year, d.month, d.day, tzinfo=w.tz) < hi:
                if not is_business(d):
                    days.append(d)
                d += timedelta(days=1)
            days = [x for x in days if lo <= business_time(r, w, x) < hi]
            if not days:
                return None
            t = business_time(r, w, r.choice(days))
            return [t] if lo <= t < hi else None
        ts = sorted(random_moment(r, w, lo, hi) for _ in range(k))
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                ts[i] = ts[i - 1] + timedelta(seconds=1)
        if ts[-1] >= hi:
            return None
        return ts

    def payment(self, r, inv, moment, amount, refs):
        tz = display_offset(r, self.w, self.same_prob)
        return new_payment(self.w, inv["entity"], inv["currency"], moment.astimezone(tz), amount, refs)

    def apply_plan(self, r, inv, plan, weekend=False):
        T = inv["total"]
        if plan == "unpaid":
            return
        if plan in ("full", "overpay", "partial", "cn_refund"):
            ts = self.pay_moments(r, inv, 1, weekend=weekend)
            if ts is None:
                return
            if plan == "full" or plan == "cn_refund":
                p = self.payment(r, inv, ts[0], T, [(inv["key"], T)])
                if plan == "cn_refund":
                    self.credit_note_and_refund(r, inv, p)
            elif plan == "overpay":
                extra = max(1, T * r.randint(1, 10) // 100)
                self.payment(r, inv, ts[0], T + extra, [(inv["key"], T)])
            else:
                a = max(1, T * r.randint(20, 80) // 100)
                self.payment(r, inv, ts[0], a, [(inv["key"], a)])
            return
        if plan == "split":
            k = r.randint(2, 3)
            ts = self.pay_moments(r, inv, k)
            if ts is None:
                return
            cuts = sorted(r.randint(1, T - 1) for _ in range(k - 1)) if T > k else []
            if len(set(cuts)) != len(cuts) or not cuts:
                self.payment(r, inv, ts[0], T, [(inv["key"], T)])
                return
            parts = [b - a for a, b in zip([0] + cuts, cuts + [T])]
            for t, a in zip(ts, parts):
                self.payment(r, inv, t, a, [(inv["key"], a)])
            return
        raise ValueError(plan)

    def credit_note_and_refund(self, r, inv, pay):
        w = self.w
        cn_t = pay["timestamp"] + timedelta(days=r.randint(1, 5), hours=r.randint(0, 6))
        rf_t = cn_t + timedelta(days=r.randint(1, 5), hours=r.randint(0, 6))
        if rf_t >= w.export_end or w.interval(cn_t) != w.interval(inv["issued_at"]):
            return
        lines = []
        for ln in inv["lines"]:
            neg = dict(ln)
            neg["unit_price_minor"] = -ln["unit_price_minor"]
            neg["amount_minor"] = -ln["amount_minor"]
            lines.append(neg)
        self.register_lines(inv["entity"], inv["currency"], lines)
        cn = new_invoice(w, inv["entity"], inv["currency"], cn_t.astimezone(display_offset(r, w, self.same_prob)),
                         lines, 0, kind="credit_note", original=inv["key"])
        cn["due_at"] = cn["issued_at"]
        new_payment(w, inv["entity"], inv["currency"], rf_t.astimezone(display_offset(r, w, self.same_prob)),
                    cn["total"], [(cn["key"], cn["total"])])

    def random_plan(self, r):
        return r.weighted([("unpaid", 34), ("full", 40), ("split", 9), ("partial", 7), ("overpay", 6),
                           ("cn_refund", 4)])


def build_documents(rng, cfg, world, target_docs):
    st = _Story(rng, cfg, world)
    w = world
    enabled = set(cfg["traps"]["enabled"] or [])
    rate = cfg["traps"]["rate"]
    force = [t for t in ("L11", "L1", "L2", "L3", "L4", "L5", "L6", "L8", "L9", "L12")
             if t in enabled and rng.fork("force-" + t).chance(rate)]
    reserve = 8

    for trap in force:
        if st.count() > target_docs - reserve:
            break
        r = st.r("trap-" + trap)
        if trap == "L11":
            for _ in range(r.randint(1, 3)):
                cust = st.customer(r)
                m = random_moment(r, w, w.export_start, w.period_start - timedelta(days=1))
                s, e = month_bounds(m.astimezone(w.tz).date())
                lines = st.unique_lines(r, cust["key"], cust["currency"], s, e)
                inv = new_invoice(w, cust["key"], cust["currency"],
                                  m.astimezone(display_offset(r, w, st.same_prob)), lines, 30, opening=True)
                st.apply_plan(r, inv, r.choice(["unpaid", "full"]))
        elif trap == "L1":
            cust = st.customer(r)
            step = r.choice([7, 14])
            first = random_moment(r, w, w.period_start, w.period_start + timedelta(days=7))
            lines0 = make_lines(r, cust["currency"], first.date(), first.date())
            for i in range(r.randint(2, 3)):
                m = first + timedelta(days=step * i)
                if m >= w.period_end:
                    break
                d0 = m.astimezone(w.tz).date()
                lines = [dict(ln, service_period_start=d0.isoformat(),
                              service_period_end=(d0 + timedelta(days=step - 1)).isoformat()) for ln in lines0]
                if not st.register_lines(cust["key"], cust["currency"], lines):
                    continue
                inv = st.invoice(r, moment=m, cust=cust, lines=lines)
                st.apply_plan(r, inv, r.choice(["full", "unpaid"]))
        elif trap == "L2":
            by_cur = {}
            for c in w.customers:
                by_cur.setdefault(c["currency"], []).append(c)
            pairs = [v for k, v in sorted(by_cur.items()) if len(v) >= 2]
            if not pairs:
                continue
            group = r.choice(pairs)
            a, b = group[0], group[1]
            m = random_moment(r, w, w.period_start, w.period_end)
            s, e = month_bounds(m.astimezone(w.tz).date())
            lines = st.unique_lines(r, a["key"], a["currency"], s, e)
            st.register_lines(b["key"], b["currency"], lines)
            for cust in (a, b):
                inv = st.invoice(r, moment=random_moment(r, w, w.period_start, w.period_end), cust=cust,
                                 lines=[dict(x) for x in lines])
                st.apply_plan(r, inv, st.random_plan(r))
        elif trap in ("L3", "L4", "L9", "L12"):
            plan = {"L3": "split", "L4": "cn_refund", "L9": "overpay", "L12": "partial"}[trap]
            inv = st.invoice(r, moment=random_moment(r, w, w.period_start,
                                                     w.period_start + (w.period_end - w.period_start) / 2))
            st.apply_plan(r, inv, plan)
        elif trap == "L5":
            foreign = [c for c in w.customers if c["currency"] != w.base]
            if not foreign:
                continue
            cust = r.choice(foreign)
            inv = st.invoice(r, moment=random_moment(r, w, w.period_start,
                                                     w.period_start + (w.period_end - w.period_start) / 2),
                             cust=cust)
            st.apply_plan(r, inv, "full", weekend=True)
        elif trap == "L6":
            _boundary_docs(st, r)
        elif trap == "L8":
            cust = st.customer(r)
            m = random_moment(r, w, w.period_start, w.period_end)
            amount = _amount_like(r, cust["currency"])
            new_payment(w, cust["key"], cust["currency"], m.astimezone(display_offset(r, w, st.same_prob)),
                        amount, None)

    min_window = max(3, target_docs // 12)
    while st.count() < target_docs:
        r = st.r("main")
        n_window = sum(1 for d in w.invoices() if w.interval(d["issued_at"]) == "window")
        if n_window < min_window:
            interval = "window"
        else:
            interval = "period" if r.below(100) < 88 else "window"
        inv = st.invoice(r, interval=interval)
        if target_docs - st.count() >= 6:
            plan = st.random_plan(r)
            if interval == "period" and _unpaid_share(w) < 0.32:
                plan = "unpaid"  # keep >= 30% unpaid period invoices (DESIGN §7.6.4)
            st.apply_plan(r, inv, plan)
    return force


def _unpaid_share(w):
    period = [d for d in w.invoices() if d["kind"] == "invoice" and not d["opening"]
              and w.interval(d["issued_at"]) == "period"]
    if not period:
        return 0.0
    unpaid = sum(1 for d in period if not w.payments_for(d["key"]))
    return unpaid / len(period)


def _amount_like(r, currency):
    usd_cents = r.randint(20000, 2000000)
    v = Decimal(usd_cents) / Decimal(100) / USD_VALUE[currency] * (Decimal(10) ** CURRENCY_EXP[currency])
    return max(100, int(v.quantize(Decimal(1))))


def _boundary_docs(st, r):
    """Trap L6: timestamps whose date string differs from the company-local date."""
    w = st.w
    tzh = offset_hours(w.tz)
    last_day = (w.period_end - timedelta(days=1)).astimezone(w.tz).date()
    h = r.randint(max(21, 24 - (14 - tzh)), 23)
    m = datetime(last_day.year, last_day.month, last_day.day, h, r.randint(0, 59), r.randint(0, 59), tzinfo=w.tz)
    shift = 24 - h
    off = min(14, tzh + shift + r.randint(0, 2))
    if off - tzh >= shift:
        st.invoice(r, moment=m, offset=parse_offset(offset_str(off)))
    first_day = w.period_end.astimezone(w.tz).date()
    h2 = r.randint(0, 2)
    m2 = datetime(first_day.year, first_day.month, first_day.day, h2, r.randint(5, 59), r.randint(0, 59),
                  tzinfo=w.tz)
    hi_off = tzh - (h2 + 1)
    if hi_off >= -12:
        off2 = r.randint(max(-12, hi_off - 3), hi_off)
        st.invoice(r, moment=m2, offset=parse_offset(offset_str(off2)))


def build_expenses(rng, cfg, world, target_rows):
    w = world
    base = w.base
    expense_accounts = w.company["accounts"]["expense"]
    i = 0
    while len(w.rows) < target_rows:
        r = rng.fork("exp-%d" % i)
        i += 1
        ts = random_moment(r, w, w.period_start, w.export_end, 9, 17)
        usd_cents = r.randint(5000, 800000)
        v = Decimal(usd_cents) / Decimal(100) / USD_VALUE[base] * (Decimal(10) ** CURRENCY_EXP[base])
        post_expense(r, w, ts, max(100, int(v.quantize(Decimal(1)))), r.choice(expense_accounts))


def build_story(rng, cfg, world):
    lo, hi = cfg["size"]["documents"] if isinstance(cfg["size"]["documents"], list) else (
        cfg["size"]["documents"], cfg["size"]["documents"])
    target_docs = pick(rng.fork("ndocs"), cfg, ("size", "documents"))
    # Leave room for duplicate documents added by errors; never go below the configured minimum.
    target_docs = min(target_docs, max(lo, 200 - MAX_NEW_DOCS_FROM_ERRORS))
    target_rows = pick(rng.fork("nrows"), cfg, ("size", "ledger_rows"))
    # Mutations add rows (duplicates, scenarios) or drop at most one row each (payment rewrites);
    # keep a margin so the final ledger always stays within the case range 200-2000.
    target_rows = min(max(target_rows, 200 + ROW_MARGIN_LOW), 2000 - ROW_MARGIN_HIGH)
    forced = build_documents(rng.fork("docs"), cfg, world, target_docs)
    post_all_documents(rng.fork("posting"), world)
    build_expenses(rng.fork("expenses"), cfg, world, target_rows)
    return {"target_docs": target_docs, "target_rows": target_rows, "forced_traps": forced}
