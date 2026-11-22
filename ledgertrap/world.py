"""Hidden company story: company, FX series, customers, documents (internal keys).

Everything here works with internal keys (inv#0001, pay#0001, cust#01).
Public IDs are assigned later by `presentation`, after mutations.
"""

from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from .config import pick
from .money import CURRENCY_EXP
from .timeutil import (daterange, is_business, last_business_on_or_before, local_midnight,
                       offset_hours, offset_str, parse_offset)

# Approximate value of 1 major unit in USD; only used to seed plausible FX levels and prices.
USD_VALUE = {
    "USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"), "CNY": Decimal("0.138"),
    "RUB": Decimal("0.0109"), "JPY": Decimal("0.0067"), "KWD": Decimal("3.25"), "KZT": Decimal("0.00194"),
}

CHARTS = {
    "kz_standard": {"bank": "1030", "ar": "1210", "advances": "3510", "revenue": "6010", "returns": "6020",
                    "fx_gain": "6250", "fx_loss": "7430", "expense": ["7210", "7110", "7470"],
                    "opening_equity": "5610"},
    "ifrs_numeric": {"bank": "1100", "ar": "1200", "advances": "2300", "revenue": "4000", "returns": "4100",
                     "fx_gain": "7100", "fx_loss": "7200", "expense": ["6100", "6200", "6300"],
                     "opening_equity": "3000"},
    "alpha_codes": {"bank": "BNK", "ar": "ARC", "advances": "ADV", "revenue": "REV", "returns": "RTN",
                    "fx_gain": "FXG", "fx_loss": "FXL", "expense": ["EXP", "EXP", "EXP"],
                    "opening_equity": "OPN"},
}

CATALOG = [
    ("SVC-CONS", "Консультационные услуги", 20000, 400000),
    ("SVC-SAAS", "Подписка на облачный сервис", 5000, 150000),
    ("SVC-RENT", "Аренда оборудования", 10000, 300000),
    ("SVC-LOGI", "Логистические услуги", 8000, 250000),
    ("SVC-TRAIN", "Обучение персонала", 15000, 200000),
    ("SVC-SUPP", "Техническая поддержка", 5000, 120000),
    ("LIC-SOFT", "Лицензия на программное обеспечение", 30000, 500000),
    ("SVC-MRKT", "Маркетинговые услуги", 10000, 350000),
    ("SVC-DEV", "Разработка программного обеспечения", 50000, 800000),
    ("GDS-PART", "Запасные части", 2000, 90000),
    ("GDS-PACK", "Упаковочные материалы", 1000, 40000),
    ("SVC-AUD", "Анализ бизнес-процессов", 25000, 450000),
]

NAME_A = ["Ак", "Жана", "Нур", "Алтын", "Кок", "Сары", "Бай", "Тау", "Дала", "Орион", "Вега", "Квант",
          "Меридиан", "Сапфир", "Атлас"]
NAME_B = ["Трейд", "Логистик", "Строй", "Агро", "Тех", "Сервис", "Маркет", "Энерго", "Фарм", "Медиа"]
NAME_C = ["ТОО", "АО", "LLC", "Ltd"]

ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _round_sig(x: Decimal, sig: int = 6) -> Decimal:
    return x.quantize(Decimal(1).scaleb(x.adjusted() - sig + 1), rounding=ROUND_HALF_EVEN)


def _ulp(x: Decimal) -> Decimal:
    return Decimal(1).scaleb(x.as_tuple().exponent)


class World:
    """Mutable container of the hidden company state (internal keys)."""

    def __init__(self):
        self.company = {}
        self.period_start = None
        self.period_end = None
        self.export_start = None
        self.export_end = None
        self.tz = None
        self.currencies = {}
        self.customers = []
        self.fx = {}            # (date, currency) -> Decimal rate to base
        self.docs = {}          # key -> doc dict
        self.rows = {}          # key -> row dict
        self.expense_groups = []
        self._counters = {}

    # ---- keys -------------------------------------------------------------------------
    def new_key(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return "%s#%05d" % (prefix, n)

    # ---- lookups ----------------------------------------------------------------------
    @property
    def base(self) -> str:
        return self.company["base_currency"]

    @property
    def rounding(self) -> str:
        return self.company["rounding"]

    def doc_moment(self, d) -> datetime:
        return d["issued_at"] if d["kind"] in ("invoice", "credit_note") else d["timestamp"]

    def rate_date(self, d) -> date:
        return last_business_on_or_before(self.doc_moment(d).astimezone(self.tz).date())

    def rate_for(self, d):
        if d["currency"] == self.base:
            return None
        return self.fx[(self.rate_date(d), d["currency"])]

    def interval(self, moment: datetime) -> str:
        if moment < self.period_start:
            return "pre"
        if moment < self.period_end:
            return "period"
        return "window"

    def invoices(self):
        return [d for d in self.docs.values() if d["kind"] in ("invoice", "credit_note")]

    def payments(self):
        return [d for d in self.docs.values() if d["kind"] == "payment"]

    def payments_for(self, inv_key):
        out = []
        for p in self.payments():
            for k, a in (p["refs"] or []):
                if k == inv_key:
                    out.append((p, a))
        out.sort(key=lambda pa: (pa[0]["timestamp"], pa[0]["key"]))
        return out

    def group_rows(self, doc_key):
        d = self.docs[doc_key]
        if d["kind"] == "payment":
            return [r for r in self.rows.values() if r["payment"] == doc_key]
        return [r for r in self.rows.values()
                if r["invoice"] == doc_key and r["payment"] is None and r["role"] != "opening_equity"]


# ---------------------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------------------

def build_company(rng, cfg, world: World):
    c = cfg["company"]
    base = pick(rng.fork("base"), cfg, ("company", "base_currency"))
    pool = [x for x in c["foreign_currencies"] if x != base]
    n_f = min(pick(rng.fork("nfx"), cfg, ("company", "n_foreign_currencies")), len(pool))
    r = rng.fork("fxset")
    pool = sorted(pool)
    r.shuffle(pool)
    foreign = sorted(pool[:n_f])
    tz_s = pick(rng.fork("tz"), cfg, ("company", "timezone"))
    template = pick(rng.fork("chart"), cfg, ("company", "chart_template"))
    accounts = dict(CHARTS[template])
    accounts["expense"] = list(accounts["expense"])
    if template == "alpha_codes":
        rs = rng.fork("chart-suffix")
        used = set()
        for role in ("bank", "ar", "advances", "revenue", "returns", "fx_gain", "fx_loss", "opening_equity"):
            accounts[role] = _unique_code(rs, accounts[role], used)
        accounts["expense"] = [_unique_code(rs, "EXP", used) for _ in range(3)]
    rn = rng.fork("name")
    name = "%s%s %s (вымышл.)" % (rn.choice(NAME_A), rn.choice(NAME_B).lower(), rn.choice(NAME_C))
    world.company = {
        "name": name,
        "base_currency": base,
        "foreign_currencies": foreign,
        "timezone": tz_s,
        "chart_template": template,
        "accounts": accounts,
        "rounding": pick(rng.fork("rounding"), cfg, ("company", "rounding")),
    }
    world.tz = parse_offset(tz_s)
    world.currencies = {cur: CURRENCY_EXP[cur] for cur in sorted([base] + foreign)}


def _unique_code(rng, prefix, used):
    while True:
        code = "%s-%s" % (prefix, rng.token(ALNUM, 2))
        if code not in used:
            used.add(code)
            return code


def build_period(rng, cfg, world: World):
    kind = pick(rng.fork("kind"), cfg, ("company", "period_kind"))
    year = cfg["company"]["period_year"]
    exact = cfg["errors"]["exact"] or {}
    need_c1 = cfg["errors"]["mode"] == "exact" and exact.get("C1", 0) > 0
    if kind == "month":
        def boundary_ok(m):  # C1 needs business days on both sides of the period end
            nxt = date(year + (m == 12), m % 12 + 1, 1)
            return is_business(nxt - timedelta(days=1)) and is_business(nxt)
        if need_c1:
            months = [m for m in range(1, 13) if boundary_ok(m)]
            m = months[rng.fork("index").below(len(months))]
        else:
            m = rng.fork("index").randint(1, 12)
        start = date(year, m, 1)
        end = date(year + (m == 12), m % 12 + 1, 1)
        label = "%04d-%02d" % (year, m)
    else:
        q = rng.fork("index").randint(1, 4)
        start = date(year, 3 * (q - 1) + 1, 1)
        end = date(year + (q == 4), (3 * q) % 12 + 1, 1)
        label = "%04d-Q%d" % (year, q)
    world.company["period_kind"] = kind
    world.company["period_label"] = label
    world.period_start = local_midnight(start, world.tz)
    world.period_end = local_midnight(end, world.tz)
    world.export_start = world.period_start - timedelta(days=45)
    world.export_end = world.period_end + timedelta(days=cfg["company"]["window_days"])


def build_fx(rng, world: World):
    first = world.export_start.date() - timedelta(days=7)
    last = world.export_end.date()
    days = [d for d in daterange(first, last) if is_business(d)]
    base = world.base
    for cur in world.company["foreign_currencies"]:
        r = rng.fork("fx-" + cur)
        level = USD_VALUE[cur] / USD_VALUE[base]
        level = level * (Decimal(1) + Decimal(r.randint(-100000, 100000)) / Decimal(10**6))
        prev = None
        for d in days:
            if prev is None:
                rate = _round_sig(level)
            else:
                step = Decimal(r.randint(1000, 8000)) / Decimal(10**6)
                if r.below(2):
                    step = -step
                rate = _round_sig(prev * (Decimal(1) + step))
                if rate == prev:
                    rate = prev + _ulp(prev)
            world.fx[(d, cur)] = rate
            prev = rate


def build_customers(rng, cfg, world: World):
    n = pick(rng.fork("n"), cfg, ("size", "customers"))
    foreign = world.company["foreign_currencies"]
    r = rng.fork("cur")
    customers = []
    for i in range(n):
        if i < len(foreign):
            cur = foreign[i]
        elif foreign and r.below(2):
            cur = r.choice(foreign)
        else:
            cur = world.base
        customers.append({"key": "cust#%03d" % (i + 1), "currency": cur})
    world.customers = customers


# ---------------------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------------------

def display_offset(rng, world: World, same_prob) -> object:
    if rng.chance(same_prob):
        return world.tz
    return parse_offset(offset_str(rng.randint(-12, 14)))


def random_moment(rng, world: World, lo: datetime, hi: datetime, hour_lo=8, hour_hi=19) -> datetime:
    """Random local business-hours moment with lo <= t < hi (local dates in company tz)."""
    lo_d = lo.astimezone(world.tz).date()
    hi_d = (hi - timedelta(seconds=1)).astimezone(world.tz).date()
    for _ in range(64):
        d = lo_d + timedelta(days=rng.randint(0, (hi_d - lo_d).days))
        t = datetime(d.year, d.month, d.day, rng.randint(hour_lo, hour_hi), rng.randint(0, 59),
                     rng.randint(0, 59), tzinfo=world.tz)
        if lo <= t < hi:
            return t
    return lo


def unit_price(rng, currency: str, usd_cents_lo: int, usd_cents_hi: int) -> int:
    usd_cents = rng.randint(usd_cents_lo, usd_cents_hi)
    value = Decimal(usd_cents) / Decimal(100) / USD_VALUE[currency]
    minor = value * (Decimal(10) ** CURRENCY_EXP[currency])
    minor = int(minor.quantize(Decimal(1)))
    # Round to a "price-like" value: 2 significant digits for the minor part.
    if minor >= 1000:
        q = 10 ** (len(str(minor)) - 3)
        minor = (minor // q) * q
    return max(minor, 100)


def make_lines(rng, currency: str, service_start: date, service_end: date):
    n = rng.randint(1, 4)
    items = list(range(len(CATALOG)))
    rng.shuffle(items)
    lines = []
    for i, idx in enumerate(items[:n]):
        sku, desc, lo, hi = CATALOG[idx]
        qty = rng.choice([1, 1, 1, 2, 3, 5, 10, 12])
        price = unit_price(rng, currency, lo, hi)
        lines.append({"line_no": i + 1, "sku": sku, "description": desc, "quantity": qty,
                      "unit_price_minor": price, "amount_minor": qty * price,
                      "service_period_start": service_start.isoformat(),
                      "service_period_end": service_end.isoformat()})
    return lines


def month_bounds(d: date):
    start = d.replace(day=1)
    nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start, nxt - timedelta(days=1)


def line_signature(lines):
    return tuple((ln["sku"], ln["description"], ln["quantity"], ln["unit_price_minor"], ln["amount_minor"],
                  ln["service_period_start"], ln["service_period_end"]) for ln in lines)


def new_invoice(world: World, entity, currency, issued_at, lines, terms_days, kind="invoice",
                original=None, opening=False):
    key = world.new_key("inv")
    doc = {"key": key, "kind": kind, "entity": entity, "currency": currency,
           "issued_at": issued_at, "due_at": issued_at + timedelta(days=terms_days),
           "lines": lines, "total": sum(ln["amount_minor"] for ln in lines),
           "original": original, "opening": opening}
    world.docs[key] = doc
    return doc


def new_payment(world: World, entity, currency, timestamp, amount, refs):
    key = world.new_key("pay")
    doc = {"key": key, "kind": "payment", "entity": entity, "currency": currency,
           "timestamp": timestamp, "amount": amount, "refs": refs}
    world.docs[key] = doc
    return doc


def offset_view(dt: datetime, tz) -> datetime:
    return dt.astimezone(tz)


def tz_hours(world: World) -> int:
    return offset_hours(world.tz)
