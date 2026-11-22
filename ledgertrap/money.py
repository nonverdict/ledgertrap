"""Integer money and Decimal FX conversion. No floats anywhere."""

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

CURRENCY_EXP = {
    "KZT": 2, "USD": 2, "EUR": 2, "CNY": 2, "RUB": 2, "GBP": 2, "JPY": 0, "KWD": 3,
}

ROUNDING_MODES = {"HALF_UP": ROUND_HALF_UP, "HALF_EVEN": ROUND_HALF_EVEN}


def conv(amount_minor: int, src: str, base: str, rate, rounding: str) -> int:
    """Convert signed minor units of `src` into minor units of `base`.

    conv(x) = sign(x) * ROUND(|x| * rate * 10^(exp(base) - exp(src)))
    `rate` is "1 major src = rate major base" (Decimal or decimal string).
    ROUND is applied once to the whole amount. HALF_UP rounds halves away from zero.
    """
    if not isinstance(amount_minor, int) or isinstance(amount_minor, bool):
        raise TypeError("amount_minor must be int")
    if src == base:
        return amount_minor
    if rate is None:
        raise ValueError("rate required for %s->%s" % (src, base))
    mode = ROUNDING_MODES[rounding]
    sign = -1 if amount_minor < 0 else 1
    with localcontext() as ctx:
        ctx.prec = 50
        q = Decimal(abs(amount_minor)) * Decimal(rate) * (Decimal(10) ** (CURRENCY_EXP[base] - CURRENCY_EXP[src]))
        r = q.quantize(Decimal(1), rounding=mode)
    return sign * int(r)


def unconv(base_minor: int, src: str, base: str, rate) -> int:
    """Best inverse of conv: base minor units back to `src` minor units (HALF_EVEN)."""
    if src == base:
        return base_minor
    with localcontext() as ctx:
        ctx.prec = 50
        q = Decimal(base_minor) / Decimal(rate) * (Decimal(10) ** (CURRENCY_EXP[src] - CURRENCY_EXP[base]))
        return int(q.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def format_rate(rate) -> str:
    return format(Decimal(rate), "f")


def format_minor(amount_minor: int, currency: str) -> str:
    """Human format for certificates: 1 234 567,89 KZT."""
    exp = CURRENCY_EXP[currency]
    sign = "-" if amount_minor < 0 else ""
    a = abs(amount_minor)
    major, minor = divmod(a, 10**exp) if exp else (a, 0)
    major_s = "{:,}".format(major).replace(",", " ")
    if exp:
        return "%s%s,%0*d %s" % (sign, major_s, exp, minor, currency)
    return "%s%s %s" % (sign, major_s, currency)

