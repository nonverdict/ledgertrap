"""Public IDs (ids stream, sampled without repeats) and record order (order stream).

IDs are assigned after mutations, so injected records look like any other record.
"""

from .world import ALNUM

DIGITS = "0123456789"

INVOICE_FORMATS = ("INV-{yy}-{d6}", "SF/{yy}/{d6}", "{a4}-{d6}")
PAYMENT_FORMATS = ("PAY-{d8}", "BK{a8}")


def _render(rng, fmt, yy):
    out = fmt.replace("{yy}", "%02d" % yy)
    for tag, alphabet, n in (("{d6}", DIGITS, 6), ("{d8}", DIGITS, 8), ("{a4}", ALNUM, 4), ("{a8}", ALNUM, 8)):
        while tag in out:
            out = out.replace(tag, rng.token(alphabet, n), 1)
    return out


class IdAssigner:
    def __init__(self, rng, year):
        self.rng = rng
        self.seen = set()
        self.yy = year % 100
        fr = rng.fork("formats")
        self.inv_fmt = fr.choice(INVOICE_FORMATS)
        self.pay_fmt = fr.choice(PAYMENT_FORMATS)

    def _unique(self, make):
        for _ in range(1000):
            x = make()
            if x not in self.seen:
                self.seen.add(x)
                return x
        raise RuntimeError("id space exhausted")

    def token(self, prefix, n):
        return self._unique(lambda: prefix + self.rng.token(ALNUM, n))

    def invoice_number(self):
        return self._unique(lambda: _render(self.rng, self.inv_fmt, self.yy))

    def payment_number(self):
        return self._unique(lambda: _render(self.rng, self.pay_fmt, self.yy))


def assign_ids(rng, world):
    """Return mapping internal key -> public ids.

    docs:  key -> {"doc_id", "number"} (number = invoice_id or payment_id)
    rows:  key -> row_id ; customers: key -> entity_id ; episode_id
    """
    ids = IdAssigner(rng, world.period_start.year)
    episode_id = ids.token("EP-", 8)
    entities = {c["key"]: ids.token("C-", 6) for c in sorted(world.customers, key=lambda c: c["key"])}
    docs = {}
    for k in sorted(world.docs):
        d = world.docs[k]
        num = ids.payment_number() if d["kind"] == "payment" else ids.invoice_number()
        docs[k] = {"doc_id": ids.token("D-", 8), "number": num}
    rows = {k: ids.token("JE-", 8) for k in sorted(world.rows)}
    return {"episode_id": episode_id, "entities": entities, "docs": docs, "rows": rows}


def order_keys(rng, keys, moment_of, mode):
    """Order internal keys by moment with random tie-break, or shuffle fully."""
    keys = sorted(keys)
    if mode == "shuffle":
        rng.shuffle(keys)
        return keys
    tie = {k: rng.u64() for k in keys}
    return sorted(keys, key=lambda k: (moment_of(k), tie[k]))
