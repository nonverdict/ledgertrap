"""Shared helpers for mutations: snapshots, eligibility, posting groups."""

from datetime import datetime


def snap_row(r):
    out = dict(r)
    out["ts"] = r["ts"].isoformat()
    return out


def snap_doc(d):
    out = {}
    for k, v in d.items():
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    if "lines" in d:
        out["lines"] = [dict(x) for x in d["lines"]]
    if "refs" in d and d["refs"] is not None:
        out["refs"] = [list(x) for x in d["refs"]]
    return out


def invoice_group(world, key):
    """Own posting rows of an invoice/credit note (not payment rows)."""
    return sorted((r for r in world.rows.values()
                   if r["invoice"] == key and r["payment"] is None and r["role"] != "opening_equity"),
                  key=lambda r: r["key"])


def payment_group(world, key):
    return sorted((r for r in world.rows.values() if r["payment"] == key), key=lambda r: r["key"])


def doc_group(world, key):
    return payment_group(world, key) if world.docs[key]["kind"] == "payment" else invoice_group(world, key)


def has_payments(world, inv_key):
    return bool(world.payments_for(inv_key))


def fx_key(world, doc):
    """Internal evidence key of the FX rate the policy prescribes for `doc`."""
    return "fx:%s:%s" % (world.rate_date(doc).isoformat(), doc["currency"])


def new_issue(itype, variant, affected, amount, currency, evidence_sets, isa, required, mutations, cert,
              extra_docs=(), allowed_global=()):
    """`required`: invariant violations this issue must cause (declared discrepancy).
    Any violation caused must concern only `affected` + `extra_docs` (or be a global code
    listed in `allowed_global`); the generator checks both after all mutations."""
    return {
        "type": itype, "variant": variant, "scenario": None, "isa_assertion": isa,
        "affected": sorted(affected), "amount": abs(amount), "currency": currency,
        "evidence_sets": sorted(sorted(s) for s in evidence_sets),
        "required": sorted(required, key=lambda x: (x[0], str(x[1]))),
        "allowed_docs": sorted(set(affected) | set(extra_docs)),
        "allowed_global": sorted(allowed_global),
        "mutations": mutations, "cert": cert,
    }
