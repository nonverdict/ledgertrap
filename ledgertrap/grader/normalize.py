"""Published matcher conventions (official FinTech check definitions of 20.11.2026; DESIGN §11.2).

- type, currency and every id are compared as exact strings: case and spaces matter, no aliases, no prefix
  stripping, no "similar id" search. A number is never turned into a string (a non-string id is a
  structure error, see validate.py).
- affected_ids and evidence_ids are sets *after* the structural limits were checked: order and repeats
  inside one list do not matter. A repeated finding is NOT merged with the first one (it stays a FP).
- claimed_amount_minor is a JSON integer compared with the expected amount within the published τ:
  |claimed - expected| <= τ. No absolute value, no currency conversion.
- τ is a fixed non-negative integer per issue type, published in policy.json under
  submission.amount_tolerance_minor (an object type -> int, or one int for all types). A type without an
  entry has τ = 0. Our generator publishes τ = 0 for every type: each expected amount is recomputable
  exactly from public records with the rules in policy.json.
"""

from collections import namedtuple

NormFinding = namedtuple("NormFinding", "index type affected amount currency evidence")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def normalize_finding(index, f):
    """f has passed validate_submission(); only set semantics are applied."""
    return NormFinding(index=index, type=f["type"], affected=frozenset(f["affected_ids"]),
                       amount=f["claimed_amount_minor"], currency=f["currency"],
                       evidence=frozenset(f["evidence_ids"]))


def tolerances_from_policy(policy):
    """Return the published τ table: an int (all types) or {type: int}. Missing -> 0."""
    sub = (policy or {}).get("submission") or {}
    tol = sub.get("amount_tolerance_minor", 0)
    if _is_int(tol) and tol >= 0:
        return tol
    if isinstance(tol, dict) and all(isinstance(k, str) and _is_int(v) and v >= 0 for k, v in tol.items()):
        return dict(tol)
    raise ValueError("policy.submission.amount_tolerance_minor must be a non-negative int or {type: int}")


def tolerance_for(tolerances, issue_type):
    if _is_int(tolerances):
        return tolerances
    return (tolerances or {}).get(issue_type, 0)


def within_tolerance(claimed, expected, tau=0):
    return abs(claimed - expected) <= tau
