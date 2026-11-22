"""Leak scanner (DESIGN §12).

1. Token scan of agent-visible data (observations, info, error messages, trajectories):
   - no truth JSON keys, no issue/certificate ids of that episode, no secret, no filesystem paths,
     no `seed`/`secret` keys; in eval mode rewards are 0 and info is {}.
2. Statistical scan over many episodes: do injected records differ in form from clean ones?
   - posting hour-of-day of mutated rows vs all other document rows (two-sample KS);
   - amount magnitude of affected documents vs documents of the same kind (KS);
   - position of affected documents in their table listing vs all documents (KS).
   Bonferroni over all tests at family level alpha = 0.01.
"""

import json
import math
from datetime import datetime

TRUTH_KEYS = frozenset(["issue_id", "evidence_sets", "expected_amount_minor", "is_clean", "scenario", "variant",
                        "certificate_id", "tolerance_minor", "seed", "secret", "provenance", "truth"])
PATH_MARKERS = ("/home/", "/tmp/", "/app/", "/root/", "C:\\", "truth.json", "provenance.json", "grader/")


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def scan_records(records, truth_by_episode, secrets=(), eval_mode=None):
    """records: iterable of trajectory dicts (or {"observation":..., "info":...}). Returns problems."""
    problems = []
    for rec in records:
        seen = {"observation": rec.get("observation"), "info": rec.get("info", {}), "action": None}
        blob = json.dumps(seen, ensure_ascii=False)
        bad_keys = set(_keys(seen)) & TRUTH_KEYS
        if bad_keys:
            problems.append(("truth_key", rec.get("episode_id"), rec.get("step"), sorted(bad_keys)))
        for m in PATH_MARKERS:
            if m in blob:
                problems.append(("path", rec.get("episode_id"), rec.get("step"), m))
        for s in secrets:
            if s and s in blob:
                problems.append(("secret", rec.get("episode_id"), rec.get("step")))
        truth = truth_by_episode.get(rec.get("episode_id"))
        if truth:
            for g in truth["issues"]:
                if g["issue_id"] in blob or g["certificate_id"] in blob:
                    problems.append(("issue_id", rec.get("episode_id"), rec.get("step"), g["issue_id"]))
        mode = rec.get("mode", eval_mode)
        if mode == "eval" and (rec.get("reward", 0.0) != 0.0 or rec.get("info", {}) not in ({}, None)):
            problems.append(("eval_not_silent", rec.get("episode_id"), rec.get("step")))
    return problems


def scan_trajectory_file(path, truth_by_episode, secrets=()):
    with open(path, encoding="utf-8") as f:
        return scan_records((json.loads(line) for line in f), truth_by_episode, secrets)


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 1.0
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 0.0
    for k in range(1, 101):
        p += 2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
    return d, max(0.0, min(1.0, p))


def _hour(ts):
    return datetime.fromisoformat(ts).hour + datetime.fromisoformat(ts).minute / 60.0


def collect_stats(episodes):
    """Gather samples from generated episodes (with provenance, trusted side)."""
    s = {"hour_mut": [], "hour_clean": [], "amt_aff": [], "amt_all": [], "pos_aff": [], "pos_all": []}
    for ep in episodes:
        mutated = set()
        for m in ep.provenance["mutations"]:
            for r in m.get("after", []) if isinstance(m.get("after"), list) else []:
                if r.get("row_id"):
                    mutated.add(r["row_id"])
        ps = ep.manifest["period_start"]
        for r in ep.records["ledger"]:
            if r["invoice_id"] is None and r["payment_id"] is None:
                continue
            if r["timestamp"] == ps:
                continue  # opening balances are dated period_start 00:00 by rule
            (s["hour_mut"] if r["row_id"] in mutated else s["hour_clean"]).append(_hour(r["timestamp"]))
        affected = {x for g in ep.truth["issues"] for x in g["affected_ids"]}
        base = ep.manifest["base_currency"]
        exps = ep.manifest["currencies"]
        level = {}
        for f in ep.records["fx"]:
            level.setdefault(f["source_currency"], float(f["rate"]))
        for table, amount_key in (("invoices", "total_minor"), ("payments", "amount_minor")):
            rows = ep.records[table]
            for i, d in enumerate(rows):
                pos = i / max(1, len(rows) - 1)
                # magnitude in base-currency major units, so currency scale (JPY vs KWD) does not dominate
                rate = 1.0 if d["currency"] == base else level.get(d["currency"], 1.0)
                value = abs(d[amount_key]) / 10 ** exps[d["currency"]] * rate
                mag = math.log10(value + 1)
                s["pos_all"].append(pos)
                s["amt_all"].append(mag)
                if d["doc_id"] in affected:
                    s["pos_aff"].append(pos)
                    s["amt_aff"].append(mag)
    return s


def stat_report(episodes, alpha=0.01):
    s = collect_stats(episodes)
    tests = {
        "posting_hour (mutated rows vs other document rows)": ks_2samp(s["hour_mut"], s["hour_clean"]),
        "amount_magnitude in base currency (affected docs vs all docs)": ks_2samp(s["amt_aff"], s["amt_all"]),
    }
    level = alpha / len(tests)
    out = {name: {"D": round(d, 4), "p": round(p, 6), "ok": p > level} for name, (d, p) in tests.items()}
    # Position is reported but not gated: cutoff/early targets are by definition after-period documents,
    # which a time-ordered listing places last (the date is visible anyway, so this is not a leak).
    d, p = ks_2samp(s["pos_aff"], s["pos_all"])
    out["listing_position (diagnostic only)"] = {"D": round(d, 4), "p": round(p, 6), "ok": True}
    out["_samples"] = {k: len(v) for k, v in s.items()}
    return out
