"""Presentation transforms (id_salt / order_seed): ID mapping between two renderings of the same story,
remapping of findings, an ID-free canonical story form, and pass^k.

Both renderings come from the same (seed, config core, secret), so internal keys coincide and the
mapping is exact. FX ids depend only on dates and currencies and do not change.
"""

from math import comb

from .canon import canonical_bytes, sha256_hex


def id_map(a, b):
    """Map every public id of episode `a` to the corresponding id of episode `b`."""
    m = {a.episode_id: b.episode_id}
    for k, ids in a.id_map["docs"].items():
        other = b.id_map["docs"][k]
        m[ids["doc_id"]] = other["doc_id"]
        m[ids["number"]] = other["number"]
    for k, rid in a.id_map["rows"].items():
        m[rid] = b.id_map["rows"][k]
    for k, eid in a.id_map["entities"].items():
        m[eid] = b.id_map["entities"][k]
    return m


def remap_findings(obj, m):
    out = {"episode_id": m.get(obj["episode_id"], obj["episode_id"]), "findings": []}
    for f in obj["findings"]:
        g = dict(f)
        g["affected_ids"] = [m.get(x, x) for x in f["affected_ids"]]
        g["evidence_ids"] = [m.get(x, x) for x in f["evidence_ids"]]
        out["findings"].append(g)
    return out


def remap_truth(truth, m):
    issues = []
    for g in truth["issues"]:
        h = dict(g)
        h["affected_ids"] = sorted(m.get(x, x) for x in g["affected_ids"])
        h["evidence_sets"] = sorted(sorted(m.get(x, x) for x in s) for s in g["evidence_sets"])
        issues.append(h)
    return issues


def story_fingerprint(ep):
    """Hash of the episode with every id replaced by its internal key and record order removed.
    Equal across id_salt / order_seed / order changes; different for a different story."""
    inv = {}
    for k, ids in ep.id_map["docs"].items():
        inv[ids["doc_id"]] = k
        inv[ids["number"]] = k
    for k, rid in ep.id_map["rows"].items():
        inv[rid] = k
    for k, eid in ep.id_map["entities"].items():
        inv[eid] = k

    def strip(obj):
        if isinstance(obj, dict):
            return {key: strip(v) for key, v in obj.items()}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        if isinstance(obj, str):
            return inv.get(obj, obj)
        return obj
    recs = {t: sorted((strip(r) for r in rows), key=lambda r: canonical_bytes(r)) for t, rows in ep.records.items()}
    manifest = dict(ep.manifest)
    manifest.pop("episode_id")
    issues = []
    for g in ep.truth["issues"]:
        h = strip({k: v for k, v in g.items() if k not in ("issue_id", "certificate_id")})
        h["affected_ids"] = sorted(h["affected_ids"])
        h["evidence_sets"] = sorted(sorted(s) for s in h["evidence_sets"])
        issues.append(h)
    issues.sort(key=lambda g: canonical_bytes(g))
    return sha256_hex(canonical_bytes({"records": recs, "manifest": manifest, "policy": ep.policy,
                                       "issues": issues}))


def pass_hat_k(success_counts, n, k):
    """tau-bench pass^k: mean over tasks of C(c, k) / C(n, k) for c successes in n i.i.d. trials."""
    if not success_counts:
        return 0.0
    return sum(comb(c, k) / comb(n, k) for c in success_counts) / len(success_counts)


def presentation(i):
    """The i-th harmless transformation used by paired checks and pass^k (i=0 is the default view)."""
    if i == 0:
        return {"id_salt": 0, "order_seed": 0, "order": "time"}
    return {"id_salt": 1000 + i, "order_seed": 2000 + i, "order": "shuffle" if i % 2 else "time"}
