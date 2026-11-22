"""One-to-one matching of findings to truth issues (case §05; official matcher conventions, 20.11.2026).

Edge f -> g exists iff ALL hold:
  same type; affected-id set equal to g's canonical set; same currency; |amount - expected| <= τ(type);
  every evidence id of f exists and is reachable through the tools; f's evidence contains at least one
  complete evidence set of g.
TP/FP/FN come from a MAXIMUM-cardinality matching of this bipartite graph (repeated findings are separate
vertices). Among maximum matchings the choice is deterministic: issues are taken in lexicographic order of
issue_id (Unicode code points) and each takes the lowest finding index that still allows a maximum
matching of the rest; this yields the lexicographically smallest sorted sequence of (issue_id, finding
index) pairs. Submission order can only change WHICH identical finding is matched, never TP/FP/FN.
"""

from collections import namedtuple

from .normalize import tolerance_for

MatchResult = namedtuple("MatchResult", "tp fp fn pairs near")


def is_tp(f, issue, known_ids, tolerances):
    return (f.type == issue["type"] and f.affected == issue["_affected"] and f.currency == issue["currency"]
            and abs(f.amount - issue["expected_amount_minor"]) <= tolerance_for(tolerances, issue["type"])
            and f.evidence <= known_ids
            and any(s <= f.evidence for s in issue["_evidence"]))


def prepare_issues(issues):
    out = []
    for g in issues:
        h = dict(g)
        h["_affected"] = frozenset(g["affected_ids"])
        h["_evidence"] = [frozenset(s) for s in g["evidence_sets"]]
        out.append(h)
    return out


def _max_size(adj, issues, blocked):
    """Kuhn's augmenting paths on the sub-graph `issues` x (findings not in `blocked`)."""
    owner = {}

    def augment(g, seen):
        for f in adj[g]:
            if f in blocked or f in seen:
                continue
            seen.add(f)
            if f not in owner or augment(owner[f], seen):
                owner[f] = g
                return True
        return False

    return sum(1 for g in issues if augment(g, set()))


def match(findings, issues, known_ids, tolerances=0):
    """findings: NormFinding list (index = position in the submitted list); issues: prepare_issues()."""
    by_id = {g["issue_id"]: g for g in issues}
    order = sorted(by_id)
    adj = {iid: [f.index for f in findings if is_tp(f, by_id[iid], known_ids, tolerances)] for iid in order}
    for iid in order:
        adj[iid].sort()
    need = _max_size(adj, order, frozenset())
    used = set()
    pairs = []
    for pos, iid in enumerate(order):
        if need == 0:
            break
        rest = order[pos + 1:]
        for fi in adj[iid]:
            if fi in used:
                continue
            if 1 + _max_size(adj, rest, frozenset(used | {fi})) == need:
                used.add(fi)
                pairs.append((fi, iid))
                need -= 1
                break
    # "near" (experimental reward only): unmatched finding with the type and id set of an unmatched issue.
    matched_issues = {iid for _, iid in pairs}
    near, near_taken = [], set()
    for f in findings:
        if f.index in used:
            continue
        for iid in order:
            g = by_id[iid]
            if iid in matched_issues or iid in near_taken:
                continue
            if g["type"] == f.type and g["_affected"] == f.affected:
                near_taken.add(iid)
                near.append((f.index, iid))
                break
    tp = len(pairs)
    return MatchResult(tp=tp, fp=len(findings) - tp, fn=len(issues) - tp, pairs=sorted(pairs), near=near)
