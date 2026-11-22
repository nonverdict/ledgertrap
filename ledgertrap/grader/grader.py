"""Grader facade: score(findings.json, inspection_steps) -> GradeResult (+ journal metrics)."""

from collections import namedtuple

from .matcher import match, prepare_issues
from .normalize import normalize_finding, tolerances_from_policy
from .reward import f1_soft, metrics, reward_main
from .validate import validate_submission

GradeResult = namedtuple("GradeResult", "valid reason reward reward_experimental metrics details")

ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")


def accessible_ids(records):
    """Every id an agent can fetch with the tools: ledger row_id; invoice and payment doc_id plus their
    invoice_id / payment_id (get_record accepts both); fx key 'YYYY-MM-DD:SRC:TGT'."""
    ids = set()
    for r in records["ledger"]:
        ids.add(r["row_id"])
    for d in records["invoices"]:
        ids.update((d["doc_id"], d["invoice_id"]))
    for d in records["payments"]:
        ids.update((d["doc_id"], d["payment_id"]))
    for f in records["fx"]:
        ids.add("%s:%s:%s" % (f["date"], f["source_currency"], f["target_currency"]))
    ids.discard(None)
    return frozenset(ids)


def _truth_ids(issues):
    out = set()
    for g in issues:
        out.update(g["affected_ids"])
        for s in g["evidence_sets"]:
            out.update(s)
    return frozenset(out)


class Grader:
    def __init__(self, truth, known_ids=None, tolerances=0):
        """known_ids: ids reachable through the tools (None -> only ids named in truth);
        tolerances: published τ table (int or {type: int}), see normalize.tolerances_from_policy."""
        self._episode_id = truth.get("episode_id")
        self._issues = prepare_issues(truth["issues"])
        self._known = frozenset(known_ids) if known_ids is not None else _truth_ids(truth["issues"])
        self._tol = tolerances

    @classmethod
    def for_episode(cls, ep):
        return cls(ep.truth, accessible_ids(ep.records), tolerances_from_policy(ep.policy))

    def score(self, submission, inspection_steps):
        """submission: findings.json object (or anything, if the agent sent garbage)."""
        reason, findings = validate_submission(submission)
        if reason is not None:
            return self.invalid(reason, inspection_steps)
        res = self._score_findings(findings, inspection_steps)
        res.metrics["episode_id_match"] = submission["episode_id"] == self._episode_id
        return res

    def invalid(self, reason, inspection_steps):
        """Malformed structure: reward -1; the reason goes to the grader journal. Internally the counts are
        0/0/n so aggregates stay numeric; judge_adapter.score_episode reports them as null."""
        m = {"tp": 0, "fp": 0, "fn": len(self._issues), "precision": 0.0, "recall": 0.0, "f1": 0.0,
             "inspection_steps": inspection_steps, "reward_main": -1.0, "reward_experimental": -1.0,
             "by_type": {}, "invalid_reason": reason, "n_issues": len(self._issues)}
        return GradeResult(False, reason, -1.0, -1.0, m, {"pairs": [], "near": []})

    def score_empty(self, inspection_steps):
        return self._score_findings([], inspection_steps)

    def _score_findings(self, findings, inspection_steps):
        norm = [normalize_finding(i, f) for i, f in enumerate(findings)]
        res = match(norm, self._issues, self._known, self._tol)
        p, r, f1 = metrics(res.tp, res.fp, res.fn)
        main = reward_main(f1, inspection_steps)
        soft = reward_main(f1_soft(res.tp, res.fp, res.fn, len(res.near)), inspection_steps)
        by_type = {}
        matched = {iid for _, iid in res.pairs}
        for g in self._issues:
            bt = by_type.setdefault(g["type"], {"tp": 0, "fp": 0, "fn": 0})
            bt["tp" if g["issue_id"] in matched else "fn"] += 1
        matched_idx = {i for i, _ in res.pairs}
        for f in norm:
            if f.index not in matched_idx:
                bt = by_type.setdefault(f.type if f.type in ISSUE_TYPES else "other", {"tp": 0, "fp": 0, "fn": 0})
                bt["fp"] += 1
        m = {"tp": res.tp, "fp": res.fp, "fn": res.fn, "precision": p, "recall": r, "f1": f1,
             "inspection_steps": inspection_steps, "reward_main": main, "reward_experimental": soft,
             "by_type": by_type, "invalid_reason": None, "n_issues": len(self._issues)}
        return GradeResult(True, None, main, soft, m, {"pairs": res.pairs, "near": res.near})
