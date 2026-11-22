"""Independent re-auditor (DESIGN §9.5).

Reads ONLY public records (manifest, policy, ledger, invoices, payments, fx) and:
1. re-finds the issues; the result must equal the truth exactly (no extra, no missing);
2. checks that every truth evidence set ALONE (only those records + manifest + policy) yields the
   issue's type, affected ids and amount within the published tolerance, and that the set is minimal;
3. produces its own certificate text per finding.
Truth is used only to COMPARE results, never to find them.
"""

from .audit_rules import Books, audit, public_findings
from .grader import Grader
from .grader.grader import accessible_ids
from .grader.normalize import tolerance_for, tolerances_from_policy
from .timeutil import parse as parse_ts


def books(manifest, policy, records, ids=None):
    if ids is None:
        return Books(manifest, policy, records["ledger"], records["invoices"], records["payments"], records["fx"])
    ids = set(ids)
    fx = [f for f in records["fx"] if "%s:%s:%s" % (f["date"], f["source_currency"], f["target_currency"]) in ids]
    return Books(manifest, policy,
                 [r for r in records["ledger"] if r["row_id"] in ids],
                 [d for d in records["invoices"] if d["doc_id"] in ids],
                 [p for p in records["payments"] if p["doc_id"] in ids], fx)


def _matches(f, issue, tol):
    return (f["type"] == issue["type"] and f["currency"] == issue["currency"]
            and set(f["affected_ids"]) == set(issue["affected_ids"])
            and abs(f["claimed_amount_minor"] - issue["expected_amount_minor"]) <= tolerance_for(tol, issue["type"]))


def verify_evidence_sets(ep):
    """Return a list of problems (empty = every evidence set is sufficient and minimal)."""
    problems = []
    tol = tolerances_from_policy(ep.policy)
    known = accessible_ids(ep.records)
    for g in ep.truth["issues"]:
        for s in g["evidence_sets"]:
            found = audit(books(ep.manifest, ep.policy, ep.records, s))
            if not any(_matches(f, g, tol) for f in found):
                problems.append((g["issue_id"], "insufficient", s))
                continue
            for x in s:
                smaller = [y for y in s if y != x]
                found = audit(books(ep.manifest, ep.policy, ep.records, smaller))
                if any(_matches(f, g, tol) for f in found):
                    problems.append((g["issue_id"], "not_minimal_without", x))
                    break
            missing = [x for x in s if x not in known]
            if missing:
                problems.append((g["issue_id"], "unknown_ids", missing))
    return problems


def unexplained(b, findings):
    """Re-post every document from public records and policy; report anything the findings do not
    explain. Clean books (and books whose only deviations are the found issues) give []."""
    fx_docs = {f["affected_ids"][0] for f in findings if f["type"] == "fx_conversion"}
    cut_docs = {f["affected_ids"][0] for f in findings if f["type"] == "period_cutoff"}
    dbl_docs = {f["affected_ids"][0] for f in findings if f.get("_variant") == "double_posting"}
    alloc_pays = {x for f in findings if f["type"] == "payment_allocation" for x in f["affected_ids"]}
    out = []
    rows = [r for rs in b.own_rows.values() for r in rs] + [r for rs in b.pay_rows.values() for r in rs]
    for r in rows:
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0) or min(r["debit_minor"], r["credit_minor"]) < 0:
            out.append(("one_side", r["row_id"]))
    for num in b.own_rows:
        if num not in b.inv_by_num:
            out.append(("dangling_invoice_ref", num))
    for num in b.pay_rows:
        if num not in b.pay_by_num:
            out.append(("dangling_payment_ref", num))
    for d in b.invoices:
        own = b.own_rows.get(d["invoice_id"], [])
        roles = sorted(r["role"] for r in own)
        opening = b.is_opening(d)
        want = (["ar", "opening_equity"] if opening else ["ar", "returns"] if d.get("kind") == "credit_note"
                else ["ar", "revenue"])
        if d["doc_id"] in dbl_docs:
            want = sorted(want * 2)
        if roles != want:
            out.append(("posting_shape", d["doc_id"]))
            continue
        if sum(r["debit_minor"] for r in own) != sum(r["credit_minor"] for r in own):
            out.append(("unbalanced", d["doc_id"]))
        try:
            right = b.conv(d["total_minor"], d)
        except KeyError:
            out.append(("no_rate", d["doc_id"]))
            continue
        if d["doc_id"] not in fx_docs and any(
                r["debit_minor"] - r["credit_minor"] != right for r in own if r["role"] == "ar"):
            out.append(("amount", d["doc_id"]))
        if opening:
            if any(r["timestamp"] != b.manifest["period_start"] for r in own):
                out.append(("opening_date", d["doc_id"]))
        elif d["doc_id"] not in cut_docs and any(
                b.interval(parse_ts(r["timestamp"])) != b.interval(b.moment(d)) for r in own):
            out.append(("interval", d["doc_id"]))
    for p in b.payments:
        prow = b.pay_rows.get(p["payment_id"], [])
        if not prow:
            out.append(("not_posted", p["doc_id"]))
            continue
        if sum(r["debit_minor"] for r in prow) != sum(r["credit_minor"] for r in prow):
            out.append(("unbalanced", p["doc_id"]))
        try:
            bank = b.conv(p["amount_minor"], p)
        except KeyError:
            out.append(("no_rate", p["doc_id"]))
            continue
        got_bank = sum(r["debit_minor"] - r["credit_minor"] for r in prow if r["role"] == "bank")
        if got_bank != bank and p["doc_id"] not in fx_docs:
            out.append(("bank", p["doc_id"]))
        if p["doc_id"] not in cut_docs and any(
                b.interval(parse_ts(r["timestamp"])) != b.interval(b.moment(p)) for r in prow):
            out.append(("interval", p["doc_id"]))
        if p["doc_id"] in alloc_pays:
            continue
        refs = p["invoice_refs"] or []
        exp = {}
        for ref in refs:
            inv = b.inv_by_num.get(ref["invoice_id"])
            if inv is None:
                out.append(("dangling_ref", p["doc_id"]))
                continue
            exp[ref["invoice_id"]] = exp.get(ref["invoice_id"], 0) + b.ar_credit(inv, p, ref["amount_minor"])
        got = {}
        for r in prow:
            if r["role"] == "ar":
                got[r["invoice_id"]] = got.get(r["invoice_id"], 0) + r["credit_minor"] - r["debit_minor"]
        if {k: v for k, v in got.items() if v} != {k: v for k, v in exp.items() if v}:
            out.append(("allocation", p["doc_id"]))
        u = p["amount_minor"] - sum(x["amount_minor"] for x in refs)
        adv = sum(r["credit_minor"] - r["debit_minor"] for r in prow if r["role"] == "advances")
        if adv != b.conv(u, p):
            out.append(("advances", p["doc_id"]))
        if any(r["role"] not in ("bank", "ar", "advances", "fx_gain", "fx_loss") for r in prow):
            out.append(("payment_roles", p["doc_id"]))
    return sorted(set(out))


def reaudit(ep):
    """Full re-audit of one generated episode. Returns a report dict; `ok` is the verdict."""
    b = books(ep.manifest, ep.policy, ep.records)
    findings = audit(b)
    leftovers = unexplained(b, findings)
    ledger = ep.records["ledger"]
    if sum(r["debit_minor"] for r in ledger) != sum(r["credit_minor"] for r in ledger):
        leftovers.append(("ledger_unbalanced", "*"))
    unref = [r for r in ledger if r["invoice_id"] is None and r["payment_id"] is None]
    if sum(r["debit_minor"] for r in unref) != sum(r["credit_minor"] for r in unref):
        leftovers.append(("expenses_unbalanced", "*"))
    res = Grader.for_episode(ep).score(
        {"episode_id": ep.episode_id, "findings": public_findings(findings)}, 0)
    m = res.metrics
    evidence_problems = verify_evidence_sets(ep)
    exact = res.valid and m["fp"] == 0 and m["fn"] == 0
    return {
        "episode_id": ep.episode_id, "exact": exact, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "evidence_ok": not evidence_problems, "evidence_problems": evidence_problems,
        "unexplained": leftovers,
        "ok": exact and not evidence_problems and not leftovers,
        "certificates": [{"type": f["type"], "affected_ids": f["affected_ids"], "text": f["_certificate"]}
                         for f in findings],
        "unmatched_findings": [f for i, f in enumerate(findings)
                               if i not in {fi for fi, _ in res.details["pairs"]}],
    }
