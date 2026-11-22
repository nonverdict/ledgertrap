"""Structural validation of findings.json (case §05; official matcher conventions of 20.11.2026).

A structure error invalidates the whole submission (reward -1, reason in the grader journal). Exactly three
kinds of structure errors exist:
  1. a value of the wrong JSON type (object/list/string/integer; bool is not an integer);
  2. a missing required field;
  3. an exceeded limit: more than 30 findings, more than 10 evidence ids in one finding (counted BEFORE
     duplicates are removed), explanation longer than 1000 Unicode code points (after JSON decoding).
Everything else is content, not structure: an unknown type string, empty or wrong affected_ids, a wrong
amount or currency, unknown evidence ids. Such a finding simply matches nothing and counts as a FP.
Unknown extra keys are ignored. A different episode_id is recorded in the journal, not penalised.
"""

MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
FINDING_FIELDS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _str_list(x):
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def validate_submission(obj):
    """Return (None, findings) when the structure is valid, else (reason_code, None)."""
    if not isinstance(obj, dict):
        return "NOT_OBJECT", None
    if "episode_id" not in obj or "findings" not in obj:
        return "MISSING_FIELD", None
    if not isinstance(obj["episode_id"], str):
        return "BAD_FIELD_TYPE", None
    findings = obj["findings"]
    if not isinstance(findings, list):
        return "BAD_FIELD_TYPE", None
    if len(findings) > MAX_FINDINGS:
        return "TOO_MANY_FINDINGS", None
    for f in findings:
        if not isinstance(f, dict):
            return "BAD_FIELD_TYPE", None
        if any(k not in f for k in FINDING_FIELDS):
            return "MISSING_FIELD", None
        if not isinstance(f["type"], str) or not isinstance(f["currency"], str):
            return "BAD_FIELD_TYPE", None
        if not isinstance(f["explanation"], str):
            return "BAD_FIELD_TYPE", None
        if not _str_list(f["affected_ids"]) or not _str_list(f["evidence_ids"]):
            return "BAD_FIELD_TYPE", None
        if not _is_int(f["claimed_amount_minor"]):
            return "BAD_FIELD_TYPE", None
        if len(f["evidence_ids"]) > MAX_EVIDENCE:
            return "TOO_MANY_EVIDENCE", None
        if len(f["explanation"]) > MAX_EXPLANATION:
            return "EXPLANATION_TOO_LONG", None
    return None, findings
