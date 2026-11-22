"""JSON Schemas (draft 2020-12 subset) for episode files, actions, observations, findings, trajectories,
plus a small stdlib validator. `python -m ledgertrap schemas` writes them to schemas/*.schema.json.

Supported keywords: type, enum, const, required, properties, additionalProperties, items, pattern,
minimum, maximum, minLength, maxLength, minItems, maxItems, oneOf, $ref (#/$defs/...).
"""

import json
import os
import re

ISO = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
DATE = r"^\d{4}-\d{2}-\d{2}$"
CUR = r"^[A-Z]{3}$"
RATE = r"^\d+(\.\d+)?$"
ID = r"^[A-Z0-9/:\-]+$"
INT = {"type": "integer"}
NN_INT = {"type": "integer", "minimum": 0}
STR = {"type": "string"}
NULL_ID = {"type": ["string", "null"], "pattern": ID}

LINE_ITEM = {"type": "object", "additionalProperties": False,
             "required": ["line_no", "sku", "description", "quantity", "unit_price_minor", "amount_minor",
                          "service_period_start", "service_period_end"],
             "properties": {"line_no": {"type": "integer", "minimum": 1}, "sku": STR, "description": STR,
                            "quantity": {"type": "integer", "minimum": 1}, "unit_price_minor": INT,
                            "amount_minor": INT, "service_period_start": {"type": "string", "pattern": DATE},
                            "service_period_end": {"type": "string", "pattern": DATE}}}

SCHEMAS = {
    "manifest": {
        "title": "manifest.json", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "generator_version", "config_version", "episode_id", "company_name",
                     "base_currency", "currencies", "timezone", "period_start", "period_end", "export_start",
                     "export_end", "counts", "rounding", "csv_null", "files"],
        "properties": {
            "schema_version": STR, "generator_version": STR, "config_version": STR,
            "episode_id": {"type": "string", "pattern": r"^EP-[A-Z0-9]{8}$"}, "company_name": STR,
            "base_currency": {"type": "string", "enum": ["KZT", "USD", "EUR"]},
            "currencies": {"type": "object", "additionalProperties": {"type": "integer", "enum": [0, 2, 3]}},
            "timezone": {"type": "string", "pattern": r"^[+-]\d{2}:\d{2}$"},
            "period_start": {"type": "string", "pattern": ISO}, "period_end": {"type": "string", "pattern": ISO},
            "export_start": {"type": "string", "pattern": ISO}, "export_end": {"type": "string", "pattern": ISO},
            "counts": {"type": "object", "additionalProperties": False,
                       "required": ["ledger", "invoices", "payments", "fx", "documents"],
                       "properties": {"ledger": {"type": "integer", "minimum": 200, "maximum": 2000},
                                      "invoices": NN_INT, "payments": NN_INT, "fx": NN_INT,
                                      "documents": {"type": "integer", "minimum": 20, "maximum": 200}}},
            "rounding": {"type": "string", "enum": ["HALF_UP", "HALF_EVEN"]},
            "csv_null": {"const": ""}, "files": {"type": "array", "items": STR}}},
    "ledger_row": {
        "title": "ledger.csv row (as returned by tools; empty CSV cell = null)", "type": "object",
        "additionalProperties": False,
        "required": ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id",
                     "invoice_id", "payment_id"],
        "properties": {"row_id": {"type": "string", "pattern": r"^JE-[A-Z0-9]{8}$"},
                       "timestamp": {"type": "string", "pattern": ISO}, "account": STR,
                       "debit_minor": NN_INT, "credit_minor": NN_INT, "currency": {"type": "string", "pattern": CUR},
                       "entity_id": NULL_ID, "invoice_id": NULL_ID, "payment_id": NULL_ID}},
    "invoice": {
        "title": "invoices.jsonl record", "type": "object", "additionalProperties": False,
        "required": ["doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor",
                     "line_items", "kind", "original_invoice_id"],
        "properties": {"doc_id": {"type": "string", "pattern": r"^D-[A-Z0-9]{8}$"},
                       "invoice_id": {"type": "string", "pattern": ID}, "entity_id": {"type": "string", "pattern": ID},
                       "issued_at": {"type": "string", "pattern": ISO}, "due_at": {"type": "string", "pattern": ISO},
                       "currency": {"type": "string", "pattern": CUR}, "total_minor": INT,
                       "line_items": {"type": "array", "minItems": 1, "items": LINE_ITEM},
                       "kind": {"type": "string", "enum": ["invoice", "credit_note"]},
                       "original_invoice_id": NULL_ID}},
    "payment": {
        "title": "payments.jsonl record", "type": "object", "additionalProperties": False,
        "required": ["doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs", "entity_id"],
        "properties": {"doc_id": {"type": "string", "pattern": r"^D-[A-Z0-9]{8}$"},
                       "payment_id": {"type": "string", "pattern": ID}, "timestamp": {"type": "string", "pattern": ISO},
                       "currency": {"type": "string", "pattern": CUR}, "amount_minor": INT,
                       "invoice_refs": {"type": ["array", "null"], "items": {
                           "type": "object", "additionalProperties": False, "required": ["invoice_id", "amount_minor"],
                           "properties": {"invoice_id": {"type": "string", "pattern": ID}, "amount_minor": INT}}},
                       "entity_id": {"type": "string", "pattern": ID}}},
    "fx_row": {
        "title": "fx.csv row", "type": "object", "additionalProperties": False,
        "required": ["date", "source_currency", "target_currency", "rate"],
        "properties": {"date": {"type": "string", "pattern": DATE}, "source_currency": {"type": "string", "pattern": CUR},
                       "target_currency": {"type": "string", "pattern": CUR},
                       "rate": {"type": "string", "pattern": RATE}}},
    "policy": {
        "title": "policy.json", "type": "object",
        "required": ["period", "timezone", "window_days", "fx_rate_date", "rounding", "conversion", "accounts",
                     "posting_rules", "issue_types", "submission"],
        "properties": {"rounding": {"type": "string", "enum": ["HALF_UP", "HALF_EVEN"]},
                       "accounts": {"type": "object"}, "posting_rules": {"type": "array", "items": STR},
                       "issue_types": {"type": "object", "required": ["duplicate_invoice", "payment_allocation",
                                                                      "fx_conversion", "period_cutoff"]}}},
    "action": {
        "title": "step(action)", "type": "object", "additionalProperties": False, "required": ["tool", "args"],
        "properties": {"tool": {"type": "string", "enum": ["list_records", "get_record", "get_document", "calculator",
                                                          "submit"]},
                       "args": {"type": "object"}}},
    "findings": {
        "title": "findings.json (submit); unknown extra keys are ignored by the grader", "type": "object",
        "required": ["episode_id", "findings"],
        "properties": {
            "episode_id": STR,
            "findings": {"type": "array", "maxItems": 30, "items": {
                "type": "object",
                "required": ["type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"],
                "properties": {"type": STR, "affected_ids": {"type": "array", "items": STR},
                               "claimed_amount_minor": INT, "currency": STR,
                               "evidence_ids": {"type": "array", "maxItems": 10, "items": STR},
                               "explanation": {"type": "string", "maxLength": 1000}}}}}},
    "observation": {
        "title": "observation", "oneOf": [
            {"type": "object", "required": ["type", "task", "manifest", "policy", "tools", "budget"],
             "properties": {"type": {"const": "reset"}}},
            {"type": "object", "additionalProperties": False,
             "required": ["type", "tool", "ok", "steps_left"],
             "properties": {"type": {"const": "tool_result"}, "tool": {"type": ["string", "null"]},
                            "ok": {"type": "boolean"}, "result": {"type": "object"},
                            "error": {"type": "object", "required": ["code", "message"],
                                      "properties": {"code": {"enum": ["BAD_ACTION", "UNKNOWN_TOOL", "BAD_ARGS",
                                                                       "UNKNOWN_TABLE", "UNKNOWN_ID", "BAD_CURSOR",
                                                                       "CALC_ERROR", "INTERNAL_ERROR",
                                                                       "EPISODE_DONE"]}}},
                            "steps_left": NN_INT}},
            {"type": "object", "additionalProperties": False, "required": ["type", "reason", "accepted", "steps_left"],
             "properties": {"type": {"const": "episode_end"},
                            "reason": {"enum": ["submitted", "invalid_submission", "max_steps", "time_limit",
                                                "policy_error"]},
                            "accepted": {"type": "boolean"}, "steps_left": NN_INT}}]},
    "judge_action": {
        "title": "judge_adapter: step(action)", "type": "object", "additionalProperties": False,
        "required": ["name", "args"],
        "properties": {"name": {"type": "string", "enum": ["list_records", "get_record", "get_document", "calculator",
                                                          "submit"]},
                       "args": {"type": "object"}}},
    "judge_observation": {
        "title": "judge_adapter: observation", "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["task", "manifest", "policy", "remaining_steps"],
             "properties": {"task": STR, "manifest": {"type": "object"}, "policy": {"type": "object"},
                            "remaining_steps": NN_INT}},
            {"type": "object", "additionalProperties": False, "required": ["result", "error", "remaining_steps"],
             "properties": {"result": {"type": ["object", "string", "null"]},
                            "error": {"oneOf": [{"type": "null"}, {
                                "type": "object", "additionalProperties": False, "required": ["code", "message"],
                                "properties": {"code": STR, "message": STR}}]},
                            "remaining_steps": NN_INT}}]},
    "score_result": {
        "title": "judge_adapter: score_episode(...) result; invalid structure -> return -1, the rest null",
        "type": "object", "additionalProperties": False,
        "required": ["tp", "fp", "fn", "precision", "recall", "f1", "return"],
        "properties": {"tp": {"type": ["integer", "null"], "minimum": 0},
                       "fp": {"type": ["integer", "null"], "minimum": 0},
                       "fn": {"type": ["integer", "null"], "minimum": 0},
                       "precision": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                       "recall": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                       "f1": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                       "return": {"type": "number", "minimum": -1, "maximum": 1}}},
    "trajectory_line": {
        "title": "trajectories.jsonl line", "type": "object",
        "required": ["episode_id", "step", "action", "observation", "reward", "terminated", "truncated"],
        "properties": {"episode_id": {"type": "string", "pattern": r"^EP-[A-Z0-9]{8}$"}, "step": NN_INT,
                       "action": {"type": ["object", "string", "null"]}, "observation": {"$ref": "#/$defs/observation"},
                       "reward": {"type": "number", "minimum": -1, "maximum": 1}, "terminated": {"type": "boolean"},
                       "truncated": {"type": "boolean"}, "policy": STR, "split": STR,
                       "mode": {"enum": ["train", "eval"]}, "env_version": STR}},
}
SCHEMAS["trajectory_line"]["$defs"] = {"observation": SCHEMAS["observation"]}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(value, schema, path="$", root=None):
    """Return a list of error strings (empty = valid)."""
    root = schema if root is None else root
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return validate(value, root["$defs"][name], path, root)
    errs = []
    if "oneOf" in schema:
        ok = [s for s in schema["oneOf"] if not validate(value, s, path, root)]
        return [] if len(ok) == 1 else ["%s: matches %d of oneOf" % (path, len(ok))]
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            return ["%s: expected %s" % (path, t)]
    if "const" in schema and value != schema["const"]:
        errs.append("%s: expected const %r" % (path, schema["const"]))
    if "enum" in schema and value not in schema["enum"]:
        errs.append("%s: not in enum" % path)
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append("%s: %r does not match %s" % (path, value[:40], schema["pattern"]))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append("%s: too long" % path)
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append("%s: too short" % path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append("%s: below minimum" % path)
        if "maximum" in schema and value > schema["maximum"]:
            errs.append("%s: above maximum" % path)
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append("%s: missing %s" % (path, k))
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], "%s.%s" % (path, k), root)
            elif extra is False:
                errs.append("%s: unexpected key %s" % (path, k))
            elif isinstance(extra, dict):
                errs += validate(v, extra, "%s.%s" % (path, k), root)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append("%s: too few items" % path)
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append("%s: too many items" % path)
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], "%s[%d]" % (path, i), root)
    return errs


def validate_episode(ep):
    errs = validate(ep.manifest, SCHEMAS["manifest"], "manifest")
    errs += validate(ep.policy, SCHEMAS["policy"], "policy")
    for name, table in (("ledger_row", "ledger"), ("invoice", "invoices"), ("payment", "payments"), ("fx_row", "fx")):
        for i, rec in enumerate(ep.records[table]):
            errs += validate(rec, SCHEMAS[name], "%s[%d]" % (table, i))
    return errs


def write_schema_files(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, schema in SCHEMAS.items():
        doc = dict(schema)
        doc["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        doc["$id"] = "ledgertrap/%s.schema.json" % name
        with open(os.path.join(out_dir, "%s.schema.json" % name), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
