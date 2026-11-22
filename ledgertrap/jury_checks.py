"""Extended internal checks (docs/JURY_CHECKS.md, part 2). Not scored: the official definitions
(genesis-exe.com/fintech-checks.html, 20.11.2026) are run by official_checks.py. These go further on the same
criteria with our own generator-side access (provenance roll-back, evidence minimality, attacks, G gate).

A (contract & isolation) and C (reward robustness): 20 fixed checks each.
B (data & truth): 20 hidden-like episodes; an episode counts only if ALL of B01-B20 hold (N/A = pass).
D (reproducibility): 20 pairs; a pair counts only if ALL of D01-D12 hold.
G: global reproducibility gate.
"""

import copy
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

from . import config as config_mod
from .audit_rules import Books, audit
from .env import LedgerTrapEnv
from .generator import generate
from .grader import Grader
from .grader.normalize import tolerance_for, tolerances_from_policy
from .idmap import id_map, presentation, remap_findings, remap_truth, story_fingerprint
from .leakscan import scan_records
from .policies.agents import make_policy
from .policies.baselines import flag_all_findings, oracle_findings
from .reaudit import books as rebooks
from .reaudit import reaudit, unexplained, verify_evidence_sets
from .runner import builtin_agent, run_episode
from .store import RecordStore
from .timeutil import ISO_RE, last_business_on_or_before, parse, parse_offset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JURY_SECRET = "jury-simulation-secret"


def _load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def jury_episodes():
    """20 hidden-like episodes (own secret, unseen seeds): 5 clean, 5 interacting, 10 mixed."""
    out = []
    for i in range(5):
        out.append(("clean", 9000 + i, {"errors": {"mode": "clean"}, "traps": {"rate": 1.0}}))
    inter = [{"C1": 1}, {"C2": 1}, {"C1": 1, "C2": 1}, {"C1": 1, "fx_conversion:wrong_rate": 1},
             {"C2": 1, "period_cutoff:late": 1, "payment_allocation:partial": 1}]
    for i, plan in enumerate(inter):
        out.append(("interacting", 9005 + i, {"errors": {"mode": "exact", "exact": plan},
                                              "size": {"documents": [60, 140]},
                                              "company": {"n_foreign_currencies": [1, 3]}}))
    mixed = ["configs/presets/L2.json", "configs/presets/L3.json", "configs/presets/L4.json",
             "configs/presets/L5.json", "configs/dev.json"]
    for i in range(10):
        cfg = _load(mixed[i % len(mixed)])
        if i % 2 == 0:
            cfg = dict(cfg, errors=dict(cfg.get("errors", {}), clean_probability=0.0))
        out.append(("mixed", 9010 + i, cfg))
    return out


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _env(seed=1, cfg=None, mode="train", grader="inproc", clock=None):
    kw = {"mode": mode, "secret": JURY_SECRET, "grader": grader}
    if clock is not None:
        kw["clock"] = clock
    env = LedgerTrapEnv(**kw)
    obs, info = env.reset(seed, cfg)
    return env, obs, info


def _calc(env):
    return env.step({"tool": "calculator", "args": {"expression": "1+1"}})


def _submit(env, obj):
    return env.step({"tool": "submit", "args": {"findings": obj}})


# ---------------------------------------------------------------------------------------------
# A. Contract & isolation (20 fixed checks)
# ---------------------------------------------------------------------------------------------

def A01():
    ok = True
    for mode in ("train", "eval"):
        env = LedgerTrapEnv(mode=mode, secret=JURY_SECRET)
        out = env.reset(1, None)
        json.dumps(out)
        ok &= isinstance(out, tuple) and len(out) == 2 and ((out[1] == {}) == (mode == "eval"))
    return ok


def A02():
    env, obs, info = _env()
    return (all(k in obs for k in ("task", "manifest", "policy", "tools", "budget"))
            and obs["budget"]["steps_left"] == 100
            and all("evidence_rule" in c for t in obs["policy"]["issue_types"].values() for c in t["cases"].values()))


def A03():
    env, obs, info = _env()
    ok = True
    for a in ({"tool": "list_records", "args": {"table": "ledger"}},
              {"tool": "get_record", "args": {"table": "fx", "id": "x"}},
              {"tool": "get_document", "args": {"doc_id": "x"}},
              {"tool": "calculator", "args": {"expression": "2"}}):
        out = env.step(a)
        json.dumps(out)
        ok &= (len(out) == 5 and isinstance(out[0], dict) and isinstance(out[1], float)
               and isinstance(out[2], bool) and isinstance(out[3], bool) and isinstance(out[4], dict))
    return ok


def A04():
    ep = generate(3, None, JURY_SECRET)
    m = ep.manifest
    need = ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts",
            "rounding", "currencies", "export_start", "export_end")
    return (all(k in m for k in need) and "config_hash" not in m
            and m["counts"]["ledger"] == len(ep.records["ledger"])
            and m["counts"]["documents"] == len(ep.records["invoices"]) + len(ep.records["payments"]))


def A05():
    from .schemas import validate_episode
    ep = generate(4, {"company": {"n_foreign_currencies": [2, 3]}}, JURY_SECRET)
    head = ep.files["ledger.csv"].decode().splitlines()[0]
    fxh = ep.files["fx.csv"].decode().splitlines()[0]
    return (head == "row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id"
            and fxh == "date,source_currency,target_currency,rate" and not validate_episode(ep))


def A06():
    ep = generate(5, {"company": {"n_foreign_currencies": [2, 3]}}, JURY_SECRET)
    ok = all(type(r["debit_minor"]) is int and type(r["credit_minor"]) is int and ISO_RE.match(r["timestamp"])
             for r in ep.records["ledger"])
    ok &= all(re.match(r"^\d+(\.\d+)?$", f["rate"]) for f in ep.records["fx"])
    ok &= all(ISO_RE.match(d["issued_at"]) and ISO_RE.match(d["due_at"]) for d in ep.records["invoices"])
    ok &= all(set(("entity_id", "invoice_id", "payment_id")) <= set(r) for r in ep.records["ledger"])
    return ok


def A07():
    env, obs, info = _env(6)
    ok = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})[0]["error"]["code"] == "BAD_ARGS"
    got, cur, first = [], None, None
    while True:
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cur}})[0]
        ok &= o["ok"] and len(o["result"]["rows"]) <= 50
        got += o["result"]["rows"]
        cur = o["result"]["next_cursor"]
        first = cur if first is None else first
        if cur is None:
            break
    ok &= len(got) == obs["manifest"]["counts"]["ledger"] == len({r["row_id"] for r in got})
    a = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": first}})[0]
    b = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": first}})[0]
    ok &= a["result"] == b["result"]
    ok &= env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": "zz"}})[0]["error"]["code"] == "BAD_CURSOR"
    return ok


def A08():
    env, obs, info = _env(7)
    inv = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]["result"]["rows"][0]
    ok = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["ok"]
    ok &= env.step({"tool": "get_record", "args": {"table": "invoices", "id": inv["invoice_id"]}})[0]["ok"]
    o = env.step({"tool": "get_record", "args": {"table": "ledger", "id": "SECRET-ID-42"}})[0]
    return ok and o["error"]["code"] == "UNKNOWN_ID" and "SECRET-ID-42" not in json.dumps(o) and o["steps_left"] == 96


def A09():
    env, obs, info = _env(8)
    rng = random.Random(0)
    junk = [None, 1, "x", [], {"tool": "zz", "args": {}}, {"tool": "list_records", "args": {"table": 5}},
            {"tool": "calculator", "args": {}}, {"tool": "calculator", "args": {"expression": "x" * 2000000}},
            {"tool": "list_records"}, {"tool": "get_record", "args": {"table": "ledger", "id": None}}]
    codes = set()
    o = None
    for _ in range(90):
        o, r, t, tr, info = env.step(copy.deepcopy(rng.choice(junk)))
        if t or tr or r != 0.0:
            return False
        codes.add(o["error"]["code"])
    return codes <= {"BAD_ACTION", "UNKNOWN_TOOL", "BAD_ARGS"} and o["steps_left"] == 10


def A10():
    env, obs, info = _env(9)
    good = env.step({"tool": "calculator", "args": {"expression": "(1.5+2)*3/7"}})[0]["result"]["value"] == "1.5"
    bad = all(env.step({"tool": "calculator", "args": {"expression": e}})[0]["error"]["code"] == "CALC_ERROR"
              for e in ("__import__('os')", "2**3", "abs(1)", "1/0", "1" * 201, "open('x')"))
    return good and bad


def A11():
    env, obs, info = _env(10)
    o, r, t, tr, i = _submit(env, env.oracle_findings())
    return t and not tr and o["reason"] == "submitted" and r == 1.0


def A12():
    good = {"type": "period_cutoff", "affected_ids": ["X"], "claimed_amount_minor": 1, "currency": "USD",
            "evidence_ids": [], "explanation": ""}
    # Structure errors are exactly: wrong JSON type, missing field, exceeded limit (official conventions 20.11).
    missing = dict(good)
    del missing["explanation"]
    cases = [("garbage", False), ({"findings": []}, False), ({"episode_id": 5, "findings": []}, False),
             ({"findings": [good] * 31}, True), ({"findings": [dict(good, evidence_ids=["e"] * 11)]}, True),
             ({"findings": [dict(good, explanation="x" * 1001)]}, True),
             ({"findings": [dict(good, claimed_amount_minor=1.0)]}, True),
             ({"findings": [dict(good, claimed_amount_minor=True)]}, True),
             ({"findings": [missing]}, True)]
    ok = True
    for extra in ({"episode_id": "EP-NOPE", "findings": []}, {"findings": [dict(good, x=1)], "note": 1}):
        env, obs, info = _env(11)  # not structure errors: content is scored (here: FP or empty answer)
        if "episode_id" not in extra:
            extra = dict(extra, episode_id=obs["manifest"]["episode_id"])
        o, r, t, tr, i = _submit(env, extra)
        ok &= bool(t and o["reason"] == "submitted" and env.journal[-1]["valid"])
    for sub, add_episode in cases:
        env, obs, info = _env(11)
        if add_episode:
            sub = dict(sub, episode_id=obs["manifest"]["episode_id"])
        o, r, t, tr, i = _submit(env, sub)
        ok &= bool(t and r == -1.0 and o["reason"] == "invalid_submission"
                   and env.journal[-1]["metrics"]["invalid_reason"])
    env, obs, info = _env(11)
    o, r, t, tr, i = env.step({"tool": "submit", "args": {"wrong": 1}})
    return ok and t and r == -1.0


def A13():
    env, obs, info = _env(12)
    for _ in range(99):
        _calc(env)
    o, r, t, tr, i = _calc(env)
    env2, obs2, _ = _env(12)
    for _ in range(99):
        _calc(env2)
    t2 = _submit(env2, {"episode_id": obs2["manifest"]["episode_id"], "findings": []})[2]
    return tr and not t and o["reason"] == "max_steps" and i["metrics"]["inspection_steps"] == 100 and t2


def A14():
    clock = _Clock()
    env, obs, info = _env(13, clock=clock)
    clock.t = 61
    o, r, t, tr, i = _calc(env)
    ok = tr and o["reason"] == "time_limit"
    from .runner import SubprocessPolicy
    env = LedgerTrapEnv(mode="eval", secret=JURY_SECRET)
    crash = SubprocessPolicy([sys.executable, "-c", "import sys; sys.exit(3)"], "crash")
    j1 = run_episode(env, crash, 13, None, "a", "crash")[1]
    crash.close()
    hang = SubprocessPolicy([sys.executable, "-c", "import sys, time\nfor l in sys.stdin:\n    time.sleep(30)\n"],
                            "hang")
    t0 = time.time()
    j2 = run_episode(env, hang, 13, {"limits": {"time_limit_s": 1}}, "a", "hang")[1]
    hang.close()
    return ok and j1["reason"] == "policy_error" and j2["reason"] == "time_limit" and time.time() - t0 < 10


def A15():
    env, obs, info = _env(14)
    _submit(env, env.oracle_findings())
    o, r, t, tr, i = _calc(env)
    r2 = _submit(env, env.oracle_findings())[1]
    return o["error"]["code"] == "EPISODE_DONE" and r == 0.0 and r2 == 0.0 and len(env.journal) == 1


def A16():
    env, obs, info = _env(15, mode="eval", grader="process")
    ok = info == {}
    o, r, t, tr, i = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    ok &= r == 0.0 and i == {}
    o, r, t, tr, i = _submit(env, env.oracle_findings())
    ok &= (r == 0.0 and i == {} and abs(env.journal[-1]["metrics"]["reward_main"] - (1.0 - 0.002)) < 1e-9
           and "metrics" not in json.dumps(o))
    env.close()
    return ok


def A17():
    env, obs, info = _env(16)
    rs = [env.step({"tool": "list_records", "args": {"table": t}})[1] for t in ("ledger", "invoices", "fx", "payments")]
    return rs == [0.0] * 4


def A18():
    probs = []
    for mode in ("train", "eval"):
        env = LedgerTrapEnv(mode=mode, secret=JURY_SECRET)
        for name in ("auditor", "naive", "random", "flag_all"):
            traj, j = run_episode(env, make_policy(name), 17, None, "a", name)
            ep = env.privileged_episode()
            probs += scan_records(traj, {ep.episode_id: ep.truth}, secrets=[JURY_SECRET])
    return not probs


def A19():
    env = LedgerTrapEnv(mode="eval", secret=JURY_SECRET, grader="process")
    err = os.path.join(ROOT, "build", "probe.err")
    os.makedirs(os.path.dirname(err), exist_ok=True)
    ag = builtin_agent("probe", stderr_path=err)
    run_episode(env, ag, 18, None, "a", "probe")
    ag.close()
    try:
        env.privileged_episode()
        truth_absent = False
    except RuntimeError:
        truth_absent = True
    env.close()
    with open(err, encoding="utf-8") as f:
        seen = json.loads(f.readline())
    env2, obs, info = _env(18)
    bad_tables = all(env2.step({"tool": "get_record", "args": {"table": t, "id": "ISS-01"}})[0]["error"]["code"]
                     == "UNKNOWN_TABLE" for t in ("truth", "provenance", "policy"))
    bad_path = env2.step({"tool": "get_document", "args": {"doc_id": "../grader/truth.json"}})[0]["error"]["code"]
    return (not any("SECRET" in k.upper() for k in seen["env_keys"]) and seen["cwd_files"] == []
            and seen["parent_environ_has_secret"] is not True and bad_tables and bad_path == "UNKNOWN_ID"
            and truth_absent)


def A20():
    env, obs, info = _env(19)
    rid = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]["result"]["rows"][0]["row_id"]
    obs2, _ = env.reset(20, None)
    o = env.step({"tool": "get_record", "args": {"table": "ledger", "id": rid}})[0]
    return o["error"]["code"] == "UNKNOWN_ID" and o["steps_left"] == 99 and obs2["budget"]["steps_left"] == 100


A_CHECKS = [A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20]


# ---------------------------------------------------------------------------------------------
# B. Data & truth (per episode)
# ---------------------------------------------------------------------------------------------

def revert_mutations(ep):
    """Undo provenance mutations on the public records -> (ledger, invoices, payments) of the clean books."""
    ledger = {r["row_id"]: dict(r) for r in ep.records["ledger"]}
    restored, added_docs = [], set()
    for m in ep.provenance["mutations"]:
        if m["op"] == "add_doc":
            added_docs.add(m["target"])
        elif m["op"] in ("add_rows", "update_rows"):
            for r in m["after"]:
                ledger.pop(r["row_id"], None)
            for r in m.get("before", []):
                restored.append({k: v for k, v in r.items() if k != "deleted"})
    rows = list(ledger.values())
    for i, r in enumerate(restored):
        if r["row_id"] is None:
            r["row_id"] = "RESTORED-%d" % i
        rows.append(r)
    gone = [d for d in ep.records["invoices"] + ep.records["payments"] if d["doc_id"] in added_docs]
    nums = {d.get("invoice_id") for d in gone} | {d.get("payment_id") for d in gone}
    nums.discard(None)
    rows = [r for r in rows if r["invoice_id"] not in nums and r["payment_id"] not in nums]
    invoices = [d for d in ep.records["invoices"] if d["doc_id"] not in added_docs]
    payments = [d for d in ep.records["payments"] if d["doc_id"] not in added_docs]
    return rows, invoices, payments, nums


LUHN_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+\d[\d\s()-]{9,}")


def _luhn(s):
    total, alt = 0, False
    for ch in reversed(s):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def b_checks_episode(seed, cfg):
    ep = generate(seed, cfg, JURY_SECRET)
    return ep, b_checks(ep)


def b_checks(ep):
    """B01-B20 on one episode (generated, or loaded from a folder with its grader files)."""
    cfg = ep.provenance["config"]
    tol = tolerances_from_policy(ep.policy)
    rep = reaudit(ep)
    r, m = ep.records, ep.manifest
    docs = {d["doc_id"]: d for d in r["invoices"] + r["payments"]}
    rows = {x["row_id"]: x for x in r["ledger"]}
    fx_ids = {"%s:%s:%s" % (f["date"], f["source_currency"], f["target_currency"]) for f in r["fx"]}
    tz = parse_offset(m["timezone"])
    issues = ep.truth["issues"]
    c = {}
    led0, inv0, pay0, added_nums = revert_mutations(ep)
    b0 = Books(m, ep.policy, led0, inv0, pay0, r["fx"])
    c["B01"] = not audit(b0) and not unexplained(b0, [])
    c["B02"] = (m["counts"]["ledger"] == len(r["ledger"]) and m["counts"]["documents"] == len(docs)
                and 200 <= len(r["ledger"]) <= 2000 and 20 <= len(docs) <= 200)
    nums = {d.get("invoice_id") for d in r["invoices"]} | {d.get("payment_id") for d in r["payments"]}
    c["B03"] = (all(x["invoice_id"] in nums or x["invoice_id"] is None for x in r["ledger"])
                and all(x["payment_id"] in nums or x["payment_id"] is None for x in r["ledger"])
                and all(ref["invoice_id"] in nums for p in r["payments"] for ref in (p["invoice_refs"] or []))
                and all(d["original_invoice_id"] in nums for d in r["invoices"] if d["original_invoice_id"]))
    fx = {(f["date"], f["source_currency"]) for f in r["fx"]}
    cov = True
    for d in docs.values():
        if d["currency"] != m["base_currency"]:
            rd = last_business_on_or_before(parse(d.get("issued_at") or d["timestamp"]).astimezone(tz).date())
            cov &= (rd.isoformat(), d["currency"]) in fx
    series = {}
    for f in r["fx"]:
        series.setdefault(f["source_currency"], []).append(f["rate"])
    c["B04"] = cov and all(a != b for s in series.values() for a, b in zip(s, s[1:]))
    c["B05"] = all(type(x["quantity"]) is int and x["quantity"] >= 1
                   and x["quantity"] * x["unit_price_minor"] == x["amount_minor"]
                   for d in r["invoices"] for x in d["line_items"]) and all(
        sum(x["amount_minor"] for x in d["line_items"]) == d["total_minor"] for d in r["invoices"])
    es, ee = parse(m["export_start"]), parse(m["export_end"])
    c["B06"] = all(es <= parse(x["timestamp"]) < ee for x in r["ledger"]) and all(
        es <= parse(d.get("issued_at") or d["timestamp"]) < ee for d in docs.values())
    c["B07"] = all(a in docs for g in issues for a in g["affected_ids"])
    c["B08"] = not verify_evidence_sets(ep) and all(
        x in docs or x in rows or x in fx_ids for g in issues for s in g["evidence_sets"] for x in s)
    ok9 = True
    for g in issues:
        found = audit(rebooks(m, ep.policy, r, g["evidence_sets"][0]))
        ok9 &= any(f["type"] == g["type"] and abs(f["claimed_amount_minor"] - g["expected_amount_minor"])
                   <= tolerance_for(tol, g["type"]) for f in found)
    c["B09"] = ok9 and len(ep.certificates["certificates"]) == len(issues)
    c["B10"] = rep["exact"]
    c["B11"] = not rep["unexplained"]
    # B12: re-applying the provenance mutations to the clean books reproduces the published ledger.
    reapplied = {x["row_id"]: x for x in led0 if not x["row_id"].startswith("RESTORED-")}
    for mu in ep.provenance["mutations"]:
        if mu["op"] in ("add_rows", "update_rows"):
            for x in mu["before"] if mu["op"] == "update_rows" else []:
                if x.get("row_id") and x["row_id"] in reapplied:
                    del reapplied[x["row_id"]]
            for x in mu["after"]:
                reapplied[x["row_id"]] = {k: v for k, v in x.items() if k != "deleted"}
    for x in r["ledger"]:  # clean postings of documents added by scenarios come back with the documents
        if (x["invoice_id"] in added_nums or x["payment_id"] in added_nums) and x["row_id"] not in reapplied:
            reapplied[x["row_id"]] = x
    c["B12"] = sorted(json.dumps(x, sort_keys=True) for x in reapplied.values()) == \
        sorted(json.dumps(x, sort_keys=True) for x in r["ledger"])
    c["B13"] = (not ep.truth["is_clean"]) or (not issues and rep["fp"] == 0
                                              and (bool(ep.provenance["traps_planted"]) or not cfg["traps"]["enabled"]))
    ok14 = True
    for g in issues:
        if g["scenario"] == "C1" and g["type"] == "fx_conversion":
            p = docs[g["affected_ids"][0]]
            local = parse(p["timestamp"]).astimezone(tz).date().isoformat()
            ok14 &= local == m["period_end"][:10] and p["timestamp"][:10] < m["period_end"][:10]
            ok14 &= any(h["scenario"] == "C1" and h["type"] == "period_cutoff" and h["affected_ids"] == g["affected_ids"]
                        for h in issues)
        if g["scenario"] == "C2" and g["type"] == "payment_allocation":
            p = next(docs[x] for x in g["affected_ids"] if "payment_id" in docs[x])
            prow = [x for x in r["ledger"] if x["payment_id"] == p["payment_id"] and x["invoice_id"]]
            ok14 &= bool(prow) and all(x["invoice_id"] != p["invoice_refs"][0]["invoice_id"] for x in prow)
    c["B14"] = ok14
    keys = Counter((g["type"], tuple(g["affected_ids"])) for g in issues)
    owner, disjoint = {}, True
    for g in issues:
        for a in g["affected_ids"]:
            tag = g["scenario"] or g["issue_id"]
            if a in owner and owner[a] != tag:
                disjoint = False
            owner[a] = tag
    c["B15"] = disjoint and max(keys.values() or [1]) == 1 and len(issues) <= 12
    traps_on = bool(cfg["traps"]["enabled"]) and cfg["traps"]["rate"] > 0
    c["B16"] = rep["fp"] == 0 and (bool(ep.provenance["traps_planted"]) or not traps_on)
    c["B17"] = all(g["expected_amount_minor"] >= 100 and g["expected_amount_minor"] > tolerance_for(tol, g["type"])
                   for g in issues)
    store = RecordStore(ep)
    ok18 = True
    for g in issues:
        for s in g["evidence_sets"]:
            for x in s:
                try:
                    if x in docs:
                        store.get_document(x)
                    elif x in rows:
                        store.get_record("ledger", x)
                    else:
                        store.get_record("fx", x)
                except Exception:  # noqa: BLE001
                    ok18 = False
    c["B18"] = ok18
    sig = Counter((d["entity_id"], d["currency"], json.dumps(d["line_items"], sort_keys=True))
                  for d in r["invoices"] if d["kind"] == "invoice")
    declared = sum(1 for g in issues if g["variant"] == "document")
    c["B19"] = sum(n - 1 for n in sig.values()) == declared and rep["fp"] == 0
    blob = b"".join(ep.files.values()).decode("utf-8")
    cards = [s for s in LUHN_RE.findall(blob) if _luhn(s)]
    c["B20"] = (not cards and not IBAN_RE.search(blob) and not EMAIL_RE.search(blob) and not PHONE_RE.search(blob)
                and "(вымышл.)" in m["company_name"])
    return c


# ---------------------------------------------------------------------------------------------
# C. Reward robustness (20 fixed checks)
# ---------------------------------------------------------------------------------------------

def _score(ep, findings, steps=0):
    return Grader.for_episode(ep).score({"episode_id": ep.episode_id, "findings": findings}, steps)


def _orc(ep):
    return oracle_findings(ep)["findings"]


def _doc_ids(ep):
    return [d["doc_id"] for d in ep.records["invoices"] + ep.records["payments"]]


def c_checks(jury):
    dirty = [generate(s, cfg, JURY_SECRET) for k, s, cfg in jury if k != "clean"][:8]
    clean = [generate(s, cfg, JURY_SECRET) for k, s, cfg in jury if k == "clean"]
    rng = random.Random(1)
    res = {}
    res["C01"] = all(_score(e, _orc(e)).reward == 1.0 for e in dirty + clean)
    res["C02"] = all(_score(e, _orc(e)).reward > _score(e, []).reward for e in dirty)

    def false_acc(e):
        d = e.records["invoices"][0]
        return [{"type": "period_cutoff", "affected_ids": [d["doc_id"]], "claimed_amount_minor": abs(d["total_minor"]),
                 "currency": d["currency"], "evidence_ids": [d["doc_id"], e.records["ledger"][0]["row_id"]],
                 "explanation": ""}]
    res["C03"] = all(_score(e, []).reward > _score(e, false_acc(e)).reward for e in clean)
    res["C04"] = all(_score(e, _orc(e) + _orc(e)[:1]).reward < 1.0
                     and _score(e, _orc(e) + _orc(e)[:1]).metrics["fp"] == 1 for e in dirty)
    weak = [_score(e, flag_all_findings(e.episode_id, e.records["invoices"], e.records["payments"])["findings"]).reward
            for e in dirty + clean]
    strong = []
    for e in dirty + clean:
        rows_of = {}
        for x in e.records["ledger"]:
            rows_of.setdefault(x["payment_id"] or x["invoice_id"], []).append(x["row_id"])
        f = []
        for d in (e.records["invoices"] + e.records["payments"])[:30]:
            num = d.get("invoice_id") or d.get("payment_id")
            f.append({"type": "period_cutoff", "affected_ids": [d["doc_id"]],
                      "claimed_amount_minor": abs(d.get("total_minor", d.get("amount_minor"))),
                      "currency": d["currency"], "evidence_ids": ([d["doc_id"]] + rows_of.get(num, []))[:10],
                      "explanation": ""})
        strong.append(_score(e, f).reward)
    res["C05"] = sum(weak) / len(weak) <= -0.5 and sum(strong) / len(strong) <= -0.5 and max(strong) < 0
    res["C06"] = all(abs(_score(e, _orc(e), 0).reward - _score(e, _orc(e), 50).reward - 0.1) < 1e-9 for e in dirty)
    guess_tp = 0
    for e in dirty:
        truth_keys = {(g["type"], tuple(g["affected_ids"])) for g in e.truth["issues"]}
        rows = [x["row_id"] for x in e.records["ledger"]]
        f = []
        for _ in range(10):
            d = rng.choice(_doc_ids(e))
            if ("period_cutoff", (d,)) not in truth_keys:
                f.append({"type": "period_cutoff", "affected_ids": [d], "claimed_amount_minor": 100, "currency": "USD",
                          "evidence_ids": [d] + rng.sample(rows, 5), "explanation": ""})
        guess_tp += _score(e, f).metrics["tp"]
    res["C07"] = guess_tp == 0
    ok8 = ok9 = ok11 = ok12 = True
    for e in dirty:
        o = _orc(e)
        # Issues on different documents (the two C1 issues share P2, so their evidence is legitimately shared).
        for f, g in ((f, g) for f in o for g in o if not set(f["affected_ids"]) & set(g["affected_ids"])):
            ok8 &= _score(e, [dict(f, evidence_ids=g["evidence_ids"])]).metrics["tp"] == 0
            m = _score(e, [dict(f, affected_ids=f["affected_ids"] + g["affected_ids"])]).metrics
            ok12 &= m["tp"] == 0 and m["fn"] == len(o)
        ok9 &= _score(e, [dict(f, currency="KWD" if f["currency"] != "KWD" else "JPY") for f in o]).metrics["tp"] == 0
        sub = [dict(f, affected_ids=f["affected_ids"][:1]) for f in o if len(f["affected_ids"]) > 1]
        extra = _doc_ids(e)[0]
        sup = [dict(f, affected_ids=f["affected_ids"] + [extra]) for f in o if extra not in f["affected_ids"]]
        ok11 &= _score(e, sub).metrics["tp"] == 0 and _score(e, sup).metrics["tp"] == 0
    res["C08"], res["C09"], res["C11"], res["C12"] = ok8, ok9, ok11, ok12
    ok10 = True
    for e in dirty:
        for f in _orc(e):
            t = tolerance_for(tolerances_from_policy(e.policy), f["type"])
            ok10 &= _score(e, [dict(f, claimed_amount_minor=f["claimed_amount_minor"] + t)]).metrics["tp"] == 1
            ok10 &= _score(e, [dict(f, claimed_amount_minor=f["claimed_amount_minor"] - t)]).metrics["tp"] == 1
            ok10 &= _score(e, [dict(f, claimed_amount_minor=f["claimed_amount_minor"] + t + 1)]).metrics["tp"] == 0
    res["C10"] = ok10
    ok13 = True
    for e in dirty[:4]:
        pool = [x["row_id"] for x in e.records["ledger"]] + _doc_ids(e)
        truth_ev = [set(s) for g in e.truth["issues"] for s in g["evidence_sets"]]
        for f in _orc(e):
            for _ in range(150):
                ev = rng.sample(pool, 10)
                tp = _score(e, [dict(f, evidence_ids=ev)]).metrics["tp"]
                ok13 &= tp == 0 or any(s <= set(ev) for s in truth_ev)
    res["C13"] = ok13
    env, obs, info = _env(2)
    _submit(env, env.oracle_findings())
    o = _submit(env, [])[0]
    res["C14"] = (o["error"]["code"] == "EPISODE_DONE" and env.journal[-1]["metrics"]["reward_main"] == 1.0
                  and len(env.journal) == 1)
    env, obs, info = _env(3)
    _, rg, tg, _, _ = _submit(env, 12345)
    env, obs, info = _env(3)
    o, rb, tb, _, _ = env.step("not json at all")
    res["C15"] = rg == -1.0 and tg and o["error"]["code"] == "BAD_ACTION" and not tb
    env, obs, info = _env(4, cfg={"errors": {"mode": "clean"}})
    for _ in range(100):
        out = _calc(env)
    clock = _Clock()
    env2, obs2, _ = _env(5, cfg={"errors": {"mode": "exact", "exact": {"period_cutoff:late": 1}}}, clock=clock)
    clock.t = 99
    out2 = _calc(env2)
    res["C16"] = (out[3] and abs(out[1] - 0.8) < 1e-9 and out2[3] and out2[1] == -1.0
                  and out2[4]["metrics"]["tp"] == 0)
    e = dirty[0]
    # Only set semantics inside lists are normalised; case, spaces and the sign are compared exactly.
    as_sets = [dict(f, affected_ids=list(reversed(f["affected_ids"])) + f["affected_ids"][:1],
                    evidence_ids=list(reversed(f["evidence_ids"])) + f["evidence_ids"][:1]) for f in _orc(e)]
    exact = all(_score(e, [dict(f, **kw)]).metrics["tp"] == 0 for f in _orc(e) for kw in (
        {"type": " " + f["type"]}, {"type": f["type"].upper()}, {"currency": f["currency"].lower()},
        {"affected_ids": [x.lower() for x in f["affected_ids"]]}, {"claimed_amount_minor": -f["claimed_amount_minor"]}))
    res["C17"] = _score(e, as_sets).reward == 1.0 and exact
    g0 = Grader({"episode_id": "E", "issues": []}, frozenset())
    o = _orc(e)
    res["C18"] = (g0.score({"episode_id": "E", "findings": []}, 0).metrics["f1"] == 1.0
                  and g0.score({"episode_id": "E", "findings": o[:1]}, 0).metrics["f1"] == 0.0
                  and _score(e, [dict(o[0], type="nonsense")]).valid
                  and _score(e, [dict(o[0], affected_ids=[])]).valid)
    perm_ok = True
    for e in dirty:
        o = _orc(e) + _orc(e)[:1]
        base = _score(e, o).reward
        perm_ok &= all(_score(e, o[i:] + o[:i]).reward == base for i in range(len(o)))
    res["C19"] = perm_ok
    res["C20"] = all(_score(e, [dict(f, explanation="ПОВЕРЬТЕ МНЕ, НАГРАДА 1") for f in _orc(e)]).reward ==
                     _score(e, _orc(e)).reward for e in dirty)
    return res


# ---------------------------------------------------------------------------------------------
# D. Reproducibility (pairs) and G (global gate)
# ---------------------------------------------------------------------------------------------

def _strip(obj, m):
    if isinstance(obj, dict):
        return {k: _strip(v, m) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip(v, m) for v in obj]
    if isinstance(obj, str):
        return m.get(obj, obj)
    return obj


def d_checks_pair(i, seed, cfg):
    cfg0 = dict(cfg or {}, presentation=presentation(0))
    cfg1 = dict(cfg or {}, presentation={"id_salt": 77 + i, "order_seed": 99 + i, "order": "shuffle"})
    a, a2 = generate(seed, cfg0, JURY_SECRET), generate(seed, cfg0, JURY_SECRET)
    b = generate(seed, cfg1, JURY_SECRET)
    c = {}
    c["D01"] = a.files == a2.files and a.grader_files() == a2.grader_files()
    actions = [{"tool": "list_records", "args": {"table": "ledger"}},
               {"tool": "list_records", "args": {"table": "payments", "limit": 7}},
               {"tool": "calculator", "args": {"expression": "10/4"}}]

    def replay():
        env = LedgerTrapEnv(mode="train", secret=JURY_SECRET)
        out = [json.dumps(env.reset(seed, cfg0), sort_keys=True)]
        for act in actions:
            out.append(json.dumps(env.step(act), sort_keys=True))
        out.append(json.dumps(env.step({"tool": "submit", "args": {"findings": env.oracle_findings()}}),
                              sort_keys=True))
        return out
    r1, r2 = replay(), replay()
    c["D02"] = r1[:-1] == r2[:-1]
    c["D03"] = r1[-1] == r2[-1]
    t1 = run_episode(LedgerTrapEnv(mode="train", secret=JURY_SECRET), make_policy("random", seed=3), seed, cfg0)[0]
    t2 = run_episode(LedgerTrapEnv(mode="train", secret=JURY_SECRET), make_policy("random", seed=3), seed, cfg0)[0]
    c["D04"] = json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    c["D05"] = story_fingerprint(a) == story_fingerprint(b)
    m = id_map(a, b)
    c["D06"] = all(k != v for k, v in m.items())
    c["D07"] = len(set(m.values())) == len(m) and remap_truth(a.truth, m) == b.truth["issues"]
    c["D08"] = Grader.for_episode(b).score(remap_findings(oracle_findings(a), m), 0).reward == 1.0
    c["D09"] = all(sorted(json.dumps(_strip(x, m), sort_keys=True) for x in a.records[t]) ==
                   sorted(json.dumps(x, sort_keys=True) for x in b.records[t]) for t in a.records) \
        and a.files["ledger.csv"] != b.files["ledger.csv"]

    def auditor_metrics(cfgx):
        traj, j = run_episode(LedgerTrapEnv(mode="train", secret=JURY_SECRET), make_policy("auditor"), seed, cfgx)
        return j["metrics"]
    ma, mb = auditor_metrics(cfg0), auditor_metrics(cfg1)
    c["D10"] = (ma["tp"], ma["fp"], ma["fn"], ma["f1"]) == (mb["tp"], mb["fp"], mb["fn"], mb["f1"])
    c["D11"] = b.public_fingerprint == generate(seed, cfg1, JURY_SECRET).public_fingerprint
    code = ("import sys, json; sys.path.insert(0, %r); from ledgertrap.generator import generate; "
            "print(generate(%d, json.loads(%r), %r).public_fingerprint)" % (ROOT, seed, json.dumps(cfg0), JURY_SECRET))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "12345"})
    c["D12"] = out.stdout.strip() == a.public_fingerprint
    return c


def g_checks():
    res = {}
    res["G1 secret changes episode, same secret reproduces"] = (
        generate(5, None, "s1").files != generate(5, None, "s2").files
        and generate(5, None, "s1").files == generate(5, None, "s1").files)
    h = config_mod.config_hash
    res["G2 config_hash covers core, not presentation"] = (
        h(config_mod.load({"size": {"documents": [30, 40]}})) != h(config_mod.load(None))
        and h(config_mod.load({"presentation": {"id_salt": 9}})) == h(config_mod.load(None)))
    try:
        config_mod.load({"bogus": 1})
        res["G3 unknown config keys rejected, versions recorded"] = False
    except config_mod.ConfigError:
        ep = generate(1, None, JURY_SECRET)
        res["G3 unknown config keys rejected, versions recorded"] = (
            ep.manifest["config_version"] == "1.0" and ep.provenance["config"]["config_version"] == "1.0"
            and ep.provenance["generator_version"] == ep.manifest["generator_version"])
    grid_ok, n = True, 0
    for base in ("KZT", "USD", "EUR"):
        for tz in ("+05:00", "-05:00", "+08:00", "+00:00"):
            for chart in ("kz_standard", "ifrs_numeric", "alpha_codes"):
                for kind in ("month", "quarter"):
                    cfg = {"company": {"base_currency": base, "timezone": tz, "chart_template": chart,
                                       "period_kind": kind}, "errors": {"clean_probability": 0.1}}
                    grid_ok &= reaudit(generate(n, cfg, JURY_SECRET))["ok"]
                    n += 1
    res["G4 new parameter combinations (72) generate and re-audit"] = grid_ok
    t0 = time.time()
    env = LedgerTrapEnv(mode="eval", secret=JURY_SECRET)
    run_episode(env, make_policy("auditor"), 1, {"size": {"ledger_rows": [2000, 2000], "documents": [195, 200]},
                                                 "errors": {"count": [10, 12]}})
    res["G5 largest episode under 60 s with 3x margin"] = time.time() - t0 < 20.0
    size = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "build", "__pycache__", "_handoff")]
        size += sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
    res["G6 solution size < 1 GB"] = size < 1024 ** 3
    return res


# ---------------------------------------------------------------------------------------------

def run_all(verbose=True, quick=False):
    t0 = time.time()
    jury = jury_episodes()
    if quick:
        jury = jury[:2] + jury[5:7] + jury[10:12]
    out, details = {}, {}

    def line(title, passed, total, extra=""):
        if verbose:
            print("  %-36s passed %2d/%-2d %s" % (title, passed, total, extra))
            sys.stdout.flush()

    a = {}
    for fn in A_CHECKS:
        try:
            a[fn.__name__] = bool(fn())
        except Exception as e:  # noqa: BLE001
            a[fn.__name__] = False
            details[fn.__name__] = "%s: %s" % (type(e).__name__, e)
    out["A"] = sum(a.values())
    line("A. Contract & isolation", out["A"], 20, _failed(a))

    b_ok, b_fail = 0, Counter()
    for kind, seed, cfg in jury:
        try:
            _, c = b_checks_episode(seed, cfg)
        except Exception as e:  # noqa: BLE001
            c = {"B_error": False}
            details["B seed %d" % seed] = "%s: %s" % (type(e).__name__, e)
        b_ok += all(c.values())
        b_fail.update(k for k, v in c.items() if not v)
    out["B"] = b_ok
    line("B. Data & truth (episodes)", b_ok, len(jury), ("failing: %s" % dict(b_fail)) if b_fail else "")

    try:
        cres = c_checks(jury)
    except Exception as e:  # noqa: BLE001
        cres = {}
        details["C"] = "%s: %s" % (type(e).__name__, e)
    out["C"] = sum(cres.values())
    line("C. Reward robustness", out["C"], 20, _failed(cres))

    d_ok, d_fail = 0, Counter()
    for i, (kind, seed, cfg) in enumerate(jury):
        try:
            r = d_checks_pair(i, seed, cfg)
        except Exception as e:  # noqa: BLE001
            r = {"D_error": False}
            details["D pair %d" % i] = "%s: %s" % (type(e).__name__, e)
        d_ok += all(r.values())
        d_fail.update(k for k, v in r.items() if not v)
    out["D"] = d_ok
    line("D. Reproducibility (pairs)", d_ok, len(jury), ("failing: %s" % dict(d_fail)) if d_fail else "")

    g = g_checks()
    out["G"] = sum(g.values())
    line("G. Global reproducibility gate", out["G"], len(g), _failed(g))
    if verbose:
        for k, v in details.items():
            print("    error %s: %s" % (k, v))
    n = len(jury)
    out["all_green"] = (out["A"] == 20 and out["B"] == n and out["C"] == 20 and out["D"] == n
                        and out["G"] == len(g) and not details)
    out["seconds"] = time.time() - t0
    out["detail"] = {"A": a, "C": cres, "G": g, "B_fail": dict(b_fail), "D_fail": dict(d_fail),
                     "errors": details, "episodes": n}
    if verbose:
        print("  internal checks: %s  (%.0fs; not scored, the official blocks above are)" % (
            "all green" if out["all_green"] else "FAILURES", out["seconds"]))
    return out


def _failed(res):
    bad = [k for k, v in res.items() if not v]
    return ("failed: " + ", ".join(bad)) if bad else ""


if __name__ == "__main__":
    sys.exit(0 if run_all(quick="--quick" in sys.argv)["all_green"] else 1)
