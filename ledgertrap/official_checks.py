"""The official FinTech checks (genesis-exe.com/fintech-checks.html, published 20.11.2026), run the way the
jury connects: through judge_adapter.py only.

  generate_episode(seed, config, dir) -> folder;  make_env().reset/step in the official envelope;
  score_episode(dir, findings, inspection_steps);  jury-side transforms are made on file copies.

Block 1  C01-C20  contract and isolation on fixtures C, E, L, U      25 x passed/20
Block 2  D01-D20  closed episodes from documented scenario configs     25 x passed/20
Block 3  N01-N20  matcher and reward unit tests (+ open examples M01-M12, reported separately)
                                                                        25 x passed/20
Block 4  P01-P20  pairs: order (P01-P05), ids (P06-P10), both (P11-P20) 15 x passed/20
Engineering E1-E5 anchors 0/1/2                                         10

Seeds, secret and parameter values here are ours (the jury's are hidden); the check definitions are theirs.
"""

import contextlib
import csv
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

JURY_SECRET = "jury-simulation-secret"
CLOSED = "closed"
TOL = 1e-9
FORBIDDEN_KEYS = {"seed", "truth", "provenance", "issue_id", "evidence_sets", "expected_amount_minor",
                  "certificate_id", "is_clean", "scenario", "variant", "mutations", "secret", "secret_id",
                  "config_hash", "metrics", "tp", "fp", "fn"}
SCEN = os.path.join(ROOT, "configs", "scenarios")


def _adapter():
    import judge_adapter
    return judge_adapter


@contextlib.contextmanager
def _jury_secret():
    old = os.environ.get("LEDGERTRAP_SECRET")
    os.environ["LEDGERTRAP_SECRET"] = JURY_SECRET
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("LEDGERTRAP_SECRET", None)
        else:
            os.environ["LEDGERTRAP_SECRET"] = old


def _scenario(name, **company):
    with open(os.path.join(SCEN, name + ".json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if company:
        cfg["company"] = dict(cfg.get("company", {}), **company)
    return cfg


# ---------------------------------------------------------------------------------------------- files (jury view)

def _p(d, sub, name):
    for q in (os.path.join(d, sub, name), os.path.join(d, name)):
        if os.path.isfile(q):
            return q
    return None


def read_folder(d):
    """Independent reader of an episode folder, following the documented formats (README / case §03)."""
    with open(_p(d, "public", "ledger.csv"), encoding="utf-8", newline="") as f:
        ledger = []
        for r in csv.DictReader(f):
            r = dict(r)
            r["debit_minor"], r["credit_minor"] = int(r["debit_minor"]), int(r["credit_minor"])
            for k in ("entity_id", "invoice_id", "payment_id"):
                r[k] = r[k] or None
            ledger.append(r)
    with open(_p(d, "public", "fx.csv"), encoding="utf-8", newline="") as f:
        fx = [dict(r) for r in csv.DictReader(f)]
    out = {"ledger": ledger, "fx": fx}
    for t in ("invoices", "payments"):
        with open(_p(d, "public", t + ".jsonl"), encoding="utf-8") as f:
            out[t] = [json.loads(line) for line in f if line.strip()]
    for name in ("manifest", "policy"):
        with open(_p(d, "public", name + ".json"), encoding="utf-8") as f:
            out[name] = json.load(f)
    q = _p(d, "grader", "truth.json")
    if q:
        with open(q, encoding="utf-8") as f:
            out["truth"] = json.load(f)
    return out


def _fx_key(r):
    return "%s:%s:%s" % (r["date"], r["source_currency"], r["target_currency"])


def _oracle(truth, episode_id):
    return {"episode_id": episode_id, "findings": [
        {"type": g["type"], "affected_ids": list(g["affected_ids"]), "claimed_amount_minor": g["expected_amount_minor"],
         "currency": g["currency"], "evidence_ids": list(g["evidence_sets"][0]), "explanation": "oracle"}
        for g in truth["issues"]]}


def _walk_keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _walk_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_keys(v, out)
    return out


def _leaks(objs, d):
    """Closed data visible to the policy? Keys of truth/provenance/grader answers, issue ids, paths, secret."""
    text = json.dumps(objs, ensure_ascii=False)
    keys = _walk_keys(objs, set())
    bad = sorted(keys & FORBIDDEN_KEYS)
    for needle in ("ISS-", "CERT-", JURY_SECRET, "truth.json", "provenance.json", os.path.abspath(d), "grader/"):
        if needle in text:
            bad.append(needle)
    return bad


def _same(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


# ---------------------------------------------------------------------------------------------- env helpers

def _reset(d, team_config=None):
    env = _adapter().make_env()
    obs, info = env.reset(0, {"mode": CLOSED, "episode_dir": d, "team_config": team_config or {}})
    return env, obs, info


def _do(env, name, **args):
    return env.step({"name": name, "args": args})


def _page_all(d, table, limit=50):
    """Collect a whole table through list_records (new reset when the budget runs out)."""
    got, pages, cursor = [], [], None
    env, obs, _ = _reset(d)
    while True:
        o = _do(env, "list_records", table=table, cursor=cursor, limit=limit)[0]
        if o["error"] is not None:
            return None, pages
        got.extend(o["result"]["records"])
        pages.append(len(o["result"]["records"]))
        cursor = o["result"]["next_cursor"]
        if cursor is None:
            return got, pages
        if o["remaining_steps"] <= 1:  # continue from the same cursor in a fresh episode: cursors are stable
            env, obs, _ = _reset(d)


# ============================================================================================== Block 1

class Fixtures:
    def __init__(self, work):
        a = _adapter()
        self.C = os.path.join(work, "fixture_C")
        self.E = os.path.join(work, "fixture_E")
        with _jury_secret():
            a.generate_episode(9201, _scenario("clean_basic"), self.C)
            a.generate_episode(9202, _scenario("duplicate_document"), self.E)
        self.L = self.C  # ledger has >= 200 rows: three pages of 50 and more
        self.fc, self.fe = read_folder(self.C), read_folder(self.E)
        self.R1 = self.fc["ledger"][0]["row_id"]
        self.D1 = self.fc["invoices"][0]["doc_id"]
        parts = []
        for dp, _, fs in os.walk(work):
            for name in fs:
                with open(os.path.join(dp, name), encoding="utf-8") as fh:
                    parts.append(fh.read())
        blob = "".join(parts)
        u = "U-%d" % 0
        i = 0
        while u in blob:  # chosen after reading the data, as in the definitions
            i += 1
            u = "U-%d" % i
        self.U = u


def block1(fx):
    res = {}
    C, E, L, U, R1, D1 = fx.C, fx.E, fx.L, fx.U, fx.R1, fx.D1

    env = _adapter().make_env()
    out = env.reset(0, {"mode": CLOSED, "episode_dir": C, "team_config": {}})
    step = _do(env, "list_records", table="ledger", cursor=None, limit=3)
    res["C01"] = isinstance(out, tuple) and len(out) == 2 and bool(json.dumps(out)) and bool(json.dumps(step))

    obs, info = out
    res["C02"] = (set(obs) == {"task", "manifest", "policy", "remaining_steps"} and isinstance(obs["task"], str)
                  and obs["remaining_steps"] == 100 and not _leaks([obs, info], C) and info == {})

    m = obs["manifest"]
    need = ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts",
            "rounding")
    f = fx.fc
    res["C03"] = (all(k in m for k in need) and m["counts"]["ledger"] == len(f["ledger"])
                  and m["counts"]["invoices"] == len(f["invoices"]) and m["counts"]["payments"] == len(f["payments"])
                  and m["counts"]["fx"] == len(f["fx"])
                  and m["counts"]["documents"] == len(f["invoices"]) + len(f["payments"])
                  and m["rounding"] == f["policy"]["rounding"])

    env, obs, _ = _reset(C)
    o = _do(env, "list_records", table="ledger", cursor=None, limit=50)
    res["C04"] = (isinstance(o, tuple) and len(o) == 5 and type(o[2]) is bool and type(o[3]) is bool
                  and bool(json.dumps(o)) and isinstance(o[0]["result"]["records"], list)
                  and o[0]["remaining_steps"] == 99 and o[0]["error"] is None)

    ok5 = True
    fl = read_folder(L)
    for t in ("ledger", "invoices", "payments", "fx"):
        got, pages = _page_all(L, t, 50)
        ok5 &= got is not None and _same(got, fl[t]) and all(n <= 50 for n in pages)
    res["C05"] = ok5 and len(fl["ledger"]) >= 101

    env, obs, _ = _reset(L)
    first = _do(env, "list_records", table="ledger", cursor=None, limit=50)[0]
    cur = first["result"]["next_cursor"]
    p1 = _do(env, "list_records", table="ledger", cursor=cur, limit=50)[0]
    p2 = _do(env, "list_records", table="ledger", cursor=cur, limit=50)[0]
    res["C06"] = (p1["result"] == p2["result"] and len(p1["result"]["records"]) == 50
                  and p1["remaining_steps"] - 1 == p2["remaining_steps"])

    env, obs, _ = _reset(C)
    o = _do(env, "get_record", table="ledger", id=R1)[0]
    res["C07"] = o["error"] is None and _same(o["result"], fx.fc["ledger"][0]) and not _leaks(o, C)

    o = _do(env, "get_document", doc_id=D1)[0]
    res["C08"] = o["error"] is None and _same(o["result"], fx.fc["invoices"][0]) and not _leaks(o, C)

    o = _do(env, "calculator", expression="0.1+0.2")[0]
    res["C09"] = o["error"] is None and isinstance(o["result"], str) and Decimal(o["result"]) == Decimal("0.3")

    ok10 = True
    for expr in ("open('/etc/passwd').read()", "__import__('os').system('echo pwned')", "1+`cat /etc/passwd`",
                 "../grader/truth.json", "2**10", "eval('1+1')", "1;import os"):
        o = _do(env, "calculator", expression=expr)[0]
        ok10 &= o["result"] is None and o["error"] is not None and "root:" not in json.dumps(o)
    res["C10"] = ok10

    env, obs, _ = _reset(C)
    o1 = _do(env, "get_record", table="ledger", id=U)
    o2 = _do(env, "get_record", table="ledger", id=R1)[0]
    res["C11"] = (o1[0]["error"] is not None and o1[0]["result"] is None and o1[0]["remaining_steps"] == 99
                  and o2["error"] is None and not o1[2] and not o1[3] and U not in json.dumps(o1[0]))

    o = _do(env, "get_document", doc_id=U)
    docs = {d["doc_id"] for d in fx.fc["invoices"] + fx.fc["payments"]}
    res["C12"] = (o[0]["error"] is not None and o[0]["result"] is None and o[0]["remaining_steps"] == 97
                  and not any(x in json.dumps(o[0]) for x in docs))

    o = env.step({"name": "delete_everything", "args": {}})
    res["C13"] = o[0]["error"] is not None and o[0]["remaining_steps"] == 96 and not o[2] and not o[3]

    env, obs, _ = _reset(C)
    a = _do(env, "get_record", table="ledger", id=R1)[0]
    b = _do(env, "get_record", table="ledger", id=U)[0]
    c = _do(env, "get_record", table="ledger", id=R1)[0]
    res["C14"] = ([a["remaining_steps"], b["remaining_steps"], c["remaining_steps"]] == [99, 98, 97]
                  and c["error"] is None and _same(a["result"], c["result"]))

    env, obs, _ = _reset(E)
    oracle = _oracle(fx.fe["truth"], fx.fe["manifest"]["episode_id"])
    _do(env, "list_records", table="ledger", cursor=None, limit=5)
    end = _do(env, "submit", findings=oracle)
    j0 = json.dumps(env.journal, sort_keys=True)
    after = [_do(env, "submit", findings={"episode_id": oracle["episode_id"], "findings": []}),
             _do(env, "get_record", table="ledger", id=fx.fe["ledger"][0]["row_id"])]
    res["C15"] = (end[2] is True and end[0]["result"] == {"accepted": True}
                  and json.dumps(env.journal, sort_keys=True) == j0
                  and all(x[0]["error"] is not None and x[1] == 0 and x[4] == {} for x in after)
                  and env.journal[-1]["metrics"]["tp"] == len(fx.fe["truth"]["issues"]))

    env, obs, _ = _reset(C)
    flags = [_do(env, "calculator", expression="1+1")[2:4] for _ in range(99)]
    end = _do(env, "submit", findings={"episode_id": fx.fc["manifest"]["episode_id"], "findings": []})
    res["C16"] = (all(f == (False, False) for f in flags) and end[2] is True and end[3] is False
                  and end[0]["result"] == {"accepted": True} and env.journal[-1]["metrics"]["inspection_steps"] == 99
                  and env.journal[-1]["reason"] == "submitted")

    env, obs, _ = _reset(E)
    outs = [_do(env, "calculator", expression="1+1") for _ in range(100)]
    j = dict(env.journal[-1])
    late = _do(env, "submit", findings=oracle)
    res["C17"] = (all(not o[2] and not o[3] for o in outs[:99]) and outs[99][3] is True and outs[99][2] is False
                  and j["reason"] == "max_steps" and j["metrics"]["tp"] == 0 and j["metrics"]["fp"] == 0
                  and len(env.journal) == 1 and late[0]["error"] is not None)

    class Clock:
        t = 1000.0

        def __call__(self):
            return self.t
    clock = Clock()
    env = _adapter().make_env()
    env._env._clock = clock  # reproducible 60 s scenario without waiting (the policy "thinks" 61 s)
    env.reset(0, {"mode": CLOSED, "episode_dir": E, "team_config": {}})
    _do(env, "list_records", table="ledger", cursor=None, limit=5)
    clock.t += 61
    late = _do(env, "submit", findings=oracle)
    ok18 = (late[3] is True and late[2] is False and env.journal[-1]["reason"] == "time_limit"
            and env.journal[-1]["metrics"]["tp"] == 0 and env.journal[-1]["metrics"]["fp"] == 0 and late[1] == 0)
    env, obs, _ = _reset(E, {"limits": {"time_limit_s": 1}})  # the same path with the real clock
    time.sleep(1.05)
    late = _do(env, "submit", findings=oracle)
    res["C18"] = ok18 and late[3] is True and env.journal[-1]["reason"] == "time_limit"

    env, obs, info = _reset(E)
    steps = [_do(env, "list_records", table="payments", cursor=None, limit=10),
             _do(env, "get_document", doc_id=fx.fe["invoices"][0]["doc_id"]),
             _do(env, "calculator", expression="7*6")]
    end = _do(env, "submit", findings=oracle)
    true = _adapter().score_episode(E, oracle, 3)
    res["C19"] = (all(type(s[1]) in (int, float) and s[1] == 0 and s[4] == {} for s in steps + [end])
                  and info == {} and end[0] == {"result": {"accepted": True}, "error": None,
                                                "remaining_steps": end[0]["remaining_steps"]}
                  and abs(true["return"] - (1.0 - 0.002 * 3)) < TOL
                  and abs(env.journal[-1]["metrics"]["reward_main"] - true["return"]) < TOL)

    ok20 = True
    for d, f in ((C, fx.fc), (E, fx.fe)):
        seen = []
        for t in ("ledger", "invoices", "payments", "fx"):
            got, _ = _page_all(d, t)
            seen.append(got)
            ok20 &= got is not None
        env, obs, info = _reset(d)
        seen += [obs, info]
        for doc in f["invoices"] + f["payments"]:
            o = _do(env, "get_document", doc_id=doc["doc_id"])
            seen.append(o)
            if o[0]["remaining_steps"] <= 1:
                env, obs, info = _reset(d)
        for probe in ({"name": "get_record", "args": {"table": "truth", "id": "ISS-01"}},
                      {"name": "get_record", "args": {"table": "../grader/truth", "id": "x"}},
                      {"name": "get_document", "args": {"doc_id": "../grader/truth.json"}},
                      {"name": "list_records", "args": {"table": "provenance"}},
                      {"name": "shell", "args": {"cmd": "ls"}}, {"name": "sql", "args": {"q": "select 1"}},
                      {"name": "http_get", "args": {"url": "http://example.com"}}):
            o = env.step(probe)
            seen.append(o)
            ok20 &= o[0]["error"] is not None and o[0]["result"] is None
        ok20 &= not _leaks(seen, d)
    res["C20"] = ok20
    return res


# ============================================================================================== Block 2

MATRIX = [
    ("D01", "clean_basic", {}, "clean", None),
    ("D02", "clean_split_payment", {}, "clean", "split"),
    ("D03", "clean_refund", {}, "clean", "refund"),
    ("D04", "clean_repeated_amounts", {}, "clean", "repeated"),
    ("D05", "clean_fx_period_boundary", {}, "clean", "boundary"),
    ("D06", "duplicate_document", {}, [("duplicate_invoice", "document")], None),
    ("D07", "duplicate_double_posting", {}, [("duplicate_invoice", "double_posting")], None),
    ("D08", "duplicate_document", {"base_currency": "USD"}, [("duplicate_invoice", "document")], None),
    ("D09", "allocation_missing", {}, [("payment_allocation", "missing")], None),
    ("D10", "allocation_partial", {}, [("payment_allocation", "partial")], None),
    ("D11", "allocation_partial", {"n_foreign_currencies": [2, 3]}, [("payment_allocation", "partial")], None),
    ("D12", "fx_wrong_date", {}, [("fx_conversion", "wrong_date")], None),
    ("D13", "fx_wrong_rate", {"base_currency": "EUR"}, [("fx_conversion", "wrong_rate")], None),
    ("D14", "cutoff_early", {}, [("period_cutoff", "early")], None),
    ("D15", "cutoff_late", {"timezone": "-05:00"}, [("period_cutoff", "late")], None),
    ("D16", "c1_boundary_fx_partial", {"base_currency": "KZT", "timezone": "+05:00"}, "C1", None),
    ("D17", "c1_boundary_fx_partial", {"base_currency": "USD", "timezone": "-05:00"}, "C1", None),
    ("D18", "c1_boundary_fx_partial", {"base_currency": "EUR", "timezone": "+08:00"}, "C1", None),
    ("D19", "c2_duplicate_wrong_invoice", {"base_currency": "EUR"}, "C2", None),
    ("D20", "c2_duplicate_wrong_invoice", {"base_currency": "KZT", "timezone": "+00:00"}, "C2", None),
]


def _distractor_present(f, kind, traps):
    inv, pay = f["invoices"], f["payments"]
    if kind == "split":
        refs = {}
        for p in pay:
            for r in p["invoice_refs"] or []:
                refs[r["invoice_id"]] = refs.get(r["invoice_id"], 0) + 1
        return "L3" in traps and max(refs.values() or [0]) >= 2
    if kind == "refund":
        return "L4" in traps and any(d["kind"] == "credit_note" for d in inv) and any(p["amount_minor"] < 0 for p in pay)
    if kind == "repeated":
        totals = {}
        for d in inv:
            totals[(d["currency"], d["total_minor"])] = totals.get((d["currency"], d["total_minor"]), 0) + 1
        return bool({"L1", "L2"} & set(traps)) and max(totals.values()) >= 2
    if kind == "boundary":
        return bool({"L5", "L6"} & set(traps))
    return True


def block2(work):
    from .episode_io import load_episode_dir
    from .jury_checks import b_checks
    from .schemas import validate_episode
    a = _adapter()
    res, why, dirs = {}, {}, {}
    for i, (cid, scen, company, expect, distractor) in enumerate(MATRIX):
        d = os.path.join(work, "block2", cid)
        try:
            with _jury_secret():
                a.generate_episode(9300 + i, _scenario(scen, **company), d)
            ep = load_episode_dir(d)
            c = b_checks(ep)
            c["formats"] = validate_episode(ep) == []
            issues = ep.truth["issues"]
            if expect == "clean":
                c["expected_issues"] = issues == []
                c["distractor"] = _distractor_present(read_folder(d), distractor, ep.provenance["traps_planted"])
            elif expect in ("C1", "C2"):
                c["expected_issues"] = sorted((g["type"], g["variant"]) for g in issues if g["scenario"] == expect) == (
                    [("fx_conversion", "wrong_date"), ("period_cutoff", "early")] if expect == "C1" else
                    [("duplicate_invoice", "document"), ("payment_allocation", "wrong_invoice")])
            else:
                c["expected_issues"] = sorted((g["type"], g["variant"]) for g in issues) == sorted(expect)
            res[cid] = all(c.values())
            why[cid] = [k for k, v in c.items() if not v]
            dirs[cid] = d
        except Exception as e:  # noqa: BLE001
            res[cid] = False
            why[cid] = ["%s: %s" % (type(e).__name__, e)]
    return res, why, dirs


# ============================================================================================== Block 3

def _fixture3(base, work, name, issues, tol=None):
    d = os.path.join(work, "block3", name)
    shutil.copytree(base, d)
    f = read_folder(d)
    with open(_p(d, "grader", "truth.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema_version": f["manifest"]["schema_version"], "episode_id": f["manifest"]["episode_id"],
                   "is_clean": not issues, "issues": issues}, fh, ensure_ascii=False)
    if tol is not None:
        pol = f["policy"]
        pol["submission"]["amount_tolerance_minor"] = tol
        with open(_p(d, "public", "policy.json"), "w", encoding="utf-8") as fh:
            json.dump(pol, fh, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
    return d


def _close(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= TOL


def _expect(out, tp, fp, fn, p, r, f1, ret):
    return (out["tp"] == tp and out["fp"] == fp and out["fn"] == fn and _close(out["precision"], p)
            and _close(out["recall"], r) and _close(out["f1"], f1) and _close(out["return"], ret))


def _invalid(out):
    return out["return"] == -1 and all(out[k] is None for k in ("tp", "fp", "fn", "precision", "recall", "f1"))


def block3(fx, work):
    a = _adapter()
    f = fx.fc
    eid = f["manifest"]["episode_id"]
    R = [d["doc_id"] for d in f["invoices"][:5]]
    D1, D2, D3 = (r["row_id"] for r in f["ledger"][:3])
    many = [r["row_id"] for r in f["ledger"][:11]]
    T, T2 = "duplicate_invoice", "period_cutoff"
    A = {"issue_id": "A", "type": T, "affected_ids": [R[0]], "expected_amount_minor": 10000, "currency": "KZT",
         "evidence_sets": [[D1, D2]]}
    B = {"issue_id": "B", "type": "fx_conversion", "affected_ids": [R[1]], "expected_amount_minor": 20000,
         "currency": "KZT", "evidence_sets": [[D3]]}
    F = {"type": T, "affected_ids": [R[0]], "claimed_amount_minor": 10000, "currency": "KZT",
         "evidence_ids": [D1, D2], "explanation": "short"}
    FB = {"type": "fx_conversion", "affected_ids": [R[1]], "claimed_amount_minor": 20000, "currency": "KZT",
          "evidence_ids": [D3], "explanation": "b"}
    fa = _fixture3(fx.C, work, "A", [A])
    f0 = _fixture3(fx.C, work, "empty", [])
    fab = _fixture3(fx.C, work, "AB", [A, B])

    def s(d, fs, steps=10):
        return a.score_episode(d, {"episode_id": eid, "findings": fs}, steps)
    res = {}
    res["N01"] = _expect(s(fa, [F]), 1, 0, 0, 1, 1, 1, 0.98)
    res["N02"] = _expect(s(fa, []), 0, 0, 1, 0, 0, 0, -1)
    res["N03"] = _expect(s(f0, []), 0, 0, 0, 0, 0, 1, 0.98)
    res["N04"] = _expect(s(f0, [dict(F, affected_ids=[R[2]])]), 0, 1, 0, 0, 0, 0, -1)
    res["N05"] = _expect(s(fa, [F, F]), 1, 1, 0, 0.5, 1, 2 / 3, 47 / 150)
    res["N06"] = _expect(s(fa, [dict(F, type=T2)]), 0, 1, 1, 0, 0, 0, -1)
    res["N07"] = _expect(s(fa, [dict(F, affected_ids=[R[3]])]), 0, 1, 1, 0, 0, 0, -1)
    res["N08"] = _expect(s(fa, [dict(F, claimed_amount_minor=10001)]), 0, 1, 1, 0, 0, 0, -1)
    res["N09"] = _expect(s(fa, [dict(F, currency="USD")]), 0, 1, 1, 0, 0, 0, -1)
    res["N10"] = _expect(s(fa, [dict(F, evidence_ids=[D1])]), 0, 1, 1, 0, 0, 0, -1)
    res["N11"] = _expect(s(fa, [dict(F, evidence_ids=[D1, fx.U])]), 0, 1, 1, 0, 0, 0, -1)
    flood = [dict(F, type=t, affected_ids=[r]) for r in R for t in
             ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")]
    res["N12"] = _expect(s(fa, flood), 1, 19, 0, 1 / 20, 1, 2 / 21, -871 / 1050)
    res["N13"] = _expect(s(fab, [F]), 1, 0, 1, 1, 0.5, 2 / 3, 47 / 150)
    res["N14"] = _expect(s(fab, [F, dict(F, type=T2, affected_ids=[R[2]])]), 1, 1, 1, 0.5, 0.5, 0.5, -0.02)
    res["N15"] = _expect(s(fa, [F], 0), 1, 0, 0, 1, 1, 1, 1.0)
    res["N16"] = _expect(s(fa, [F], 50), 1, 0, 0, 1, 1, 1, 0.90)
    res["N17"] = _expect(s(fa, [F], 100), 1, 0, 0, 1, 1, 1, 0.80)
    res["N18"] = _invalid(s(fa, [F] * 31))
    res["N19"] = _invalid(s(fa, [dict(F, evidence_ids=many)]))
    res["N20"] = _invalid(s(fa, [dict(F, explanation="a" * 1001)]))

    # Open examples M01-M12 (one issue A with affected {R1, R2}; D3 extra existing evidence; U absent).
    A2 = dict(A, affected_ids=[R[0], R[1]])
    F2 = dict(F, affected_ids=[R[0], R[1]])
    fm = _fixture3(fx.C, work, "M", [A2])
    fm8 = _fixture3(fx.C, work, "M08", [A2], tol={T: 1})
    m = {}
    m["M01"] = _expect(s(fm, [dict(F2, affected_ids=[R[1], R[0], R[0]])]), 1, 0, 0, 1, 1, 1, 0.98)
    m["M02"] = _expect(s(fm, [dict(F2, affected_ids=[R[0], R[1], R[2]])]), 0, 1, 1, 0, 0, 0, -1)
    m["M03"] = _expect(s(fm, [dict(F2, evidence_ids=[D2, D1, D3])]), 1, 0, 0, 1, 1, 1, 0.98)
    m["M04"] = _expect(s(fm, [dict(F2, evidence_ids=[D1, D2, fx.U])]), 0, 1, 1, 0, 0, 0, -1)
    m["M05"] = _invalid(s(fm, [dict(F2, evidence_ids=[D1] * 10 + [D2])]))
    m["M06"] = all(_expect(s(fm, [dict(F2, currency=c)]), 0, 1, 1, 0, 0, 0, -1) for c in ("kzt", " KZT", "KZT "))
    m["M07"] = _expect(s(fm, [dict(F2, claimed_amount_minor=10001)]), 0, 1, 1, 0, 0, 0, -1)
    m["M08"] = (_expect(s(fm8, [dict(F2, claimed_amount_minor=10001)]), 1, 0, 0, 1, 1, 1, 0.98)
                and _expect(s(fm8, [dict(F2, claimed_amount_minor=9999)]), 1, 0, 0, 1, 1, 1, 0.98)
                and _expect(s(fm8, [dict(F2, claimed_amount_minor=10002)]), 0, 1, 1, 0, 0, 0, -1))
    m["M09"] = _invalid(s(fm, [dict(F2, claimed_amount_minor="10000")])) and _invalid(
        s(fm, [dict(F2, claimed_amount_minor=True)]))
    Am, Bm = dict(A2, issue_id="A"), dict(A2, issue_id="B", evidence_sets=[[D3]])
    f10 = _fixture3(fx.C, work, "M10", [Bm, Am])
    out = s(f10, [dict(F2, evidence_ids=[D1, D2, D3]), F2])
    pairs = sorted(a.JOURNAL[-1]["pairs"], key=lambda p: p[1])
    m["M10"] = _expect(out, 2, 0, 0, 1, 1, 1, 0.98) and [list(p) for p in pairs] == [[1, "A"], [0, "B"]]
    f11 = _fixture3(fx.C, work, "M11", [dict(A2, issue_id="B"), dict(A2, issue_id="A")])
    out = s(f11, [F2, dict(F2, explanation="other")])
    pairs = sorted(a.JOURNAL[-1]["pairs"], key=lambda p: p[1])
    out2 = s(f11, [dict(F2, explanation="other"), F2])
    m["M11"] = (_expect(out, 2, 0, 0, 1, 1, 1, 0.98) and [list(p) for p in pairs] == [[0, "A"], [1, "B"]]
                and out2 == out)
    m["M12"] = _expect(s(fm, [F2, F2]), 1, 1, 0, 0.5, 1, 2 / 3, 47 / 150)
    return res, m


# ============================================================================================== Block 4

ID_SPACES = {"row": "row_%04d", "doc": "doc-%04d", "inv": "inv/%04d", "pay": "pay#%04d", "ent": "ent:%04d"}


def transform(src, dst, permute, remap, seed):
    """Jury-style transform of a COPY: permute table records and/or replace ids consistently (all spaces,
    every reference, truth and provenance). line_items order is never touched; fx keys stay."""
    rng = random.Random(seed)
    f = read_folder(src)
    shutil.copytree(src, dst)
    spaces = {k: [] for k in ID_SPACES}
    spaces["row"] = [r["row_id"] for r in f["ledger"]]
    spaces["doc"] = [d["doc_id"] for d in f["invoices"] + f["payments"]]
    spaces["inv"] = [d["invoice_id"] for d in f["invoices"]]
    spaces["pay"] = [d["payment_id"] for d in f["payments"]]
    spaces["ent"] = sorted({d["entity_id"] for d in f["invoices"] + f["payments"]} |
                           {r["entity_id"] for r in f["ledger"] if r["entity_id"]})
    mapping = {}
    if remap:
        for sp, ids in spaces.items():
            nums = list(range(len(ids)))
            rng.shuffle(nums)
            for old, n in zip(ids, nums):
                mapping[old] = ID_SPACES[sp] % n

    def mp(x):
        return mapping.get(x, x) if isinstance(x, str) else x

    def deep(obj):
        if isinstance(obj, dict):
            return {k: deep(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep(v) for v in obj]
        return mp(obj)

    ledger = [dict(r, row_id=mp(r["row_id"]), entity_id=mp(r["entity_id"]), invoice_id=mp(r["invoice_id"]),
                   payment_id=mp(r["payment_id"])) for r in f["ledger"]]
    invoices = [dict(d, doc_id=mp(d["doc_id"]), invoice_id=mp(d["invoice_id"]), entity_id=mp(d["entity_id"]),
                     original_invoice_id=mp(d["original_invoice_id"])) for d in f["invoices"]]
    payments = [dict(d, doc_id=mp(d["doc_id"]), payment_id=mp(d["payment_id"]), entity_id=mp(d["entity_id"]),
                     invoice_refs=None if d["invoice_refs"] is None else
                     [dict(x, invoice_id=mp(x["invoice_id"])) for x in d["invoice_refs"]]) for d in f["payments"]]
    fxr = list(f["fx"])
    if permute:
        for t in (ledger, invoices, payments, fxr):
            rng.shuffle(t)

    def w(name, data):
        with open(_p(dst, "public" if not name.startswith("grader:") else "grader", name.split(":")[-1]), "w",
                  encoding="utf-8", newline="") as fh:
            fh.write(data)
    cols = list(f["ledger"][0].keys())
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(cols)
    for r in ledger:
        wr.writerow(["" if r[c] is None else r[c] for c in cols])
    w("ledger.csv", buf.getvalue())
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["date", "source_currency", "target_currency", "rate"])
    for r in fxr:
        wr.writerow([r["date"], r["source_currency"], r["target_currency"], r["rate"]])
    w("fx.csv", buf.getvalue())
    w("invoices.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in invoices))
    w("payments.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in payments))
    for name in ("truth.json", "provenance.json", "certificates.json"):
        q = _p(dst, "grader", name)
        if q:
            with open(q, encoding="utf-8") as fh:
                obj = json.load(fh)
            with open(q, "w", encoding="utf-8") as fh:
                json.dump(deep(obj), fh, ensure_ascii=False)
    return mapping


def _script(d, f):
    """Fixed exploration actions built from the files of `d` (3 inspection steps) + oracle submit."""
    return [{"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}},
            {"name": "get_record", "args": {"table": "ledger", "id": f["ledger"][0]["row_id"]}},
            {"name": "get_document", "args": {"doc_id": f["invoices"][0]["doc_id"]}},
            {"name": "submit", "args": {"findings": _oracle(f["truth"], f["manifest"]["episode_id"])}}]


def _run(env_dir, actions, env=None):
    if env is None:
        env = _adapter().make_env()
    out = [env.reset(0, {"mode": CLOSED, "episode_dir": env_dir, "team_config": {}})]
    out += [env.step(act) for act in actions]
    return json.loads(json.dumps(out)), env


def pair_check(i, src, work):
    kind = "order" if i < 5 else ("ids" if i < 10 else "both")
    dst = os.path.join(work, "block4", "P%02d" % (i + 1))
    mapping = transform(src, dst, permute=kind in ("order", "both"), remap=kind in ("ids", "both"), seed=500 + i)
    inv = {v: k for k, v in mapping.items()}
    fs, fd = read_folder(src), read_folder(dst)
    c = {}
    # 1. exact replay on the original, and reset does not keep state
    run1, env = _run(src, _script(src, fs))
    run2, _ = _run(src, _script(src, fs))
    run3, _ = _run(src, _script(src, fs), env)
    c["replay"] = run1 == run2 == run3
    # 2. oracle with new ids on the variant; after the inverse mapping the answers are the same
    oa, ob = _oracle(fs["truth"], fs["manifest"]["episode_id"]), _oracle(fd["truth"], fd["manifest"]["episode_id"])

    def back(o):
        return sorted(json.dumps({"type": x["type"], "affected": sorted(inv.get(y, y) for y in x["affected_ids"]),
                                  "amount": x["claimed_amount_minor"], "currency": x["currency"],
                                  "evidence": sorted(inv.get(y, y) for y in x["evidence_ids"])}, sort_keys=True)
                      for x in o["findings"])
    truth_ids = {x for g in fd["truth"]["issues"] for x in g["affected_ids"] + sum(g["evidence_sets"], [])}
    c["oracle_back_mapped"] = back(ob) == back(oa) and not (set(mapping) & truth_ids)
    # 3. TP/FP/FN/P/R/F1 and terminal reward equal with the same inspection steps (via score_episode and the env)
    a = _adapter()
    sa, sb = a.score_episode(src, oa, 3), a.score_episode(dst, ob, 3)
    rv, envb = _run(dst, _script(dst, fd))
    ja, jb = env.journal[-1]["metrics"], envb.journal[-1]["metrics"]
    keys = ("tp", "fp", "fn", "precision", "recall", "f1", "reward_main", "inspection_steps")
    c["metrics_equal"] = sa == sb and all(ja[k] == jb[k] for k in keys) and sa["return"] == ja["reward_main"]
    # 4. no new leak, every evidence id still reachable through the tools, records served as in the files
    envc = a.make_env()
    envc.reset(0, {"mode": CLOSED, "episode_dir": dst, "team_config": {}})
    docs = {d["doc_id"] for d in fd["invoices"] + fd["payments"]}
    rows = {r["row_id"] for r in fd["ledger"]}
    reach = True
    for g in fd["truth"]["issues"]:
        for x in g["evidence_sets"][0]:
            if x in docs:
                o = _do(envc, "get_document", doc_id=x)[0]
            elif x in rows:
                o = _do(envc, "get_record", table="ledger", id=x)[0]
            else:
                o = _do(envc, "get_record", table="fx", id=x)[0]
            reach &= o["error"] is None
    page = _do(envc, "list_records", table="ledger", cursor=None, limit=50)[0]["result"]["records"]
    c["evidence_reachable"] = reach
    c["served_as_files"] = _same(page, fd["ledger"][:50]) and (kind == "ids" or page != fs["ledger"][:50]
                                                               or len(fs["ledger"]) < 2)
    c["no_leak"] = not _leaks(rv, dst)
    return all(c.values()), [k for k, v in c.items() if not v]


def block4(dirs, work):
    res, why = {}, {}
    for i in range(20):
        cid = "D%02d" % (i + 1)
        pid = "P%02d" % (i + 1)
        if cid not in dirs:
            res[pid], why[pid] = False, ["no source episode %s" % cid]
            continue
        try:
            res[pid], why[pid] = pair_check(i, dirs[cid], work)
        except Exception as e:  # noqa: BLE001
            res[pid], why[pid] = False, ["%s: %s" % (type(e).__name__, e)]
    return res, why


# ============================================================================================== Engineering

def engineering():
    def read(p):
        try:
            with open(os.path.join(ROOT, p), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""
    readme, docker, report = read("README.md"), read("Dockerfile"), read("reports/diagnostics.md")
    e = {}
    one = "python -m ledgertrap check" in readme and "ledgertrap" in docker and "check" in docker
    e["E1 one command"] = 2 if one else (1 if os.path.exists(os.path.join(ROOT, "ledgertrap", "cli.py")) else 0)
    versions = read("ledgertrap/version.py")
    cfg_ok = all('"config_version"' in read(p) for p in ("configs/train.json", "configs/dev.json"))
    e["E2 config versions"] = 2 if ("GENERATOR_VERSION" in versions and cfg_ok and "splits_version" in
                                    read("configs/splits.json") and "requires-python" in read("pyproject.toml")) else 1
    traj_ok = False
    try:
        with open(os.path.join(ROOT, "trajectories.jsonl"), encoding="utf-8") as f:
            lines = [json.loads(next(f)) for _ in range(50)]
        need = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
        traj_ok = all(need <= set(x) for x in lines)
    except (OSError, StopIteration, ValueError):
        pass
    e["E3 trajectory logs"] = 2 if traj_ok else (1 if os.path.exists(os.path.join(ROOT, "trajectories.jsonl")) else 0)
    measured = "Разбор сбоев" in report and "Причина" in report and "Повторная проверка" in report
    e["E4 measured failure analysis"] = 2 if measured else (1 if report else 0)
    lim = ("Limitations" in readme or "Ограничения" in readme)
    e["E5 honest limitations"] = 2 if lim and "python -m ledgertrap" in readme else (
        1 if "Ограничения" in read("docs/DESIGN.md") or "ограничения" in read("docs/STATUS.md") else 0)
    return e


# ============================================================================================== runner

def run_official(verbose=True, keep=False):
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="ledgertrap-official-")
    out, why = {}, {}

    def line(title, passed, total, extra=""):
        if verbose:
            print("  %-44s passed %2d/%-2d %s" % (title, passed, total, extra))
            sys.stdout.flush()
    try:
        fx = Fixtures(work)
        b1 = block1(fx)
        out["block1"] = sum(b1.values())
        line("Block 1. Contract tests C01-C20 (25)", out["block1"], 20, _failed(b1))
        b2, why2, dirs = block2(work)
        out["block2"] = sum(b2.values())
        line("Block 2. Closed episodes D01-D20 (25)", out["block2"], 20,
             "; ".join("%s: %s" % (k, ",".join(v)) for k, v in why2.items() if v))
        b3, mm = block3(fx, work)
        out["block3"] = sum(b3.values())
        line("Block 3. Matcher & reward N01-N20 (25)", out["block3"], 20, _failed(b3))
        out["examples"] = sum(mm.values())
        line("       open examples M01-M12", out["examples"], 12, _failed(mm))
        b4, why4 = block4(dirs, work)
        out["block4"] = sum(b4.values())
        line("Block 4. Pairs P01-P20 (15)", out["block4"], 20,
             "; ".join("%s: %s" % (k, ",".join(v)) for k, v in why4.items() if v))
        e = engineering()
        out["engineering"] = sum(e.values())
        if verbose:
            print("  %-44s %d/10   %s" % ("Engineering E1-E5 (0/1/2)", out["engineering"],
                                          ", ".join("%s=%d" % (k.split(" ")[0], v) for k, v in e.items())))
        out["points"] = (25 * out["block1"] / 20 + 25 * out["block2"] / 20 + 25 * out["block3"] / 20
                         + 15 * out["block4"] / 20 + out["engineering"])
        out["all_green"] = (out["block1"] == 20 and out["block2"] == 20 and out["block3"] == 20
                            and out["examples"] == 12 and out["block4"] == 20)
        out["detail"] = {"block1": b1, "block2": b2, "block2_why": why2, "block3": b3, "examples": mm,
                         "block4": b4, "block4_why": why4, "engineering": e}
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    out["seconds"] = time.time() - t0
    if verbose:
        print("  official-definitions score (our run): %.1f / 100  (%.0fs)" % (out["points"], out["seconds"]))
    return out


def _failed(res):
    bad = [k for k, v in res.items() if not v]
    return ("failed: " + ", ".join(bad)) if bad else ""


if __name__ == "__main__":
    sys.exit(0 if run_official()["all_green"] else 1)
