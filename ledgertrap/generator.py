"""Episode generation pipeline (DESIGN §9.1) and export."""

import csv
import io
import os

from . import invariants
from .canon import canonical_bytes, canonical_json, hmac_sha256, sha256_hex
from .config import config_hash, load
from .errors import inject_errors
from .policy import build_policy
from .presentation import assign_ids, order_keys
from .rng import DetRng
from .story import build_story
from .timeutil import fmt
from .truth import detect_traps, public_issues
from .version import DEFAULT_SECRET, GENERATOR_VERSION, SCHEMA_VERSION
from .world import World, build_company, build_customers, build_fx, build_period

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id",
                  "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
PUBLIC_FILES = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"]


class GeneratorInvariantError(RuntimeError):
    pass


def secret_id(secret: str) -> str:
    return sha256_hex(secret.encode("utf-8"))[:12]


def _validate_seed(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed < 2**63):
        raise ValueError("seed must be an int in [0, 2^63)")


class Episode:
    """Generated episode: public records/files and grader-only truth/provenance."""

    def __init__(self):
        self.seed = None
        self.cfg = None
        self.secret_id = None
        self.config_hash = None
        self.episode_id = None
        self.manifest = None
        self.policy = None
        self.records = None       # tables: ledger, invoices, payments, fx (public, ordered)
        self.files = None         # name -> bytes (public)
        self.truth = None
        self.provenance = None
        self.certificates = None
        self.aliases = None       # public doc_id / invoice_id / payment_id -> doc_id
        self.public_fingerprint = None
        self.id_map = None        # internal key -> public ids (for idmap in later stages)

    def grader_files(self):
        return {
            "truth.json": canonical_bytes(self.truth) + b"\n",
            "provenance.json": canonical_bytes(self.provenance) + b"\n",
            "certificates.json": canonical_bytes(self.certificates) + b"\n",
        }

    def write(self, out_dir):
        for sub, files in (("public", self.files), ("grader", self.grader_files())):
            d = os.path.join(out_dir, sub)
            os.makedirs(d, exist_ok=True)
            for name, data in files.items():
                with open(os.path.join(d, name), "wb") as f:
                    f.write(data)


def _public_doc(d, pub, world):
    ids = pub["docs"][d["key"]]
    ent = pub["entities"][d["entity"]]
    if d["kind"] == "payment":
        refs = None
        if d["refs"] is not None:
            refs = [{"invoice_id": pub["docs"][k]["number"], "amount_minor": a} for k, a in d["refs"]]
        return {"doc_id": ids["doc_id"], "payment_id": ids["number"], "timestamp": fmt(d["timestamp"]),
                "currency": d["currency"], "amount_minor": d["amount"], "invoice_refs": refs, "entity_id": ent}
    return {"doc_id": ids["doc_id"], "invoice_id": ids["number"], "entity_id": ent,
            "issued_at": fmt(d["issued_at"]), "due_at": fmt(d["due_at"]), "currency": d["currency"],
            "total_minor": d["total"], "line_items": [dict(ln) for ln in d["lines"]], "kind": d["kind"],
            "original_invoice_id": pub["docs"][d["original"]]["number"] if d["original"] else None}


def _public_row(r, pub, world):
    ts = r["ts"]
    return {"row_id": pub["rows"][r["key"]], "timestamp": ts if isinstance(ts, str) else fmt(ts),
            "account": r["account"], "debit_minor": r["debit"], "credit_minor": r["credit"],
            "currency": world.base,
            "entity_id": pub["entities"][r["entity"]] if r["entity"] else None,
            "invoice_id": pub["docs"][r["invoice"]]["number"] if r["invoice"] else None,
            "payment_id": pub["docs"][r["payment"]]["number"] if r["payment"] else None}


def _csv_bytes(columns, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(columns)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in columns])
    return buf.getvalue().encode("utf-8")


def _jsonl_bytes(rows):
    return "".join(canonical_json(r) + "\n" for r in rows).encode("utf-8")


def fingerprint(files):
    parts = []
    for name in PUBLIC_FILES:
        parts.append(name.encode("utf-8") + b"\0" + files[name] + b"\0")
    return sha256_hex(b"".join(parts))


def generate(seed, config=None, secret=DEFAULT_SECRET):
    _validate_seed(seed)
    if not isinstance(secret, str) or not secret:
        raise ValueError("secret must be a non-empty string")
    cfg = load(config)
    ch = config_hash(cfg)
    master = hmac_sha256(secret.encode("utf-8"), ("ledgertrap/v1|%d|%s" % (seed, ch)).encode("utf-8"))
    story_key = hmac_sha256(master, b"story")
    ids_key = hmac_sha256(master, ("ids|%d" % cfg["presentation"]["id_salt"]).encode())
    order_key = hmac_sha256(master, ("order|%d" % cfg["presentation"]["order_seed"]).encode())
    story_rng, ids_rng, order_rng = DetRng(story_key), DetRng(ids_key), DetRng(order_key)

    world = World()
    build_company(story_rng.fork("company"), cfg, world)
    build_period(story_rng.fork("period"), cfg, world)
    build_fx(story_rng.fork("fx"), world)
    build_customers(story_rng.fork("customers"), cfg, world)
    story_meta = build_story(story_rng.fork("story"), cfg, world)

    pre = invariants.check(world)
    if pre:
        raise GeneratorInvariantError("clean books violate invariants: %r" % (pre[:10],))
    traps = detect_traps(world)

    issues, plan_meta = inject_errors(story_rng.fork("errors"), cfg, world)
    post = invariants.check(world)
    check_declared_discrepancy(world, issues, post)
    if not (200 <= len(world.rows) <= 2000 and 20 <= len(world.docs) <= 200):
        raise GeneratorInvariantError("episode size outside the case range: %d rows, %d documents" % (
            len(world.rows), len(world.docs)))

    pub = assign_ids(ids_rng, world)
    mode = cfg["presentation"]["order"]
    led_keys = order_keys(order_rng.fork("ledger"), list(world.rows), lambda k: world.rows[k]["ts"], mode)
    inv_keys = order_keys(order_rng.fork("invoices"), [d["key"] for d in world.invoices()],
                          lambda k: world.docs[k]["issued_at"], mode)
    pay_keys = order_keys(order_rng.fork("payments"), [d["key"] for d in world.payments()],
                          lambda k: world.docs[k]["timestamp"], mode)

    ledger = [_public_row(world.rows[k], pub, world) for k in led_keys]
    invoices = [_public_doc(world.docs[k], pub, world) for k in inv_keys]
    payments = [_public_doc(world.docs[k], pub, world) for k in pay_keys]
    fx = [{"date": d.isoformat(), "source_currency": cur, "target_currency": world.base, "rate": format(r, "f")}
          for (d, cur), r in sorted(world.fx.items(), key=lambda kv: (kv[0][0], kv[0][1]))]

    policy = build_policy(world)
    manifest = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "config_version": cfg["config_version"], "episode_id": pub["episode_id"],
        "company_name": world.company["name"], "base_currency": world.base,
        "currencies": dict(world.currencies), "timezone": world.company["timezone"],
        "period_start": fmt(world.period_start), "period_end": fmt(world.period_end),
        "export_start": fmt(world.export_start), "export_end": fmt(world.export_end),
        "counts": {"ledger": len(ledger), "invoices": len(invoices), "payments": len(payments), "fx": len(fx),
                   "documents": len(invoices) + len(payments)},
        "rounding": world.company["rounding"], "csv_null": "", "files": list(PUBLIC_FILES),
    }
    files = {
        "manifest.json": canonical_bytes(manifest) + b"\n",
        "ledger.csv": _csv_bytes(LEDGER_COLUMNS, ledger),
        "invoices.jsonl": _jsonl_bytes(invoices),
        "payments.jsonl": _jsonl_bytes(payments),
        "fx.csv": _csv_bytes(FX_COLUMNS, fx),
        "policy.json": canonical_bytes(policy) + b"\n",
    }
    fp = fingerprint(files)

    truth_issues, certs = public_issues(issues, world, pub)
    truth = {"schema_version": SCHEMA_VERSION, "episode_id": pub["episode_id"], "is_clean": not truth_issues,
             "issues": truth_issues}

    def pub_snap(snap):
        if "role" in snap:
            if snap["key"] not in pub["rows"]:  # row removed by the mutation: no public id
                deleted = dict(snap)
                pub["rows"][snap["key"]] = None
                out = _public_row(deleted, pub, world)
                del pub["rows"][snap["key"]]
                out["deleted"] = True
                return out
            return _public_row(snap, pub, world)
        return {"doc_id": pub["docs"][snap["key"]]["doc_id"]}

    mutations = []
    for i, iss in enumerate(issues, start=1):
        for m in iss["mutations"]:
            entry = {"issue_id": "ISS-%02d" % i, "op": m["op"], "target": pub["docs"][m["target"]]["doc_id"]}
            for side in ("before", "after"):
                if side in m:
                    val = m[side]
                    if isinstance(val, list):
                        entry[side] = [pub_snap(x) for x in val]
                    else:
                        entry[side] = _public_doc(world.docs[m["target"]], pub, world)
            mutations.append(entry)

    provenance = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "seed": seed,
        "secret_id": secret_id(secret), "config": cfg, "config_hash": ch,
        "stream_commitments": {"story": sha256_hex(story_key), "ids": sha256_hex(ids_key),
                               "order": sha256_hex(order_key)},
        "company": {"chart_template": world.company["chart_template"],
                    "period_kind": world.company["period_kind"], "period_label": world.company["period_label"],
                    "foreign_currencies": world.company["foreign_currencies"]},
        "story": story_meta,
        "stages": ["config", "story", "documents", "clean_postings", "invariants_pre", "errors",
                   "declared_discrepancy", "presentation", "export"],
        "invariants_pre": "ok",
        "invariants_post": [[c, _pub_subject(s, pub)] for c, s in post],
        "errors": plan_meta,
        "mutations": mutations,
        "traps_planted": traps,
        "public_fingerprint": fp,
        "file_sha256": {name: sha256_hex(files[name]) for name in PUBLIC_FILES},
    }
    aliases = {}
    for k, ids in pub["docs"].items():
        aliases[ids["doc_id"]] = ids["doc_id"]
        aliases[ids["number"]] = ids["doc_id"]

    ep = Episode()
    ep.seed, ep.cfg, ep.secret_id, ep.config_hash = seed, cfg, secret_id(secret), ch
    ep.episode_id = pub["episode_id"]
    ep.manifest, ep.policy = manifest, policy
    ep.records = {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}
    ep.files = files
    ep.truth, ep.provenance = truth, provenance
    ep.certificates = {"schema_version": SCHEMA_VERSION, "episode_id": pub["episode_id"], "certificates": certs}
    ep.aliases = aliases
    ep.public_fingerprint = fp
    ep.id_map = pub
    return ep


def _subject_docs(world, subj):
    if subj == "*":
        return None
    if isinstance(subj, tuple):
        return set(subj)
    if subj in world.rows:
        r = world.rows[subj]
        return {r["payment"] or r["invoice"] or r["group"]}
    return {subj}


def check_declared_discrepancy(world, issues, post):
    """After mutations: every violation is explained by a declared issue (it concerns only that
    issue's documents), and every issue causes its required violations."""
    post_set = set(post)
    for code, subj in post:
        docs = _subject_docs(world, subj)
        ok = False
        for iss in issues:
            if docs is None:
                ok = ok or code in iss["allowed_global"]
            else:
                ok = ok or docs <= set(iss["allowed_docs"])
        if not ok:
            raise GeneratorInvariantError("undeclared discrepancy %r" % ((code, subj),))
    for iss in issues:
        for v in iss["required"]:
            if tuple(v) not in post_set:
                raise GeneratorInvariantError("issue %s:%s does not show required discrepancy %r" % (
                    iss["type"], iss["variant"], v))


def _pub_subject(s, pub):
    def one(k):
        if k in pub["docs"]:
            return pub["docs"][k]["doc_id"]
        if k in pub["rows"]:
            return pub["rows"][k]
        return k
    if isinstance(s, tuple):
        return [one(k) for k in s]
    return one(s)
