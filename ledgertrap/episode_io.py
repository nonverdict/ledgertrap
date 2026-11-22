"""Episode folders on disk: load an episode exactly as the files say (judge adapter, closed mode).

Layout written by `Episode.write` and `judge_adapter.generate_episode`:

    <episode_dir>/public/  manifest.json ledger.csv invoices.jsonl payments.jsonl fx.csv policy.json
    <episode_dir>/grader/  truth.json provenance.json certificates.json        (closed: jury / grader only)

A flat folder (all files directly in <episode_dir>) is accepted too. Nothing is regenerated, nothing is
re-identified: records keep the ids and the order of the files, so a jury copy with permuted records and
a consistent id mapping is served as is. Types follow case §03: ledger debit_minor/credit_minor are
integers, an empty CSV cell (manifest.csv_null = "") is null; fx rate stays a decimal string.
"""

import csv
import io
import json
import os

from .config import load as load_config
from .generator import FX_COLUMNS, LEDGER_COLUMNS, PUBLIC_FILES, fingerprint

CLOSED_FILES = ("truth.json", "provenance.json", "certificates.json")
LEDGER_INT = ("debit_minor", "credit_minor")
LEDGER_NULLABLE = ("entity_id", "invoice_id", "payment_id")


class EpisodeFormatError(ValueError):
    pass


def _path(episode_dir, sub, name):
    for p in (os.path.join(episode_dir, sub, name), os.path.join(episode_dir, name)):
        if os.path.isfile(p):
            return p
    return None


def _read(episode_dir, sub, name, required=True):
    p = _path(episode_dir, sub, name)
    if p is None:
        if required:
            raise EpisodeFormatError("episode file %s is missing" % name)
        return None
    with open(p, "rb") as f:
        return f.read()


def _csv_rows(data, columns, name):
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    header = next(reader, None)
    if header != columns:
        raise EpisodeFormatError("%s header must be %s" % (name, ",".join(columns)))
    out = []
    for row in reader:
        if len(row) != len(columns):
            raise EpisodeFormatError("%s: row with %d cells" % (name, len(row)))
        out.append(dict(zip(columns, row)))
    return out


def parse_ledger(data):
    rows = []
    for r in _csv_rows(data, LEDGER_COLUMNS, "ledger.csv"):
        for k in LEDGER_INT:
            try:
                r[k] = int(r[k])
            except ValueError:
                raise EpisodeFormatError("ledger.csv: %s must be an integer" % k) from None
        for k in LEDGER_NULLABLE:
            if r[k] == "":
                r[k] = None
        rows.append(r)
    return rows


def parse_fx(data):
    return _csv_rows(data, FX_COLUMNS, "fx.csv")


def parse_jsonl(data, name):
    out = []
    for i, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise EpisodeFormatError("%s line %d is not JSON" % (name, i)) from None
        if not isinstance(obj, dict):
            raise EpisodeFormatError("%s line %d is not an object" % (name, i))
        out.append(obj)
    return out


class LoadedEpisode:
    """Same attributes the environment and grader use on a generated Episode."""

    def __init__(self):
        self.seed = None
        self.cfg = None
        self.secret_id = None
        self.config_hash = None
        self.episode_id = None
        self.manifest = None
        self.policy = None
        self.records = None
        self.files = None
        self.truth = None
        self.provenance = None
        self.certificates = None
        self.aliases = None
        self.public_fingerprint = None
        self.id_map = None
        self.source_dir = None


def load_episode_dir(episode_dir, team_config=None, with_closed=True):
    """Load a folder. `team_config` only supplies limits (max_steps, page_size, time_limit_s); the records
    always come from the files. with_closed=False never opens truth/provenance (public side only)."""
    if not isinstance(episode_dir, str) or not os.path.isdir(episode_dir):
        raise EpisodeFormatError("episode_dir is not a directory")
    files = {name: _read(episode_dir, "public", name) for name in PUBLIC_FILES}
    try:
        manifest = json.loads(files["manifest.json"].decode("utf-8"))
        policy = json.loads(files["policy.json"].decode("utf-8"))
    except ValueError:
        raise EpisodeFormatError("manifest.json / policy.json must be JSON") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episode_id"), str):
        raise EpisodeFormatError("manifest.json needs a string episode_id")
    ep = LoadedEpisode()
    ep.source_dir = episode_dir
    ep.manifest, ep.policy, ep.files = manifest, policy, files
    ep.episode_id = manifest["episode_id"]
    ep.records = {"ledger": parse_ledger(files["ledger.csv"]),
                  "invoices": parse_jsonl(files["invoices.jsonl"], "invoices.jsonl"),
                  "payments": parse_jsonl(files["payments.jsonl"], "payments.jsonl"),
                  "fx": parse_fx(files["fx.csv"])}
    ep.public_fingerprint = fingerprint(files)
    aliases = {}
    for d in ep.records["invoices"] + ep.records["payments"]:
        aliases[d["doc_id"]] = d["doc_id"]
        aliases[d.get("invoice_id") or d.get("payment_id")] = d["doc_id"]
    ep.aliases = aliases
    ep.cfg = {"limits": load_config(team_config)["limits"]}
    if with_closed:
        for name, attr in (("truth.json", "truth"), ("provenance.json", "provenance"),
                           ("certificates.json", "certificates")):
            data = _read(episode_dir, "grader", name, required=False)
            if data is not None:
                setattr(ep, attr, json.loads(data.decode("utf-8")))
        if ep.provenance:
            ep.seed = ep.provenance.get("seed")
            ep.config_hash = ep.provenance.get("config_hash")
            ep.secret_id = ep.provenance.get("secret_id")
    return ep


def make_episode(seed, config, secret):
    """Generator or folder: a config with "episode_dir" loads that folder (closed/open judge runs),
    anything else is a team generator config."""
    if isinstance(config, dict) and "episode_dir" in config:
        return load_episode_dir(config["episode_dir"], config.get("team_config"))
    from .generator import generate
    return generate(seed, config, secret)
