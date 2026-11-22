"""Judge adapter for the official FinTech checks (genesis-exe.com/fintech-checks.html, 20.11.2026).

A thin translation layer over LedgerTrap; it does not choose actions and does not change the task.
Run from the repository root (or put it on PYTHONPATH); standard library only, Python 3.11.

Entry points
------------
generate_episode(seed, config, output_dir)
    Team generator (ledgertrap.generator.generate) -> files in output_dir:
        public/  manifest.json ledger.csv invoices.jsonl payments.jsonl fx.csv policy.json
        grader/  truth.json provenance.json certificates.json          (closed part)
    `config` is a team config (see configs/ and configs/scenarios/); {"team_config": {...}} is accepted too.
    Secret for the HMAC seed derivation: env LEDGERTRAP_SECRET (default: the open secret).

make_env() -> JudgeEnv with
    reset(seed, config) -> (observation, info)
        config = {"mode": "closed", "episode_dir": dir, "team_config": {...}}: loads exactly that folder (flat or
        public/ + grader/), never regenerates or re-identifies; reward is 0 on every step, info is {}.
        config = {"mode": "train", "team_config": {...}}: generates (seed, team_config); terminal reward and
        metrics are returned. "episode_dir" may be given in train mode too. team_config only sets limits
        when a folder is loaded.
        First observation: {"task", "manifest", "policy", "remaining_steps": 100}.
    step({"name": <action>, "args": {...}}) -> (observation, reward, terminated, truncated, info)
        names: list_records(table, cursor, limit) | get_record(table, id) | get_document(doc_id) |
               calculator(expression) | submit(findings)
        tables: ledger | invoices | payments | fx; get_record ids: ledger row_id, invoices doc_id or
        invoice_id, payments doc_id or payment_id, fx "YYYY-MM-DD:SRC:TGT".
        Observation: {"result": value | null, "error": null | {"code", "message"}, "remaining_steps": int}
          list_records -> {"records": [...], "next_cursor": str | null}; get_record / get_document -> the
          record object as in the files; calculator -> decimal string; submit -> {"accepted": true} or error
          INVALID_SUBMISSION (structure only, never correctness).
        Every non-submit action (also erroneous ones) costs one step and counts in inspection_steps; submit is
        one of the 100 actions but not an inspection step. 100 inspection actions -> truncated, empty answer.
        60 s per episode from the start of reset (policy time included) -> truncated, empty answer.

score_episode(episode_dir, findings, inspection_steps) -> {"tp","fp","fn","precision","recall","f1","return"}
    Grader-side call: reads truth.json + the public files of episode_dir. Invalid structure -> return = -1 and
    the other fields null; the reason is appended to JOURNAL (and env LEDGERTRAP_JOURNAL, if set, gets one JSON
    line per call). Matching rules: ledgertrap/grader/ (exact ids/currency, τ from policy.json, maximum matching).

Issue types (case §02 -> LedgerTrap `type`): duplicate invoices -> duplicate_invoice; missing or partial payment
allocation -> payment_allocation (variants missing, partial, wrong_invoice); wrong conversion date or rate ->
fx_conversion (wrong_date, wrong_rate); reporting-period error -> period_cutoff (early, late).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledgertrap.env import LedgerTrapEnv  # noqa: E402
from ledgertrap.episode_io import load_episode_dir  # noqa: E402
from ledgertrap.generator import generate  # noqa: E402
from ledgertrap.grader import Grader  # noqa: E402
from ledgertrap.version import DEFAULT_SECRET  # noqa: E402

ACTION_NAMES = ("list_records", "get_record", "get_document", "calculator", "submit")
ENVELOPE_KEYS = frozenset(["mode", "episode_dir", "team_config"])
JOURNAL = []  # grader journal of this (jury) process

END_ERRORS = {
    "invalid_submission": ("INVALID_SUBMISSION", "findings.json does not have the required structure (case §05)"),
    "max_steps": ("STEP_LIMIT", "100 actions used; episode truncated with empty findings"),
    "time_limit": ("TIME_LIMIT", "60 s episode limit reached; episode truncated with empty findings"),
    "policy_error": ("POLICY_ERROR", "policy stopped; episode truncated with empty findings"),
}


def _secret():
    return os.environ.get("LEDGERTRAP_SECRET", DEFAULT_SECRET)


def _journal(entry):
    JOURNAL.append(entry)
    path = os.environ.get("LEDGERTRAP_JOURNAL")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _team_config(config):
    if isinstance(config, dict) and config and set(config) <= ENVELOPE_KEYS:
        return config.get("team_config")
    return config


def generate_episode(seed, config, output_dir):
    ep = generate(seed, _team_config(config), _secret())
    ep.write(output_dir)


def _translate_action(action):
    if isinstance(action, dict) and set(action) == {"name", "args"}:
        return {"tool": action["name"], "args": action["args"]}
    return action  # anything else is answered by the environment (BAD_ACTION / UNKNOWN_TOOL / BAD_ARGS)


def _translate_obs(obs):
    if obs.get("type") == "episode_end":
        if obs["reason"] == "submitted":
            return {"result": {"accepted": True}, "error": None, "remaining_steps": obs["steps_left"]}
        code, msg = END_ERRORS[obs["reason"]]
        return {"result": None, "error": {"code": code, "message": msg}, "remaining_steps": obs["steps_left"]}
    if not obs["ok"]:
        return {"result": None, "error": dict(obs["error"]), "remaining_steps": obs["steps_left"]}
    res, tool = obs["result"], obs["tool"]
    if tool == "list_records":
        value = {"records": res["rows"], "next_cursor": res["next_cursor"]}
    elif tool == "get_record":
        value = res["record"]
    elif tool == "get_document":
        value = res["document"]
    else:  # calculator
        value = res["value"]
    return {"result": value, "error": None, "remaining_steps": obs["steps_left"]}


class JudgeEnv:
    """reset/step in the official envelope over ledgertrap.env.LedgerTrapEnv."""

    def __init__(self):
        self._env = LedgerTrapEnv(mode="train", secret=_secret())
        self.mode = None

    def reset(self, seed, config=None):
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if config and not set(config) <= ENVELOPE_KEYS:  # a bare team config: open (train) mode
            config = {"mode": "train", "team_config": config}
        mode = config.get("mode", "train")
        if mode not in ("train", "closed"):
            raise ValueError("mode must be 'train' or 'closed'")
        if "episode_dir" in config:
            inner = {"episode_dir": config["episode_dir"], "team_config": config.get("team_config")}
        elif mode == "closed":
            raise ValueError("closed mode needs episode_dir")
        else:
            inner = config.get("team_config")
        self._env.mode = "eval" if mode == "closed" else "train"
        self.mode = mode
        obs, info = self._env.reset(seed, inner)
        first = {"task": obs["task"], "manifest": obs["manifest"], "policy": obs["policy"],
                 "remaining_steps": obs["budget"]["steps_left"]}
        return first, ({} if mode == "closed" else info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(_translate_action(action))
        if obs.get("type") == "episode_end":
            _journal(dict(self._env.journal[-1], source="episode", mode=self.mode))
        if self.mode == "closed":
            return _translate_obs(obs), 0, terminated, truncated, {}
        return _translate_obs(obs), reward, terminated, truncated, info

    @property
    def journal(self):
        """Grader journal of this environment (jury side only)."""
        return self._env.journal

    def close(self):
        self._env.close()


def make_env():
    return JudgeEnv()


def score_episode(episode_dir, findings, inspection_steps):
    if not (isinstance(inspection_steps, int) and not isinstance(inspection_steps, bool) and inspection_steps >= 0):
        raise ValueError("inspection_steps must be a non-negative integer")
    ep = load_episode_dir(episode_dir)
    if ep.truth is None:
        raise FileNotFoundError("truth.json not found in episode_dir (grader/ or flat)")
    res = Grader.for_episode(ep).score(findings, inspection_steps)
    m = res.metrics
    _journal({"source": "score_episode", "episode_id": ep.episode_id, "valid": res.valid,
              "invalid_reason": res.reason, "inspection_steps": inspection_steps, "metrics": m,
              "pairs": res.details["pairs"]})
    if not res.valid:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    return {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "return": res.reward}
