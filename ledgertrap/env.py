"""LedgerTrapEnv: JSON-compatible Gymnasium-style contract (case §04, DESIGN §10).

reset(seed, config) -> (observation, info)
step(action)        -> (observation, reward, terminated, truncated, info)

Modes:
- train: terminal reward and aggregated metrics are returned to the policy.
- eval:  reward is always 0.0 and info is always {}; true results go to the grader journal only.
"""

import json
import time
import traceback

from .calculator import CalcError, evaluate
from .canon import canonical_json
from .episode_io import make_episode
from .grader import Grader
from .policy import TASK_TEXT
from .store import TABLES, RecordStore, StoreError
from .version import DEFAULT_SECRET, ENV_VERSION, GENERATOR_VERSION, SCHEMA_VERSION

MAX_ACTION_BYTES = 1_000_000

ERROR_MESSAGES = {
    "BAD_ACTION": "action must be a JSON object {\"tool\": str, \"args\": object}",
    "UNKNOWN_TOOL": "unknown tool",
    "BAD_ARGS": "invalid arguments for this tool",
    "UNKNOWN_TABLE": "unknown table",
    "UNKNOWN_ID": "no record with this id",
    "BAD_CURSOR": "invalid cursor",
    "CALC_ERROR": "expression rejected: only numbers, + - * /, parentheses; max 200 chars",
    "INTERNAL_ERROR": "internal error",
    "EPISODE_DONE": "episode has ended; call reset",
}

TOOL_SPECS = [
    {"name": "list_records",
     "description": "Page through a table in its file order. Records are returned as in the episode files "
                    "(case section 03 fields). Returns rows, next_cursor (null at the end) and total.",
     "args_schema": {"table": {"type": "string", "enum": list(TABLES), "required": True},
                     "cursor": {"type": ["string", "null"], "required": False},
                     "limit": {"type": "integer", "minimum": 1, "maximum": 50, "required": False}}},
    {"name": "get_record",
     "description": "Fetch one record. ledger: row_id; invoices: invoice_id or doc_id; payments: payment_id "
                    "or doc_id; fx: 'YYYY-MM-DD:SRC:TGT'.",
     "args_schema": {"table": {"type": "string", "enum": list(TABLES), "required": True},
                     "id": {"type": "string", "required": True}}},
    {"name": "get_document",
     "description": "Fetch a full document (invoice with line_items, or payment) by doc_id.",
     "args_schema": {"doc_id": {"type": "string", "required": True}}},
    {"name": "calculator",
     "description": "Exact decimal arithmetic: numbers, + - * /, parentheses, unary minus. Max 200 chars.",
     "args_schema": {"expression": {"type": "string", "required": True}}},
    {"name": "submit",
     "description": "Submit findings.json {episode_id, findings:[{type, affected_ids, claimed_amount_minor, "
                    "currency, evidence_ids, explanation}]}. Ends the episode.",
     "args_schema": {"findings": {"type": "object", "required": True}}},
]

_ARGS = {
    "list_records": ({"table"}, {"cursor", "limit"}),
    "get_record": ({"table", "id"}, set()),
    "get_document": ({"doc_id"}, set()),
    "calculator": ({"expression"}, set()),
}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


class _ToolError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class LedgerTrapEnv:
    def __init__(self, mode="train", secret=DEFAULT_SECRET, clock=time.monotonic, grader="inproc"):
        """grader="process" runs generation, truth and scoring in a separate grader process
        (ledgertrap.grader_server); this process then never holds truth, provenance or the seed key."""
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if grader not in ("inproc", "process"):
            raise ValueError("grader must be 'inproc' or 'process'")
        self.mode = mode
        self.grader_mode = grader
        self._secret = secret if grader == "inproc" else None
        self._clock = clock
        self.journal = []          # grader journal (trusted side only)
        self.internal_log = []     # tracebacks of INTERNAL_ERROR (trusted side only)
        self._episode = None
        self._done = True
        self._final_flags = (False, False)
        self._remote = RemoteGrader(secret) if grader == "process" else None

    def close(self):
        if self._remote is not None:
            self._remote.close()

    # ---------------------------------------------------------------- reset
    def reset(self, seed, config=None):
        """config: a team generator config (see configs/), or {"episode_dir": path, "team_config": {...}}
        to serve an episode folder exactly as written (judge adapter; nothing is regenerated)."""
        gen_t0 = self._clock()
        if self._remote is None:
            ep = make_episode(seed, config, self._secret)
            if ep.truth is None:  # folder without grader files: structure is still checked, nothing scored
                ep.truth = {"episode_id": ep.episode_id, "issues": [], "unscored": True}
            grader = Grader.for_episode(ep)
        else:
            ep = self._remote.reset(seed, config)
            grader = self._remote
        self._gen_seconds = self._clock() - gen_t0
        self._episode = ep
        lim = ep.cfg["limits"]
        self._max_steps = lim["max_steps"]
        self._time_limit = lim["time_limit_s"]
        self._store = RecordStore(ep, page_size=lim["page_size"])
        self._grader = grader
        self._steps = 0
        self._done = False
        self._final_flags = (False, False)
        self._t0 = gen_t0  # the 60 s episode budget runs from the start of reset (official checks, C18)
        obs = {
            "type": "reset",
            "task": TASK_TEXT,
            "manifest": json.loads(json.dumps(ep.manifest)),
            "policy": json.loads(json.dumps(ep.policy)),
            "tools": json.loads(json.dumps(TOOL_SPECS)),
            "budget": {"steps_left": self._max_steps, "max_steps": self._max_steps,
                       "page_size": lim["page_size"], "time_limit_s": self._time_limit},
        }
        if self.mode == "eval":
            return obs, {}
        info = {"episode_id": ep.episode_id, "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION, "env_version": ENV_VERSION,
                "config_hash": ep.config_hash, "secret_id": ep.secret_id,
                "public_fingerprint": ep.public_fingerprint, "mode": self.mode}
        return obs, info

    # ---------------------------------------------------------------- step
    def step(self, action):
        if self._episode is None:
            raise RuntimeError("call reset() first")
        if self._done:
            obs = {"type": "tool_result", "tool": None, "ok": False,
                   "error": {"code": "EPISODE_DONE", "message": ERROR_MESSAGES["EPISODE_DONE"]},
                   "steps_left": self._max_steps - self._steps}
            return obs, 0.0, self._final_flags[0], self._final_flags[1], {}

        if self._clock() - self._t0 > self._time_limit:
            return self._finish_empty("time_limit")

        tool = action.get("tool") if isinstance(action, dict) else None
        if tool == "submit":
            return self._submit(action)

        self._steps += 1
        try:
            result = self._dispatch(action)
            obs = {"type": "tool_result", "tool": tool, "ok": True, "result": result,
                   "steps_left": self._max_steps - self._steps}
        except _ToolError as e:
            obs = self._error_obs(tool, e.code)
        except Exception:  # never leak tracebacks to the agent
            self.internal_log.append(traceback.format_exc())
            obs = self._error_obs(tool, "INTERNAL_ERROR")
        if self._steps >= self._max_steps:
            return self._finish_empty("max_steps")
        return obs, 0.0, False, False, {}

    def _error_obs(self, tool, code):
        return {"type": "tool_result", "tool": tool if isinstance(tool, str) and len(tool) <= 64 else None,
                "ok": False, "error": {"code": code, "message": ERROR_MESSAGES[code]},
                "steps_left": self._max_steps - self._steps}

    def _dispatch(self, action):
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            raise _ToolError("BAD_ACTION")
        if not isinstance(action["tool"], str) or not isinstance(action["args"], dict):
            raise _ToolError("BAD_ACTION")
        try:
            size = len(canonical_json(action).encode("utf-8"))
        except (TypeError, ValueError):
            raise _ToolError("BAD_ACTION") from None
        if size > MAX_ACTION_BYTES:
            raise _ToolError("BAD_ACTION")
        tool, args = action["tool"], action["args"]
        if tool not in _ARGS:
            raise _ToolError("UNKNOWN_TOOL")
        required, optional = _ARGS[tool]
        keys = set(args)
        if not required <= keys or not keys <= (required | optional):
            raise _ToolError("BAD_ARGS")
        try:
            if tool == "list_records":
                table, cursor, limit = args["table"], args.get("cursor"), args.get("limit")
                if not isinstance(table, str) or (cursor is not None and not isinstance(cursor, str)):
                    raise _ToolError("BAD_ARGS")
                if limit is not None and not (_is_int(limit) and 1 <= limit <= self._store.page_size):
                    raise _ToolError("BAD_ARGS")
                return self._store.list_records(table, cursor, limit)
            if tool == "get_record":
                if not isinstance(args["table"], str) or not isinstance(args["id"], str):
                    raise _ToolError("BAD_ARGS")
                return self._store.get_record(args["table"], args["id"])
            if tool == "get_document":
                if not isinstance(args["doc_id"], str):
                    raise _ToolError("BAD_ARGS")
                return self._store.get_document(args["doc_id"])
            if tool == "calculator":
                try:
                    return {"value": evaluate(args["expression"])}
                except CalcError:
                    raise _ToolError("CALC_ERROR") from None
        except StoreError as e:
            raise _ToolError(e.code) from None
        raise _ToolError("UNKNOWN_TOOL")

    # ---------------------------------------------------------------- terminal
    def _submit(self, action):
        ok_shape = (isinstance(action, dict) and set(action) == {"tool", "args"}
                    and isinstance(action["args"], dict) and set(action["args"]) == {"findings"})
        try:
            too_big = len(canonical_json(action).encode("utf-8")) > MAX_ACTION_BYTES
        except (TypeError, ValueError):
            too_big, ok_shape = True, False
        if too_big:
            res = self._grader.invalid("SUBMISSION_TOO_LARGE", self._steps)
        elif not ok_shape:
            res = self._grader.invalid("BAD_SUBMIT_ARGS", self._steps)
        else:
            res = self._grader.score(action["args"]["findings"], self._steps)
        reason = "submitted" if res.valid else "invalid_submission"
        return self._finish(res, reason, terminated=True)

    def _finish_empty(self, reason):
        res = self._grader.score_empty(self._steps)
        return self._finish(res, reason, terminated=False)

    def _finish(self, res, reason, terminated):
        self._done = True
        truncated = not terminated
        self._final_flags = (terminated, truncated)
        elapsed = self._clock() - self._t0
        self.journal.append({
            "episode_id": self._episode.episode_id, "seed": self._episode.seed, "mode": self.mode,
            "reason": reason, "valid": res.valid, "metrics": res.metrics, "pairs": res.details["pairs"],
            "near": res.details["near"], "generation_s": round(self._gen_seconds, 6),
            "elapsed_s": round(elapsed, 6), "steps": self._steps,
        })
        obs = {"type": "episode_end", "reason": reason, "accepted": bool(res.valid and terminated),
               "steps_left": self._max_steps - self._steps}
        if self.mode == "eval":
            return obs, 0.0, terminated, truncated, {}
        metrics = dict(res.metrics)
        return obs, float(res.reward), terminated, truncated, {"metrics": metrics}

    def time_left(self):
        """Seconds left in the current episode's time budget (trusted runner uses it to bound agent waits)."""
        if self._episode is None or self._done:
            return 0.0
        return max(0.0, self._time_limit - (self._clock() - self._t0))

    def abort(self, reason="policy_error"):
        """Trusted runner: end the episode with an empty answer (policy crashed, hung or quit)."""
        if self._done:
            return None
        return self._finish_empty(reason)

    # ---------------------------------------------------------------- trusted-side helpers
    def privileged_episode(self):
        """Grader-side access (reports, tests). Not available when the grader runs in its own process."""
        if self._remote is not None:
            raise RuntimeError("truth lives in the grader process")
        return self._episode

    def oracle_findings(self):
        """Grader-side oracle answer (case §06). Trusted runner only; never pass to an agent."""
        if self._remote is not None:
            return self._remote.oracle()
        from .policies.baselines import oracle_findings
        return oracle_findings(self._episode)


class _PublicEpisode:
    """What the environment process knows in process-grader mode: public records only."""

    def __init__(self, pub):
        self.episode_id = pub["episode_id"]
        self.seed = pub["seed"]
        self.manifest = pub["manifest"]
        self.policy = pub["policy"]
        self.records = pub["records"]
        self.aliases = pub["aliases"]
        self.cfg = {"limits": pub["limits"]}
        self.config_hash = pub["info"]["config_hash"]
        self.secret_id = pub["info"]["secret_id"]
        self.public_fingerprint = pub["info"]["public_fingerprint"]


class RemoteGrader:
    """Client of ledgertrap.grader_server over JSON lines (stdin/stdout)."""

    def __init__(self, secret):
        import os
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": root, "PYTHONHASHSEED": "0",
               "LANG": "C.UTF-8"}
        self._p = subprocess.Popen([sys.executable, "-m", "ledgertrap.grader_server"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=root,
                                   text=True, encoding="utf-8", bufsize=1)
        self._call({"op": "init", "secret": secret})

    def _call(self, req):
        self._p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._p.stdin.flush()
        line = self._p.stdout.readline()
        if not line:
            raise RuntimeError("grader process died")
        out = json.loads(line)
        if not out.get("ok"):
            if out.get("error") == "ValueError":
                raise ValueError(out.get("message", "invalid reset arguments"))
            raise RuntimeError("grader error: %s" % out.get("error"))
        return out

    def reset(self, seed, config):
        return _PublicEpisode(self._call({"op": "reset", "seed": seed, "config": config})["public"])

    @staticmethod
    def _res(d):
        from .grader import GradeResult
        return GradeResult(d["valid"], d["reason"], d["reward"], d["reward_experimental"], d["metrics"],
                           {"pairs": [tuple(x) for x in d["pairs"]], "near": [tuple(x) for x in d["near"]]})

    def score(self, submission, steps):
        return self._res(self._call({"op": "score", "submission": submission, "steps": steps})["result"])

    def invalid(self, reason, steps):
        return self._res(self._call({"op": "invalid", "reason": reason, "steps": steps})["result"])

    def score_empty(self, steps):
        return self._res(self._call({"op": "score_empty", "steps": steps})["result"])

    def oracle(self):
        return self._call({"op": "oracle"})["findings"]

    def close(self):
        try:
            self._p.stdin.write(json.dumps({"op": "close"}) + "\n")
            self._p.stdin.flush()
            self._p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._p.kill()
            self._p.wait()
        for f in (self._p.stdin, self._p.stdout):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass

    def __del__(self):
        try:
            if self._p.poll() is None:
                self._p.kill()
        except Exception:  # noqa: BLE001
            pass
