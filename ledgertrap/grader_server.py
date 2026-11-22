"""Grader process: owns the secret, generates episodes, keeps truth/provenance in memory and scores.

Protocol: one JSON object per line on stdin, one reply per line on stdout.
  {"op":"init","secret":str}                          -> {"ok":true}
  {"op":"reset","seed":int,"config":obj|null}         -> {"ok":true,"public":{...}}   (no truth)
  {"op":"score","submission":any,"steps":int}         -> {"ok":true,"result":{...}}
  {"op":"invalid","reason":str,"steps":int}           -> {"ok":true,"result":{...}}
  {"op":"score_empty","steps":int}                    -> {"ok":true,"result":{...}}
  {"op":"oracle"}                                     -> {"ok":true,"findings":{...}}   (trusted runner only)
  {"op":"close"}
Errors -> {"ok":false,"error":"<code>"} with no internal details. Truth never leaves this process except
through the oracle op, which only the trusted runner uses (case §06: oracle runs on the grader side).
"""

import json
import sys

from .episode_io import make_episode
from .grader import Grader
from .policies.baselines import oracle_findings
from .version import DEFAULT_SECRET


def _result(res):
    return {"valid": res.valid, "reason": res.reason, "reward": res.reward,
            "reward_experimental": res.reward_experimental, "metrics": res.metrics,
            "pairs": res.details["pairs"], "near": res.details["near"]}


def public_part(ep):
    return {"episode_id": ep.episode_id, "seed": ep.seed, "manifest": ep.manifest, "policy": ep.policy,
            "records": ep.records, "aliases": ep.aliases, "limits": ep.cfg["limits"],
            "info": {"config_hash": ep.config_hash, "secret_id": ep.secret_id,
                     "public_fingerprint": ep.public_fingerprint}}


def serve(stdin=sys.stdin, stdout=sys.stdout):
    secret = DEFAULT_SECRET
    ep = grader = None
    for line in stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "init":
                secret = req["secret"]
                out = {"ok": True}
            elif op == "reset":
                ep = make_episode(req["seed"], req.get("config"), secret)
                if ep.truth is None:
                    ep.truth = {"episode_id": ep.episode_id, "issues": [], "unscored": True}
                grader = Grader.for_episode(ep)
                out = {"ok": True, "public": public_part(ep)}
            elif op == "score":
                out = {"ok": True, "result": _result(grader.score(req["submission"], req["steps"]))}
            elif op == "invalid":
                out = {"ok": True, "result": _result(grader.invalid(req["reason"], req["steps"]))}
            elif op == "score_empty":
                out = {"ok": True, "result": _result(grader.score_empty(req["steps"]))}
            elif op == "oracle":
                out = {"ok": True, "findings": oracle_findings(ep)}
            elif op == "close":
                break
            else:
                out = {"ok": False, "error": "UNKNOWN_OP"}
        except ValueError as e:  # bad seed / config from the caller: safe to name the class only
            out = {"ok": False, "error": type(e).__name__, "message": str(e)[:300]}
        except Exception as e:  # noqa: BLE001  never leak internals
            out = {"ok": False, "error": "INTERNAL_ERROR", "kind": type(e).__name__}
        stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        stdout.flush()


if __name__ == "__main__":
    serve()
