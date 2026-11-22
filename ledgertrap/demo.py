"""Demo for the video: one episode, the naive reconciler versus our auditor, explained.

python -m ledgertrap demo [--seed N] [--config configs/train.json]
"""

import json
import os
import time

from .env import LedgerTrapEnv
from .policies.agents import make_policy
from .report import classify_fp
from .runner import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _say(text="", pause=0.0):
    print(text, flush=True)
    if pause:
        time.sleep(pause)


def demo(seed=1004, config="configs/train.json", pause=0.0):
    with open(os.path.join(ROOT, config), encoding="utf-8") as f:
        cfg = json.load(f)
    _say("=" * 78)
    _say("LedgerTrap demo: RL environment for audit agents  (seed %d, %s)" % (seed, config))
    _say("=" * 78, pause)
    env = LedgerTrapEnv(mode="train")
    obs, info = env.reset(seed, cfg)
    m = obs["manifest"]
    ep = env.privileged_episode()
    _say("\n1) reset(seed, config) -> observation: task, manifest, policy, tools, budget")
    _say("   company  : %s, base %s, timezone %s" % (m["company_name"], m["base_currency"], m["timezone"]))
    _say("   period   : [%s, %s)" % (m["period_start"], m["period_end"]))
    _say("   records  : %d ledger rows, %d invoices, %d payments, %d fx rates" % (
        m["counts"]["ledger"], m["counts"]["invoices"], m["counts"]["payments"], m["counts"]["fx"]))
    _say("   budget   : %d steps, page <= %d rows, %d s" % (obs["budget"]["steps_left"], obs["budget"]["page_size"],
                                                           obs["budget"]["time_limit_s"]))
    _say("   hidden from the agent: %d planted issues, traps %s" % (
        len(ep.truth["issues"]), ", ".join(ep.provenance["traps_planted"])), pause)

    _say("\n2) The agent may only call tools. Example step:")
    o = env.step({"tool": "list_records", "args": {"table": "payments", "limit": 2}})[0]
    _say("   step(list_records payments, limit 2) -> %d rows, next_cursor=%s, steps_left=%d" % (
        len(o["result"]["rows"]), o["result"]["next_cursor"], o["steps_left"]))
    o = env.step({"tool": "get_record", "args": {"table": "truth", "id": "ISS-01"}})[0]
    _say("   step(get_record truth ISS-01)        -> error %s (no back door to the answer key)" % o["error"]["code"])
    o = env.step({"tool": "calculator", "args": {"expression": "__import__('os')"}})[0]
    _say("   step(calculator __import__('os'))    -> error %s (arithmetic only)" % o["error"]["code"], pause)

    for name in ("naive", "auditor"):
        env = LedgerTrapEnv(mode="train")
        traj, j = run_episode(env, make_policy(name), seed, cfg, "demo", name)
        mt = j["metrics"]
        findings = traj[-1]["action"]["args"]["findings"]["findings"] if traj[-1]["action"] else []
        matched = {i for i, _ in j["pairs"]}
        _say("\n3) %s: %d tool steps, %d findings submitted" % (
            "NAIVE RECONCILER (dates as strings, posting-date FX, totals only)" if name == "naive"
            else "OUR AUDITOR (timezones, weekend rates, line items, cumulative AR)", mt["inspection_steps"],
            len(findings)))
        for i, f in enumerate(findings[:8]):
            verdict = "TP" if i in matched else "FP: " + classify_fp(env.privileged_episode(), f)
            _say("   - %-18s %-28s %-6s -> %s" % (f["type"], ",".join(f["affected_ids"])[:28],
                                                  f["currency"], verdict))
        if len(findings) > 8:
            _say("   ... %d more" % (len(findings) - 8))
        _say("   TP=%d FP=%d FN=%d  P=%.2f R=%.2f F1=%.2f  reward=%.3f" % (
            mt["tp"], mt["fp"], mt["fn"], mt["precision"], mt["recall"], mt["f1"], mt["reward_main"]), pause)

    ep = env.privileged_episode()
    if ep.certificates["certificates"]:
        _say("\n4) Certificate of one planted issue (checkable by hand, grader side):")
        _say("   " + ep.certificates["certificates"][0]["text"], pause)
    _say("\n5) Reward cannot be gamed: duplicates are FP, flag-all ~ -1, guessed ids and borrowed evidence")
    _say("   never match; 15 attack tests + 20 jury-style checks run in `python -m ledgertrap check`.")
    _say("=" * 78)
    return 0
