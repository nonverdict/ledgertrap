"""A minimal external agent speaking LedgerTrap's JSON-lines protocol (stdin/stdout). No imports from ledgertrap.

Run it isolated against the open dev split:
    python -m ledgertrap evaluate --config configs/dev.json --seeds 2000-2009 --policy-cmd "python examples/minimal_agent.py"

Protocol: the runner sends {"event":"reset"|"step","observation":...} lines; after every non-terminal event the
agent prints exactly one action {"tool":..., "args":...}. This agent pages through the ledger, then flags every
document whose posting falls in a different interval than its date (period vs after-period window), using the
company timezone from the manifest.
"""

import json
import sys
from datetime import datetime


def interval(ts, start, end):
    t = datetime.fromisoformat(ts)
    return "pre" if t < start else ("period" if t < end else "window")


def main():
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("event") == "close":
            return
        obs = msg["observation"]
        if msg.get("terminated") or msg.get("truncated"):
            continue
        if msg["event"] == "reset":
            m = obs["manifest"]
            state = {"m": m, "start": datetime.fromisoformat(m["period_start"]),
                     "end": datetime.fromisoformat(m["period_end"]), "rows": [], "docs": [],
                     "queue": ["ledger", "invoices", "payments"], "cursor": None}
        elif obs.get("ok") and obs.get("tool") == "list_records":
            key = "rows" if state["queue"][0] == "ledger" else "docs"
            state[key] += obs["result"]["rows"]
            state["cursor"] = obs["result"]["next_cursor"]
            if state["cursor"] is None:
                state["queue"].pop(0)
        if state["queue"] and obs.get("steps_left", 100) > 1:
            action = {"tool": "list_records", "args": {"table": state["queue"][0], "cursor": state["cursor"]}}
        else:
            findings = []
            for d in state["docs"]:
                num = d.get("invoice_id") or d.get("payment_id")
                rows = [r for r in state["rows"] if (r["payment_id"] == num) or
                        (r["invoice_id"] == num and r["payment_id"] is None)]
                when = d.get("issued_at") or d.get("timestamp")
                if not rows or interval(when, state["start"], state["end"]) == "pre":
                    continue
                if interval(when, state["start"], state["end"]) != interval(rows[0]["timestamp"], state["start"], state["end"]):
                    amount = abs(d.get("total_minor", d.get("amount_minor", 0)))
                    findings.append({"type": "period_cutoff", "affected_ids": [d["doc_id"]],
                                     "claimed_amount_minor": amount, "currency": d["currency"],
                                     "evidence_ids": [d["doc_id"], rows[0]["row_id"]],
                                     "explanation": "posting interval differs from document interval"})
            action = {"tool": "submit", "args": {"findings": {"episode_id": state["m"]["episode_id"],
                                                             "findings": findings[:30]}}}
        sys.stdout.write(json.dumps(action) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
