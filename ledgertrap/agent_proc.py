"""Run a built-in policy as an isolated agent process (JSON lines on stdin/stdout).

Protocol (DESIGN §10.9), one JSON object per line:
  runner -> agent : {"event":"reset","observation":...}
                    {"event":"step","observation":...,"reward":...,"terminated":...,"truncated":...}
                    {"event":"close"}
  agent -> runner : one action {"tool":...,"args":...} after every non-terminal event.
Any external agent can speak the same protocol: python -m ledgertrap evaluate --policy-cmd "...".

`probe` is a diagnostic agent for isolation tests: it reports what it can see on stderr.
"""

import json
import os
import sys

from .policies.agents import make_policy


def _probe_report():
    ppid = os.getppid()
    seen = {"env_keys": sorted(os.environ), "cwd": os.getcwd(), "cwd_files": sorted(os.listdir(".")),
            "parent_environ_has_secret": None}
    try:
        with open("/proc/%d/environ" % ppid, "rb") as f:
            seen["parent_environ_has_secret"] = b"SECRET" in f.read().upper()
    except OSError:
        seen["parent_environ_has_secret"] = "unreadable"
    return seen


def main(argv):
    name = argv[0] if argv else "silent"
    seed = int(argv[1]) if len(argv) > 1 else 0
    probe = name == "probe"
    policy = make_policy("silent" if probe else name, seed=seed)
    out = sys.stdout
    for line in sys.stdin:
        msg = json.loads(line)
        ev = msg.get("event")
        if ev == "close":
            break
        obs = msg["observation"]
        if ev == "reset":
            policy.reset(obs, None)
            if probe:
                sys.stderr.write(json.dumps(_probe_report()) + "\n")
                sys.stderr.flush()
        elif msg.get("terminated") or msg.get("truncated"):
            continue
        out.write(json.dumps(policy.act(obs), ensure_ascii=False) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
