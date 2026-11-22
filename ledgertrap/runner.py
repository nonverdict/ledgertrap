"""Trusted runner: drives a policy through episodes, records trajectories, collects grader results.

Trajectory lines follow case §09: episode_id, step, action, observation, reward, terminated, truncated
(+ policy, split, mode, env_version). Step 0 is reset (action = null). Rewards are exactly what the
policy received, so eval-mode trajectories carry 0.0; true results live in the grader journal.

Isolation (`isolate=True`): every agent runs as its own process (ledgertrap.agent_proc or any command
speaking the same JSON-lines protocol) with an empty working directory and a minimal environment
(no secret); the grader can run in a third process (`grader="process"`), so neither the agent nor the
environment process ever holds the truth.
"""

import json
import os
import select
import subprocess
import sys
import tempfile
import time

from .canon import canonical_json
from .env import LedgerTrapEnv
from .policies.agents import make_policy
from .version import DEFAULT_SECRET, ENV_VERSION

MAX_LOOP = 1000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ERROR = object()     # agent crashed, closed its output or wrote nothing
POLICY_TIMEOUT = object()   # agent did not answer within the episode's remaining time


class SubprocessPolicy:
    """Agent in its own process. `cmd` is an argv list. The runner sets `timeout_s` to the episode's
    remaining time before every action; an agent that stays silent longer is killed (and restarted on the
    next reset)."""

    def __init__(self, cmd, name, stderr_path=None, timeout_s=60.0):
        self.name = name
        self.cmd = cmd
        self.timeout_s = timeout_s
        self.workdir = tempfile.mkdtemp(prefix="ledgertrap-agent-")
        self._err = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
        self._spawn()
        self.last = (0.0, False, False)
        self._pending = None

    def _spawn(self):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ROOT, "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, env=env,
                                  cwd=self.workdir, text=True, encoding="utf-8", bufsize=1)

    def reset(self, obs, info=None):
        if self.p.poll() is not None:  # killed or crashed in a previous episode
            self._spawn()
        self._pending = {"event": "reset", "observation": obs}

    def act(self, obs):
        if self._pending is not None:
            msg, self._pending = self._pending, None
        else:
            r, term, trunc = self.last
            msg = {"event": "step", "observation": obs, "reward": r, "terminated": term, "truncated": trunc}
        try:
            self.p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            return POLICY_ERROR
        ready, _, _ = select.select([self.p.stdout], [], [], max(0.0, self.timeout_s))
        if not ready:
            self.p.kill()
            self.p.wait()
            return POLICY_TIMEOUT
        line = self.p.stdout.readline()
        if not line:
            return POLICY_ERROR
        try:
            return json.loads(line)
        except ValueError:
            return line.strip()  # not JSON -> the environment answers BAD_ACTION and spends a step

    def notify_end(self, obs, r, term, trunc):
        try:
            self.p.stdin.write(json.dumps({"event": "step", "observation": obs, "reward": r, "terminated": term,
                                           "truncated": trunc}, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        try:
            self.p.stdin.write('{"event":"close"}\n')
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.p.kill()
        if self._err is not subprocess.DEVNULL:
            self._err.close()
        try:
            os.rmdir(self.workdir)
        except OSError:
            pass


def builtin_agent(name, seed=0, stderr_path=None):
    return SubprocessPolicy([sys.executable, "-m", "ledgertrap.agent_proc", name, str(seed)], name, stderr_path)


def run_episode(env, policy, seed, config, split="", policy_name=""):
    obs, info = env.reset(seed, config)
    t0 = time.perf_counter()
    policy.reset(obs, info)
    episode_id = obs["manifest"]["episode_id"]

    def line(step, action, o, r, term, trunc):
        return {"episode_id": episode_id, "step": step, "action": action, "observation": o, "reward": r,
                "terminated": term, "truncated": trunc, "policy": policy_name, "split": split,
                "mode": env.mode, "env_version": ENV_VERSION}

    traj = [line(0, None, obs, 0.0, False, False)]
    r, term, trunc = 0.0, False, False
    for step in range(1, MAX_LOOP):
        if hasattr(policy, "last"):
            policy.last = (r, term, trunc)
        if hasattr(policy, "timeout_s"):
            policy.timeout_s = env.time_left() + 0.5  # bounded by the episode budget (case §07: 60 s)
        action = policy.act(obs)
        if action is POLICY_ERROR or action is POLICY_TIMEOUT:
            obs, r, term, trunc, info = env.abort("time_limit" if action is POLICY_TIMEOUT else "policy_error")
            traj.append(line(step, None, obs, r, term, trunc))
            break
        obs, r, term, trunc, info = env.step(action)
        traj.append(line(step, action, obs, r, term, trunc))
        if term or trunc:
            if hasattr(policy, "notify_end"):
                policy.notify_end(obs, r, term, trunc)
            break
    journal = dict(env.journal[-1])
    journal["policy_seconds"] = round(time.perf_counter() - t0, 6)
    journal["policy"] = policy_name
    journal["split"] = split
    return traj, journal


def run_many(policy_names, episodes, mode="eval", secret=DEFAULT_SECRET, traj_dir=None, policy_seed=0,
             isolate=False, grader="inproc"):
    """episodes: list of (split, seed, config). Returns list of journal entries."""
    journals = []
    files = {}
    try:
        for name in policy_names:
            env = LedgerTrapEnv(mode=mode, secret=secret, grader=grader)
            agent = None
            if isolate and name != "oracle":
                agent = builtin_agent(name, policy_seed)
            try:
                for split, seed, cfg in episodes:
                    policy = agent if agent is not None else make_policy(name, env=env, seed=policy_seed)
                    traj, journal = run_episode(env, policy, seed, cfg, split, name)
                    journal["isolated"] = agent is not None
                    journal["grader"] = grader
                    journals.append(journal)
                    if traj_dir is not None:
                        sub = "privileged" if name == "oracle" else split
                        path = os.path.join(traj_dir, sub, "%s.jsonl" % name)
                        if path not in files:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            files[path] = open(path, "w", encoding="utf-8")
                        for rec in traj:
                            files[path].write(canonical_json(rec) + "\n")
            finally:
                if agent is not None:
                    agent.close()
                env.close()
    finally:
        for f in files.values():
            f.close()
    return journals


def run_external(cmd, name, episodes, mode="eval", secret=DEFAULT_SECRET, traj_dir=None, grader="process"):
    """Run any agent command (argv list) speaking the JSON-lines protocol, isolated, on `episodes`."""
    journals = []
    env = LedgerTrapEnv(mode=mode, secret=secret, grader=grader)
    agent = SubprocessPolicy(cmd, name)
    out = None
    try:
        if traj_dir is not None:
            os.makedirs(traj_dir, exist_ok=True)
            out = open(os.path.join(traj_dir, "%s.jsonl" % name), "w", encoding="utf-8")
        for split, seed, cfg in episodes:
            traj, journal = run_episode(env, agent, seed, cfg, split, name)
            journal["isolated"] = True
            journals.append(journal)
            if out is not None:
                for rec in traj:
                    out.write(canonical_json(rec) + "\n")
    finally:
        agent.close()
        env.close()
        if out is not None:
            out.close()
    return journals


def merge_trajectories(traj_dir, out_path, splits=("train", "dev")):
    """Concatenate non-privileged trajectories into one trajectories.jsonl (case §09)."""
    with open(out_path, "w", encoding="utf-8") as out:
        for split in splits:
            d = os.path.join(traj_dir, split)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    for line in f:
                        out.write(line)


def summarize(journals):
    """Mean metrics per (policy, split)."""
    groups = {}
    for j in journals:
        groups.setdefault((j["policy"], j["split"]), []).append(j)
    rows = []
    for (policy, split), js in sorted(groups.items()):
        n = len(js)
        m = [j["metrics"] for j in js]
        tp, fp, fn = sum(x["tp"] for x in m), sum(x["fp"] for x in m), sum(x["fn"] for x in m)
        rows.append({
            "policy": policy, "split": split, "episodes": n,
            "reward_mean": sum(x["reward_main"] for x in m) / n,
            "reward_min": min(x["reward_main"] for x in m), "reward_max": max(x["reward_main"] for x in m),
            "precision_micro": tp / (tp + fp) if tp + fp else 0.0,
            "recall_micro": tp / (tp + fn) if tp + fn else 0.0,
            "f1_micro": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "f1_mean": sum(x["f1"] for x in m) / n,
            "steps_mean": sum(x["inspection_steps"] for x in m) / n,
            "invalid": sum(1 for j in js if not j["valid"]),
            "truncated": sum(1 for j in js if j["reason"] in ("max_steps", "time_limit", "policy_error")),
            "seconds_mean": sum(j["policy_seconds"] for j in js) / n,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
