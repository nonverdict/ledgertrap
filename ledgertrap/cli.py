"""Command line: python -m ledgertrap <command>.

check     one-command verification (tests, determinism, re-audit, isolated agents, leaks,
          official jury checks through judge_adapter.py, extended internal checks)
jury      official jury checks (blocks 1-4 + engineering), then the extended internal checks
generate  generate episodes into a directory
export    write the open train/dev episodes to episodes/
run       write the published trajectories (trajectories.jsonl)
report    measure policies and write reports/diagnostics.md
evaluate  run policies on seeds and print metrics
reaudit   independent re-audit of generated episodes against truth
schemas   write JSON schemas to schemas/
demo      one episode, naive vs auditor, explained step by step
play      one episode by hand (JSON actions on stdin)
serve     one episode as a JSON-lines server on stdin/stdout
"""

import argparse
import json
import os
import shutil
import sys
import time
import unittest

from .generator import generate
from .leakscan import scan_trajectory_file, stat_report
from .reaudit import reaudit
from .runner import merge_trajectories, run_many, summarize, write_json
from .version import DEFAULT_SECRET, ENV_VERSION, GENERATOR_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JURY_POLICIES = ["random", "naive", "flag_all", "oracle"]
ALL_POLICIES = JURY_POLICIES + ["auditor", "silent"]


def _load_json(path):
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def _read_secret(args):
    if getattr(args, "secret_file", None):
        with open(args.secret_file, encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_SECRET


def _seeds(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def split_episodes(quick=False):
    """[(split, seed, config)] for the open train/dev splits from configs/splits.json."""
    sp = _load_json("configs/splits.json")
    eps = []
    for split in ("train", "dev"):
        cfg = _load_json(sp[split]["config"])
        seeds = sp[split]["seeds"]
        for seed in (seeds[:2] if quick else seeds):
            eps.append((split, seed, cfg))
    return eps


class Check:
    def __init__(self):
        self.results = []

    def record(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" - " + detail) if detail else ""))
        sys.stdout.flush()
        return ok

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.results)


def print_table(rows):
    print("    %-9s %-7s %4s %8s %7s %7s %6s %6s %6s %7s %5s %5s" % (
        "policy", "split", "eps", "reward", "min", "max", "P", "R", "F1", "steps", "trunc", "inval"))
    for r in rows:
        print("    %-9s %-7s %4d %8.3f %7.3f %7.3f %6.2f %6.2f %6.2f %7.1f %5d %5d" % (
            r["policy"], r["split"], r["episodes"], r["reward_mean"], r["reward_min"], r["reward_max"],
            r["precision_micro"], r["recall_micro"], r["f1_micro"], r["steps_mean"], r["truncated"], r["invalid"]))


def cmd_check(args):
    secret = _read_secret(args)
    print("LedgerTrap check  generator=%s env=%s python=%s" % (GENERATOR_VERSION, ENV_VERSION, sys.version.split()[0]))
    if sys.version_info[:2] != (3, 11):
        print("  note: the reference interpreter is Python 3.11; running on %d.%d" % sys.version_info[:2])
    t0 = time.time()
    chk = Check()
    out_dir = os.path.join(ROOT, "build", "check")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/7] unit tests")
    if args.skip_tests:
        print("  (skipped)")
    else:
        suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"), top_level_dir=ROOT)
        with open(os.devnull, "w") as devnull:
            t1 = time.time()
            res = unittest.TextTestRunner(stream=devnull, verbosity=0).run(suite)
        for test, tb in res.failures + res.errors:
            print("    %s\n      %s" % (test.id(), tb.strip().splitlines()[-1]))
        chk.record("unit tests", res.wasSuccessful(), "%d run, %d failures, %d errors, %.1fs" % (
            res.testsRun, len(res.failures), len(res.errors), time.time() - t1))

    print("\n[2/7] open train/dev episodes: regenerate twice, compare bytes (and with episodes/ if present)")
    eps = split_episodes(args.quick)
    det_ok, same_as_published, counts = True, True, {"issues": 0, "clean": 0}
    for split, seed, cfg in eps:
        ep = generate(seed, cfg, secret)
        again = generate(seed, cfg, secret)
        det_ok &= ep.files == again.files and ep.grader_files() == again.grader_files()
        pub = os.path.join(ROOT, "episodes", split, ep.episode_id, "public")
        if os.path.isdir(pub) and secret == DEFAULT_SECRET:
            for name, data in ep.files.items():
                with open(os.path.join(pub, name), "rb") as f:
                    same_as_published &= f.read() == data
        counts["issues"] += len(ep.truth["issues"])
        counts["clean"] += ep.truth["is_clean"]
    chk.record("%d episodes byte-identical on regeneration" % len(eps), det_ok,
               "%d issues, %d clean episodes" % (counts["issues"], counts["clean"]))
    if os.path.isdir(os.path.join(ROOT, "episodes")) and secret == DEFAULT_SECRET:
        chk.record("published episodes/ equal regenerated bytes", same_as_published)

    print("\n[3/7] independent re-audit (public records only) vs truth")
    n_extra = 20 if args.quick else 200
    targets = list(eps) + [("default", s, None) for s in range(n_extra)]
    bad, issues, t1, generated = [], 0, time.time(), []
    for split, seed, cfg in targets:
        ep = generate(seed, cfg, secret)
        generated.append(ep)
        rep = reaudit(ep)
        issues += rep["tp"] + rep["fn"]
        if not rep["ok"]:
            bad.append((split, seed, rep["tp"], rep["fp"], rep["fn"], rep["evidence_problems"][:2], rep["unexplained"][:2]))
    for b in bad[:10]:
        print("    mismatch:", b)
    chk.record("re-audit == truth on %d episodes" % len(targets), not bad,
               "%d issues; exact match, evidence sets sufficient and minimal, nothing unexplained; %.1fs" % (
                   issues, time.time() - t1))

    print("\n[4/7] baseline agents as isolated processes, grader in its own process, eval mode")
    traj_dir = os.path.join(out_dir, "trajectories")
    t1 = time.time()
    journals = run_many(ALL_POLICIES, eps, mode="eval", secret=secret, traj_dir=traj_dir, isolate=True,
                        grader="process")
    rows = summarize(journals)
    print_table(rows)
    by = {}
    for r in rows:
        by.setdefault(r["policy"], []).append(r)

    def mean(policy):
        rs = by[policy]
        return sum(r["reward_mean"] * r["episodes"] for r in rs) / sum(r["episodes"] for r in rs)
    chk.record("oracle reward == 1.0 on every episode",
               all(j["metrics"]["reward_main"] == 1.0 for j in journals if j["policy"] == "oracle"))
    chk.record("our auditor clearly beats the naive reconciler", mean("auditor") > mean("naive") + 0.5,
               "auditor %.3f vs naive %.3f" % (mean("auditor"), mean("naive")))
    chk.record("flag-all attack is bad", mean("flag_all") <= -0.5, "mean %.3f" % mean("flag_all"))
    truths = {ep.episode_id: ep.truth for ep in generated}
    problems, n_files = [], 0
    for split in ("train", "dev"):
        for name in sorted(os.listdir(os.path.join(traj_dir, split))):
            problems += scan_trajectory_file(os.path.join(traj_dir, split, name), truths, secrets=[secret])
            n_files += 1
    for p in problems[:5]:
        print("    leak:", p)
    chk.record("leak scan of agent trajectories (keys, ids, paths, secret, eval silence)", not problems,
               "%d files, %.1fs" % (n_files, time.time() - t1))
    write_json(os.path.join(out_dir, "summary.json"), rows)

    print("\n[5/7] statistical leak scan: do injected records differ in form from clean ones?")
    rep = stat_report(generated)
    for name, r in rep.items():
        if not name.startswith("_"):
            print("    %-62s D=%.4f p=%.4f" % (name, r["D"], r["p"]))
    chk.record("no statistical tell (Bonferroni, alpha 0.01)", all(r["ok"] for k, r in rep.items()
                                                                  if not k.startswith("_")))

    print("\n[6/7] official jury checks (genesis-exe.com/fintech-checks.html, 20.11.2026) through judge_adapter.py")
    from .official_checks import run_official
    off = run_official()
    chk.record("official blocks 1-4 all green", off["all_green"],
               "contract %d/20, closed episodes %d/20, matcher %d/20 (+M %d/12), pairs %d/20, engineering %d/10" % (
                   off["block1"], off["block2"], off["block3"], off["examples"], off["block4"], off["engineering"]))
    write_json(os.path.join(out_dir, "official.json"), off)

    print("\n[7/7] extended internal checks (docs/JURY_CHECKS.md, part 2)")
    from .jury_checks import run_all
    jr = run_all(quick=args.quick)
    n = jr["detail"]["episodes"]
    chk.record("internal checks A/B/C/D/G all green", jr["all_green"],
               "A %d/20, B %d/%d, C %d/20, D %d/%d, G %d/%d" % (
                   jr["A"], jr["B"], n, jr["C"], jr["D"], n, jr["G"], len(jr["detail"]["G"])))
    write_json(os.path.join(out_dir, "jury.json"), {k: v for k, v in jr.items()})

    passed = sum(1 for _, ok, _ in chk.results if ok)
    print("\nRESULT: %s  (%d/%d checks passed, %.1fs)" % ("PASS" if chk.ok else "FAIL", passed, len(chk.results),
                                                        time.time() - t0))
    return 0 if chk.ok else 1


def cmd_generate(args):
    secret = _read_secret(args)
    cfg = _load_json(args.config) if args.config else None
    for seed in _seeds(args.seeds):
        ep = generate(seed, cfg, secret)
        d = os.path.join(args.out, ep.episode_id)
        ep.write(d)
        print("seed %d -> %s (%d issues)" % (seed, d, len(ep.truth["issues"])))
    return 0


def cmd_export(args):
    out = os.path.join(ROOT, "episodes")
    shutil.rmtree(out, ignore_errors=True)
    index = []
    for split, seed, cfg in split_episodes():
        ep = generate(seed, cfg, DEFAULT_SECRET)
        ep.write(os.path.join(out, split, ep.episode_id))
        index.append({"split": split, "seed": seed, "episode_id": ep.episode_id, "config": cfg["name"],
                      "issues": len(ep.truth["issues"]), "is_clean": ep.truth["is_clean"],
                      "ledger_rows": ep.manifest["counts"]["ledger"], "documents": ep.manifest["counts"]["documents"],
                      "public_fingerprint": ep.public_fingerprint})
    write_json(os.path.join(out, "index.json"), index)
    print("exported %d episodes to episodes/" % len(index))
    return 0


def cmd_reaudit(args):
    secret = _read_secret(args)
    cfg = _load_json(args.config) if args.config else None
    bad, issues, n = 0, 0, 0
    for seed in _seeds(args.seeds):
        rep = reaudit(generate(seed, cfg, secret))
        n += 1
        issues += rep["tp"] + rep["fn"]
        if not rep["ok"]:
            bad += 1
            print("seed %d mismatch tp=%d fp=%d fn=%d %r %r" % (seed, rep["tp"], rep["fp"], rep["fn"],
                                                               rep["evidence_problems"][:3], rep["unexplained"][:3]))
    print("re-audit: %d episodes, %d issues, %d mismatches" % (n, issues, bad))
    return 0 if bad == 0 else 1


def cmd_evaluate(args):
    secret = _read_secret(args)
    cfg = _load_json(args.config) if args.config else None
    name = args.name or (os.path.splitext(os.path.basename(args.config))[0] if args.config else "default")
    eps = [(name, s, cfg) for s in _seeds(args.seeds)]
    t0 = time.time()
    if args.policy_cmd:
        import shlex
        from .runner import run_external
        cmd = shlex.split(args.policy_cmd)
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable
        cmd = [os.path.abspath(x) if os.path.exists(x) else x for x in cmd]  # the agent runs in an empty dir
        policies = ["external"]
        journals = run_external(cmd, "external", eps, secret=secret, traj_dir=args.traj_dir)
    else:
        policies = args.policies.split(",") if args.policies else ALL_POLICIES
        journals = run_many(policies, eps, mode="eval", secret=secret, traj_dir=args.traj_dir, isolate=args.isolate,
                            grader="process" if args.isolate else "inproc")
    print_table(summarize(journals))
    print("(%d episodes x %d policies, %.1fs)" % (len(eps), len(policies), time.time() - t0))
    if args.out:
        write_json(args.out, {"summary": summarize(journals), "journal": journals})
    return 0


def cmd_run(args):
    """Write trajectories of all policies on the open train/dev splits (train mode: real rewards)."""
    secret = _read_secret(args)
    tmp = os.path.join(ROOT, "build", "trajectories")
    shutil.rmtree(tmp, ignore_errors=True)
    journals = run_many(ALL_POLICIES, split_episodes(), mode=args.mode, secret=secret, traj_dir=tmp)
    merge_trajectories(tmp, os.path.join(ROOT, "trajectories.jsonl"))
    os.makedirs(os.path.join(ROOT, "trajectories"), exist_ok=True)
    shutil.copy(os.path.join(tmp, "privileged", "oracle.jsonl"),
                os.path.join(ROOT, "trajectories", "oracle.privileged.jsonl"))
    print_table(summarize(journals))
    print("wrote trajectories.jsonl and trajectories/oracle.privileged.jsonl")
    return 0


def cmd_report(args):
    from .report import build_report
    build_report(quick=args.quick)
    return 0


def cmd_schemas(args):
    from .schemas import write_schema_files
    write_schema_files(os.path.join(ROOT, "schemas"))
    print("schemas written to schemas/")
    return 0


def cmd_jury(args):
    from .jury_checks import run_all
    from .official_checks import run_official
    print("Official checks (genesis-exe.com/fintech-checks.html, 20.11.2026), through judge_adapter.py:")
    off = run_official()
    print("\nExtended internal checks (docs/JURY_CHECKS.md, part 2):")
    jr = run_all(quick=args.quick)
    return 0 if off["all_green"] and jr["all_green"] else 1


def cmd_play(args):
    """One episode driven over stdin: one JSON action per line, e.g.
    {"tool": "list_records", "args": {"table": "ledger", "limit": 5}}. `play` pretty-prints for a human,
    `serve` writes exactly one JSON object per line (for any external program)."""
    from .env import LedgerTrapEnv
    pretty = args.cmd == "play"
    env = LedgerTrapEnv(mode=args.mode, secret=_read_secret(args))
    cfg = _load_json(args.config) if args.config else None

    def emit(obj):
        if pretty:
            print(json.dumps(obj, ensure_ascii=False, indent=1))
            print("action> ", end="", file=sys.stderr, flush=True)
        else:
            print(json.dumps(obj, ensure_ascii=False, sort_keys=True), flush=True)
    try:
        obs, info = env.reset(args.seed, cfg)
        emit({"observation": obs, "info": info})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                action = json.loads(line)
            except ValueError:
                action = line  # the environment answers BAD_ACTION and spends a step
            obs, r, term, trunc, info = env.step(action)
            emit({"observation": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info})
            if term or trunc:
                break
    finally:
        env.close()
    return 0


def cmd_demo(args):
    from .demo import demo
    return demo(args.seed, args.config)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ledgertrap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="one-command verification")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--skip-tests", action="store_true")
    c.add_argument("--secret-file")
    c.set_defaults(func=cmd_check)
    j = sub.add_parser("jury", help="official jury checks, then our extended checks")
    j.add_argument("--quick", action="store_true")
    j.set_defaults(func=cmd_jury)
    g = sub.add_parser("generate", help="generate episodes into a directory")
    g.add_argument("--config")
    g.add_argument("--seeds", required=True, help="N, A-B or a comma list")
    g.add_argument("--out", required=True)
    g.add_argument("--secret-file")
    g.set_defaults(func=cmd_generate)
    x = sub.add_parser("export", help="write the open train/dev episodes to episodes/")
    x.set_defaults(func=cmd_export)
    u = sub.add_parser("run", help="write trajectories.jsonl for the open splits")
    u.add_argument("--mode", default="train", choices=["train", "eval"])
    u.add_argument("--secret-file")
    u.set_defaults(func=cmd_run)
    rp = sub.add_parser("report", help="write reports/diagnostics.md")
    rp.add_argument("--quick", action="store_true")
    rp.set_defaults(func=cmd_report)
    e = sub.add_parser("evaluate", help="run policies on seeds (eval mode) and print metrics")
    e.add_argument("--config")
    e.add_argument("--seeds", required=True)
    e.add_argument("--policies")
    e.add_argument("--name")
    e.add_argument("--out")
    e.add_argument("--traj-dir")
    e.add_argument("--isolate", action="store_true")
    e.add_argument("--policy-cmd", help="external agent command speaking the JSON-lines protocol")
    e.add_argument("--secret-file")
    e.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("reaudit", help="re-audit generated episodes against truth")
    r.add_argument("--config")
    r.add_argument("--seeds", required=True)
    r.add_argument("--secret-file")
    r.set_defaults(func=cmd_reaudit)
    s = sub.add_parser("schemas", help="write JSON schemas to schemas/")
    s.set_defaults(func=cmd_schemas)
    for name, text in (("play", "play one episode by hand: JSON actions on stdin, pretty output"),
                       ("serve", "one episode as a JSON-lines server on stdin/stdout")):
        pl = sub.add_parser(name, help=text)
        pl.add_argument("--seed", type=int, default=1000)
        pl.add_argument("--config")
        pl.add_argument("--mode", default="train", choices=["train", "eval"])
        pl.add_argument("--secret-file")
        pl.set_defaults(func=cmd_play)
    d = sub.add_parser("demo", help="one episode, naive vs auditor, explained")
    d.add_argument("--seed", type=int, default=1004)
    d.add_argument("--config", default="configs/train.json")
    d.set_defaults(func=cmd_demo)
    args = p.parse_args(argv)
    return args.func(args)
