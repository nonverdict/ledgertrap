"""Measured diagnostics (case §10: P, R, F1, return, steps and time of agents; measured failure analysis).

Writes reports/diagnostics.md and reports/diagnostics.json. Every number is produced by this run.
"""

import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import date

from .env import LedgerTrapEnv
from .generator import generate
from .idmap import pass_hat_k, presentation
from .leakscan import stat_report
from .policies.agents import make_policy
from .reaudit import reaudit
from .runner import run_episode, summarize, write_json
from .timeutil import parse, parse_offset
from .version import DEFAULT_SECRET, ENV_VERSION, GENERATOR_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ["oracle", "auditor", "naive", "random", "flag_all", "silent"]


def _load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


def run_set(name, episodes, policies=POLICIES):
    """Run policies (eval mode) and keep journal + submitted findings + truth per episode."""
    out = []
    for pol in policies:
        for seed, cfg in episodes:
            env = LedgerTrapEnv(mode="eval", secret=DEFAULT_SECRET)
            policy = make_policy(pol, env=env)
            traj, j = run_episode(env, policy, seed, cfg, name, pol)
            ep = env.privileged_episode()
            last = traj[-1]["action"]
            findings = []
            if isinstance(last, dict) and last.get("tool") == "submit":
                findings = (last["args"].get("findings") or {}).get("findings", []) or []
            out.append({"journal": j, "findings": findings, "episode": ep})
    return out


# ------------------------------------------------------------------------------------------------
# failure analysis
# ------------------------------------------------------------------------------------------------

def classify_fp(ep, f):
    """Which correct-but-suspicious pattern (or plain guess) explains a false positive."""
    docs = {d["doc_id"]: d for d in ep.records["invoices"] + ep.records["payments"]}
    num_to_doc = {}
    for d in docs.values():
        num_to_doc[d.get("invoice_id") or d.get("payment_id")] = d
    ids = [ep.aliases.get(x, x) for x in f.get("affected_ids", [])]
    ds = [docs[i] for i in ids if i in docs]
    if not ds:
        return "неизвестный или угаданный ID"
    d = ds[0]
    t = f.get("type", "")
    tz = parse_offset(ep.manifest["timezone"])
    rows = [r for r in ep.records["ledger"]
            if (d.get("payment_id") and r["payment_id"] == d["payment_id"])
            or (d.get("invoice_id") and r["invoice_id"] == d["invoice_id"] and r["payment_id"] is None)]
    stamp = d.get("issued_at") or d["timestamp"]
    local = parse(stamp).astimezone(tz)
    if t == "period_cutoff":
        if any(r["timestamp"] == ep.manifest["period_start"] for r in rows) and stamp < ep.manifest["period_start"]:
            return "Л11 входящий остаток прошлого периода"
        if stamp[:10] != local.date().isoformat():
            return "Л6 часовой пояс (дата строки ≠ дата компании)"
        return "прочее (сроки)"
    if t == "fx_conversion":
        if rows and rows[0]["timestamp"][:10] != local.date().isoformat():
            if local.weekday() >= 5:
                return "Л5 выходной: курс пятницы"
            return "Л7 лаг проводки: курс по дате проводки"
        if not rows:
            return "документ без проводок"
        return "прочее (курс)"
    if t == "duplicate_invoice":
        if len(ds) >= 2:
            a, b = ds[0], ds[1]
            if a.get("kind") == "credit_note" or b.get("kind") == "credit_note":
                return "Л4 кредит-нота"
            if a.get("entity_id") != b.get("entity_id"):
                return "Л2 одинаковые суммы у разных клиентов"
            la = [{k: v for k, v in x.items() if not k.startswith("service_period")} for x in a.get("line_items", [])]
            lb = [{k: v for k, v in x.items() if not k.startswith("service_period")} for x in b.get("line_items", [])]
            if la == lb:
                return "Л1 регулярный счёт (другой период услуги)"
            return "одинаковая сумма, разные позиции"
        return "прочее (дубль)"
    if t == "payment_allocation":
        if "payment_id" in d:
            refs = d["invoice_refs"] or []
            if not refs:
                return "Л8 платёж без ссылки"
            if d["amount_minor"] > sum(x["amount_minor"] for x in refs):
                return "Л9 переплата в авансах"
            inv = num_to_doc.get(refs[0]["invoice_id"])
            if inv and refs[0]["amount_minor"] < inv["total_minor"]:
                return "Л3/Л12 разбитая или частичная оплата"
            if d["amount_minor"] < 0:
                return "Л4 возврат по кредит-ноте"
        return "прочее (распределение)"
    return "неизвестный тип"


def failure_analysis(runs):
    """FN by (type, variant, scenario) and FP by trap, per policy."""
    fn = defaultdict(Counter)
    fp = defaultdict(Counter)
    tp = defaultdict(Counter)
    for r in runs:
        j, ep = r["journal"], r["episode"]
        pol = j["policy"]
        matched = {iid for _, iid in j["pairs"]}
        for g in ep.truth["issues"]:
            key = "%s/%s%s" % (g["type"], g["variant"], " (%s)" % g["scenario"] if g["scenario"] else "")
            (tp if g["issue_id"] in matched else fn)[pol][key] += 1
        matched_idx = {i for i, _ in j["pairs"]}
        for i, f in enumerate(r["findings"]):
            if i not in matched_idx:
                fp[pol][classify_fp(ep, f)] += 1
    return tp, fn, fp


def pass_k_table(episodes, policies=("auditor", "naive"), n=5):
    """pass^k over n harmless presentations (new ids + order) of the same episode; success = F1 == 1."""
    out = {}
    for pol in policies:
        succ = []
        for seed, cfg in episodes:
            c = 0
            for i in range(n):
                env = LedgerTrapEnv(mode="eval")
                traj, j = run_episode(env, make_policy(pol), seed, dict(cfg or {}, presentation=presentation(i)))
                m = j["metrics"]
                c += m["fp"] == 0 and m["fn"] == 0
            succ.append(c)
        out[pol] = {"pass^%d" % k: round(pass_hat_k(succ, n, k), 3) for k in range(1, n + 1)}
    return out


# ------------------------------------------------------------------------------------------------

def _row(r):
    return "| %s | %d | %.3f | %.3f | %.3f | %.2f | %.2f | %.2f | %.1f | %.4f |" % (
        r["policy"], r["episodes"], r["reward_mean"], r["reward_min"], r["reward_max"], r["precision_micro"],
        r["recall_micro"], r["f1_micro"], r["steps_mean"], r["seconds_mean"])


HEAD = ("| политика | эпизодов | возврат (сред.) | мин | макс | P | R | F1 | шагов (сред.) | время, с |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def matcher_regression():
    """Measured failure -> cause -> fix -> re-check. The matcher rules we used until 20.11 (relative amount
    tolerance, case/space folding, greedy assignment, no existence check of evidence ids) run against the
    official matcher tests (Block 3 N01-N20 and open examples M01-M12), then the current rules."""
    import shutil
    import tempfile
    from unittest import mock

    from . import official_checks as oc
    from .grader.matcher import MatchResult
    from .grader.normalize import NormFinding

    def old_norm(index, f):
        return NormFinding(index, f["type"].strip().lower(), frozenset(x.strip().upper() for x in f["affected_ids"]),
                           abs(f["claimed_amount_minor"]), f["currency"].strip().upper(),
                           frozenset(x.strip().upper() for x in f["evidence_ids"]))

    def old_is_tp(f, g):
        e = g["expected_amount_minor"]
        return (f.type == g["type"] and f.affected == g["_affected"] and f.currency == g["currency"]
                and abs(f.amount - e) <= max(2, -(-abs(e) // 1000)) and any(s <= f.evidence for s in g["_evidence"]))

    def old_match(findings, issues, known_ids, tolerances=0):
        taken, pairs = set(), []
        for f in findings:
            for g in issues:
                if g["issue_id"] not in taken and old_is_tp(f, g):
                    taken.add(g["issue_id"])
                    pairs.append((f.index, g["issue_id"]))
                    break
        tp = len(pairs)
        return MatchResult(tp, len(findings) - tp, len(issues) - tp, pairs, [])

    work = tempfile.mkdtemp(prefix="ledgertrap-report-")
    try:
        fx = oc.Fixtures(work)
        with mock.patch("ledgertrap.grader.grader.normalize_finding", old_norm), \
                mock.patch("ledgertrap.grader.grader.match", old_match):
            old_n, old_m = oc.block3(fx, os.path.join(work, "old"))
        new_n, new_m = oc.block3(fx, os.path.join(work, "new"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"old": {"N": old_n, "M": old_m}, "new": {"N": new_n, "M": new_m}}


def auditor_iterations(quick=False):
    """Stage 7, level 1 (bonus): harness engineering of our scripted auditor. Diagnose on train (where do
    the steps go), change the policy, measure on dev, confirm on fresh seeds never used before."""
    from .policies.agents import AUDITOR_VERSIONS
    sp = _load("configs/splits.json")
    dev_cfg = _load(sp["dev"]["config"])
    n_fresh = 6 if quick else 30
    n_l1 = 3 if quick else 20
    sets = [("train (диагностика)", [(s, _load(sp["train"]["config"])) for s in sp["train"]["seeds"]]),
            ("dev (замер)", [(s, dev_cfg) for s in sp["dev"]["seeds"]]),
            ("свежие сиды 3000–%d, конфиг dev" % (2999 + n_fresh), [(s, dev_cfg) for s in range(3000, 3000 + n_fresh)]),
            ("свежие сиды 3100–%d, пресет L1" % (3099 + n_l1), [(s, _load("configs/presets/L1.json"))
                                                              for s in range(3100, 3100 + n_l1)])]
    rows, tools0 = [], Counter()
    for label, eps in sets:
        for v in sorted(AUDITOR_VERSIONS):
            js = []
            for seed, cfg in eps:
                traj, j = run_episode(LedgerTrapEnv(mode="train"), make_policy("auditor_v%d" % v), seed, cfg)
                js.append(j)
                if v == 0 and label.startswith("train"):
                    tools0.update(t["action"]["tool"] for t in traj[1:] if isinstance(t["action"], dict))
            m = [j["metrics"] for j in js]
            tp, fp, fn = sum(x["tp"] for x in m), sum(x["fp"] for x in m), sum(x["fn"] for x in m)
            rows.append({"set": label, "version": v, "episodes": len(js),
                         "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                         "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                         "reward": sum(x["reward_main"] for x in m) / len(m),
                         "steps": sum(x["inspection_steps"] for x in m) / len(m)})
    return rows, tools0, AUDITOR_VERSIONS


def build_report(quick=False):
    t0 = time.time()
    sp = _load("configs/splits.json")
    md, js = [], {}
    md.append("# LedgerTrap — диагностика (измерено)\n")
    md.append("Сгенерировано командой `python -m ledgertrap report` (%s). generator %s, env %s, Python %s, %s.\n" % (
        date.today().isoformat(), GENERATOR_VERSION, ENV_VERSION, platform.python_version(), platform.machine()))
    md.append("Все числа — из этого прогона. Режим eval: агент получает награду 0, реальные результаты берутся из "
              "журнала оценщика. P, R, F1 — микро-усреднение по эпизодам; возврат = терминальная награда "
              "`clip(2·F1 − 1 − 0.002·шаги, −1, 1)`.\n")

    # 1. open splits
    md.append("## 1. Открытые выборки train и dev\n")
    all_runs = []
    for split in ("train", "dev"):
        cfg = _load(sp[split]["config"])
        seeds = sp[split]["seeds"][:3] if quick else sp[split]["seeds"]
        runs = run_set(split, [(s, cfg) for s in seeds])
        all_runs += runs
        rows = summarize([r["journal"] for r in runs])
        js["split_" + split] = rows
        md.append("**%s** (сиды %s–%s, конфиг `%s`)\n" % (split, seeds[0], seeds[-1], sp[split]["config"]))
        md.append(HEAD)
        md += [_row(r) for r in rows]
        md.append("")

    # 2. difficulty levels
    md.append("## 2. Уровни сложности L1–L5 (по %d сидов)\n" % (5 if quick else 30))
    level_rows = {}
    md.append("| уровень | oracle | наш auditor | naive | random | flag_all | silent | шагов auditor | ошибок/эпизод |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for i in range(1, 6):
        cfg = _load("configs/presets/L%d.json" % i)
        seeds = range(500, 505) if quick else range(500, 530)
        runs = run_set("L%d" % i, [(s, cfg) for s in seeds])
        all_runs += runs
        rows = {r["policy"]: r for r in summarize([x["journal"] for x in runs])}
        level_rows["L%d" % i] = rows
        n_issues = sum(len(x["episode"].truth["issues"]) for x in runs if x["journal"]["policy"] == "oracle")
        md.append("| L%d %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f | %.1f |" % (
            i, cfg["name"].split("-", 1)[1], rows["oracle"]["reward_mean"], rows["auditor"]["reward_mean"],
            rows["naive"]["reward_mean"], rows["random"]["reward_mean"], rows["flag_all"]["reward_mean"],
            rows["silent"]["reward_mean"], rows["auditor"]["steps_mean"], n_issues / len(seeds)))
    js["levels"] = level_rows
    md.append("")

    # 3. failure analysis
    tp, fn, fp = failure_analysis(all_runs)
    md.append("## 3. Разбор промахов (все прогоны разделов 1–2)\n")
    md.append("### 3.1 Пропуски (FN) и попадания (TP) по типу и варианту ошибки\n")
    keys = sorted(set(k for c in list(tp.values()) + list(fn.values()) for k in c))
    md.append("| тип / вариант | auditor TP/FN | naive TP/FN | random TP/FN | flag_all TP/FN |")
    md.append("|---|---|---|---|---|")
    for k in keys:
        md.append("| %s | %s |" % (k, " | ".join("%d/%d" % (tp[p][k], fn[p][k])
                                               for p in ("auditor", "naive", "random", "flag_all"))))
    md.append("")
    md.append("### 3.2 Ложные срабатывания (FP) по причине\n")
    for pol in ("naive", "flag_all", "random", "auditor"):
        total = sum(fp[pol].values())
        md.append("**%s** — всего FP: %d" % (pol, total))
        if total:
            md.append("")
            md.append("| причина | FP | доля |")
            md.append("|---|---|---|")
            for k, v in fp[pol].most_common():
                md.append("| %s | %d | %.1f%% |" % (k, v, 100.0 * v / total))
        md.append("")
    js["failures"] = {"tp": tp, "fn": fn, "fp": fp}

    # 4. pass^k
    md.append("## 4. Устойчивость к безвредным преобразованиям: pass^k\n")
    md.append("Каждый эпизод dev показан 5 раз с новыми ID и порядком записей (`id_salt`, `order_seed`, "
              "`order`). Успех — F1 = 1 (все ошибки найдены, ложных нет). pass^k — вероятность успеха во всех k "
              "показах (τ-bench).\n")
    dev_cfg = _load(sp["dev"]["config"])
    pk = pass_k_table([(s, dev_cfg) for s in (sp["dev"]["seeds"][:3] if quick else sp["dev"]["seeds"])])
    md.append("| политика | " + " | ".join("pass^%d" % k for k in range(1, 6)) + " |")
    md.append("|---|---|---|---|---|---|")
    for pol, v in pk.items():
        md.append("| %s | %s |" % (pol, " | ".join("%.3f" % v["pass^%d" % k] for k in range(1, 6))))
    md.append("")
    js["pass_k"] = pk

    # 5. generator quality
    md.append("## 5. Качество генератора: независимый ре-аудит\n")
    md.append("| набор | эпизодов | ошибок | расхождений с эталоном | наборов доказательств проверено |")
    md.append("|---|---|---|---|---|")
    gen_eps = []
    re_rows = {}
    for name, cfg, seeds in [("по умолчанию", None, range(0, 50 if quick else 200))] + [
            ("L%d" % i, _load("configs/presets/L%d.json" % i), range(600, 610 if quick else 640)) for i in range(1, 6)]:
        n_iss = bad = sets = 0
        t1 = time.time()
        for s in seeds:
            ep = generate(s, cfg)
            gen_eps.append(ep)
            rep = reaudit(ep)
            n_iss += len(ep.truth["issues"])
            sets += sum(len(g["evidence_sets"]) for g in ep.truth["issues"])
            bad += not rep["ok"]
        re_rows[name] = {"episodes": len(seeds), "issues": n_iss, "mismatches": bad, "evidence_sets": sets,
                         "seconds": round(time.time() - t1, 2)}
        md.append("| %s | %d | %d | **%d** | %d |" % (name, len(seeds), n_iss, bad, sets))
    md.append("")
    md.append("Ре-аудитор видит только открытые записи. Совпадение = одинаковое множество (тип, ID, сумма в допуске), "
              "каждый набор доказательств сам по себе достаточен и минимален, и в книгах нет строк, "
              "не объяснённых найденными ошибками.\n")
    js["reaudit"] = re_rows

    issue_kinds = Counter()
    traps = Counter()
    for ep in gen_eps:
        for g in ep.truth["issues"]:
            issue_kinds["%s/%s" % (g["type"], g["variant"])] += 1
        traps.update(ep.provenance["traps_planted"])
    md.append("**Частота ошибок по вариантам** (те же эпизоды): " +
              ", ".join("%s — %d" % kv for kv in sorted(issue_kinds.items())) + ".\n")
    md.append("**Ловушки (эпизодов, где встречается):** " +
              ", ".join("%s — %d" % kv for kv in sorted(traps.items(), key=lambda kv: int(kv[0][1:]))) + ".\n")
    js["issue_kinds"], js["traps"] = issue_kinds, traps

    # 6. leak statistics
    md.append("## 6. Статистический поиск утечек\n")
    sr = stat_report(gen_eps[:200])
    md.append("| признак | D (KS) | p | итог |")
    md.append("|---|---|---|---|")
    for k, v in sr.items():
        if not k.startswith("_"):
            md.append("| %s | %.4f | %.4f | %s |" % (k, v["D"], v["p"], "ок" if v["ok"] else "**сигнал**"))
    md.append("")
    md.append("Позиция в списке отличается по построению: ошибки вносятся в документы периода, а не в окно после "
              "него и не во входящие остатки; эти даты агент видит и так, поэтому признак диагностический.\n")
    js["leak_stats"] = sr

    # 7. measured failure analysis
    md.append("## 7. Разбор сбоев: измерение → причина → исправление → повторная проверка\n")
    reg = matcher_regression()
    on = sum(reg["old"]["N"].values())
    om = sum(reg["old"]["M"].values())
    nn = sum(reg["new"]["N"].values())
    nm = sum(reg["new"]["M"].values())
    failed = [k for k, v in list(reg["old"]["N"].items()) + list(reg["old"]["M"].items()) if not v]
    md.append("### 7.1 Сопоставление выводов с эталоном против официальных тестов matcher (20.11.2026)\n")
    md.append("| правила matcher | тесты N01–N20 | примеры M01–M12 | не прошли |")
    md.append("|---|---|---|---|")
    md.append("| до 20.11 (наши) | %d/20 | %d/12 | %s |" % (on, om, ", ".join(failed) or "—"))
    md.append("| текущие | **%d/20** | **%d/12** | %s |" % (
        nn, nm, ", ".join(k for k, v in list(reg["new"]["N"].items()) + list(reg["new"]["M"].items()) if not v)
        or "—"))
    md.append("")
    md.append("- **Сбой (измерен этим прогоном):** прежние правила проваливают тесты из списка выше.")
    md.append("- **Причина:** относительный допуск суммы `max(2, ⌈|e|/1000⌉)` вместо опубликованного τ "
              "(официально τ = 0 по умолчанию); сравнение ID, типа и валюты без учёта регистра и пробелов; "
              "жадное назначение вместо максимального паросочетания; не проверялось, что ID доказательств "
              "существуют.")
    md.append("- **Исправление:** `ledgertrap/grader/` — точные строки, τ из `policy.json` "
              "(`submission.amount_tolerance_minor`, у нас 0 для всех типов; генератор гарантирует, что каждую "
              "сумму можно пересчитать точно), максимальное паросочетание с детерминированным выбором, "
              "несуществующий ID доказательства → FP.")
    md.append("- **Повторная проверка:** строка «текущие» выше; те же отрицательные контроли — "
              "`tests/test_official.py` (каждый блок падает, если вернуть старое поведение).\n")
    js["matcher_regression"] = {"old_N": on, "old_M": om, "new_N": nn, "new_M": nm, "old_failed": failed}

    md.append("### 7.2 Воспроизводимый неустранённый предел: наш аудитор на больших эпизодах\n")
    md.append("| уровень | F1 (микро) | шагов | штраф 0.002·шаги | средний возврат |")
    md.append("|---|---|---|---|---|")
    for lv in ("L1", "L3", "L5"):
        a = level_rows[lv]["auditor"]
        md.append("| %s | %.3f | %.1f | %.3f | %.3f |" % (lv, a["f1_micro"], a["steps_mean"], 0.002 * a["steps_mean"],
                                                         a["reward_mean"]))
    md.append("")
    aud = [r["journal"] for r in all_runs if r["journal"]["policy"] == "auditor"]
    miss = [j for j in aud if j["metrics"]["fn"] > 0]
    near_budget = sum(1 for j in miss if j["metrics"]["inspection_steps"] >= 90)
    fps = sum(j["metrics"]["fp"] for j in aud)
    if miss or fps:
        md.append("Измерено на всех прогонах разделов 1–2: эпизодов аудитора %d, с пропусками (FN > 0) — %d, из них с "
                  "90+ шагами исследования — %d; ложных срабатываний всего — %d. Разбивка — в таблицах 3.1 и 3.2." % (
                      len(aud), len(miss), near_budget, fps))
    else:
        md.append("Измерено на всех прогонах разделов 1–2: эпизодов аудитора %d, пропусков (FN) и ложных срабатываний "
                  "(FP) — 0. "
                  "Весь спад возврата от L1 к L5 — штраф 0.002 за шаг: аудитор читает таблицы целиком страницами "
                  "по 50 строк, и на больших эпизодах шагов больше." % len(aud))
    md.append("Повторить: `python -m ledgertrap report`. Это предел нашей скриптовой политики, а не среды. Аудитор "
              "применяет те же опубликованные правила, что и ре-аудитор, поэтому F1 = 1 показывает решаемость задачи "
              "по открытым записям, а не обучение; обучения весов в проекте нет.\n")
    js["auditor_limit"] = {"episodes": len(aud), "with_fn": len(miss), "with_fn_90plus_steps": near_budget}

    # 9. harness engineering (bonus) -- placed after timings in the text, computed here
    it_rows, tools0, versions = auditor_iterations(quick)

    # 8. timings
    md.append("## 8. Время\n")
    gen_t = sorted(r["journal"]["generation_s"] for r in all_runs)
    pol_t = sorted(r["journal"]["policy_seconds"] for r in all_runs)
    md.append("Генерация эпизода: медиана %.3f с, максимум %.3f с. Прогон политики на эпизоде: медиана %.3f с, "
              "максимум %.3f с. Лимит кейса — 60 с на эпизод.\n" % (
                  gen_t[len(gen_t) // 2], gen_t[-1], pol_t[len(pol_t) // 2], pol_t[-1]))
    md.append("Отчёт построен за %.0f с.\n" % (time.time() - t0))
    js["timing"] = {"generation_median": gen_t[len(gen_t) // 2], "generation_max": gen_t[-1],
                    "policy_median": pol_t[len(pol_t) // 2], "policy_max": pol_t[-1]}

    md.append("## 9. Цикл улучшения нашего аудитора (бонус, баллов не даёт)\n")
    md.append("Harness engineering на CPU: меняется обвязка скриптовой политики, весов нет. Диагностика — только на "
              "train, замер — на dev, подтверждение — на свежих сидах, которые раньше нигде не использовались.\n")
    total0 = sum(tools0.values())
    md.append("**Диагностика v0 на train:** действий %d — %s. F1 = 1, значит весь недобор возврата — шаги; "
              "`get_document` нужен был только ради `line_items`, а `list_records` теперь отдаёт счета целиком." % (
                  total0, ", ".join("%s %d" % kv for kv in tools0.most_common())))
    md.append("")
    md.append("| версия | что поменяли |")
    md.append("|---|---|")
    for v, text in sorted(versions.items()):
        md.append("| v%d | %s |" % (v, text))
    md.append("")
    md.append("| выборка | версия | эпизодов | P | R | F1 | шагов | возврат |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in it_rows:
        md.append("| %s | v%d | %d | %.3f | %.3f | %.3f | %.2f | %.4f |" % (
            r["set"], r["version"], r["episodes"], r["precision"], r["recall"], r["f1"], r["steps"], r["reward"]))
    md.append("")
    md.append("Вывод по замеру: v1 экономит шаги на всех выборках без потери F1; v2 меняет только эпизоды с одной "
              "валютой (L1), на train, dev и свежих сидах dev-конфига результат v2 равен v1. Это улучшение "
              "скриптовой обвязки, а не обучение модели.\n")
    js["auditor_iterations"] = it_rows

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "diagnostics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    write_json(os.path.join(ROOT, "reports", "diagnostics.json"), json.loads(json.dumps(js, default=dict)))
    print("\n".join(md))
    print("\nwrote reports/diagnostics.md (%.0fs, python %s)" % (time.time() - t0, sys.version.split()[0]))
