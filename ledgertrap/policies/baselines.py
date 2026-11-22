"""Simple scripted policies used by `check` and by reward-property tests.

- oracle (grader side): submits truth with the first evidence set at step 1 -> reward 1.0
- silent: submits an empty findings list immediately
- flag_all: fills 30 findings over documents and types, evidence = the document itself
"""

ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")


def oracle_findings(episode):
    """Grader-side only: needs truth. Never available to agents."""
    return {"episode_id": episode.episode_id, "findings": [
        {"type": g["type"], "affected_ids": list(g["affected_ids"]),
         "claimed_amount_minor": g["expected_amount_minor"], "currency": g["currency"],
         "evidence_ids": list(g["evidence_sets"][0]), "explanation": "oracle"}
        for g in episode.truth["issues"]]}


def submit_action(findings_obj):
    return {"tool": "submit", "args": {"findings": findings_obj}}


def silent_action(observation):
    return submit_action({"episode_id": observation["manifest"]["episode_id"], "findings": []})


def flag_all_findings(episode_id, invoices, payments):
    docs = [(d["doc_id"], abs(d["total_minor"]), d["currency"]) for d in invoices]
    docs += [(d["doc_id"], abs(d["amount_minor"]), d["currency"]) for d in payments]
    findings = []
    i = 0
    while len(findings) < 30 and docs:
        doc_id, amount, cur = docs[i % len(docs)]
        t = ISSUE_TYPES[(i // len(docs)) % len(ISSUE_TYPES)] if len(docs) < 30 else ISSUE_TYPES[i % 4]
        findings.append({"type": t, "affected_ids": [doc_id], "claimed_amount_minor": amount, "currency": cur,
                         "evidence_ids": [doc_id], "explanation": "flag all"})
        i += 1
    return {"episode_id": episode_id, "findings": findings}


def run_episode(env, seed, config, policy):
    """Run a policy in-process (train/dev convenience). policy(env, obs) -> list of actions or None.

    Returns (final_reward, info, n_steps, trajectory)."""
    obs, info = env.reset(seed, config)
    traj = [{"step": 0, "action": None, "observation": obs, "reward": 0.0, "terminated": False,
             "truncated": False}]
    actions = policy(env, obs)
    reward, term, trunc, last_info = 0.0, False, False, info
    for n, a in enumerate(actions, start=1):
        obs, reward, term, trunc, last_info = env.step(a)
        traj.append({"step": n, "action": a, "observation": obs, "reward": reward, "terminated": term,
                     "truncated": trunc})
        if term or trunc:
            break
    return reward, last_info, len(traj) - 1, traj


def oracle_policy(env, obs):
    return [submit_action(oracle_findings(env.privileged_episode()))]


def silent_policy(env, obs):
    return [silent_action(obs)]
