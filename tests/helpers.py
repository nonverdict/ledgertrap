"""Shared fixtures for tests."""

import functools

from ledgertrap.env import LedgerTrapEnv
from ledgertrap.generator import generate
from ledgertrap.policies.baselines import oracle_findings

DIRTY = {"errors": {"mode": "exact", "exact": {"duplicate_invoice:document": 1,
                                                "duplicate_invoice:double_posting": 1,
                                                "period_cutoff:early": 1, "period_cutoff:late": 1}}}
CLEAN = {"errors": {"mode": "clean"}}
ALL_VARIANTS = {"errors": {"mode": "exact", "exact": {
    "duplicate_invoice:document": 1, "duplicate_invoice:double_posting": 1,
    "payment_allocation:missing": 1, "payment_allocation:partial": 1, "payment_allocation:wrong_invoice": 1,
    "fx_conversion:wrong_date": 1, "fx_conversion:wrong_rate": 1,
    "period_cutoff:early": 1, "period_cutoff:late": 1}},
    "size": {"documents": [100, 160], "ledger_rows": [400, 900]},
    "company": {"n_foreign_currencies": [2, 3]}}


@functools.lru_cache(maxsize=64)
def episode(seed, kind="dirty"):
    cfg = {"dirty": DIRTY, "clean": CLEAN, "default": None, "all": ALL_VARIANTS}[kind]
    return generate(seed, cfg)


def env_for(seed, kind="dirty", mode="train", clock=None):
    cfg = {"dirty": DIRTY, "clean": CLEAN, "default": None}[kind]
    env = LedgerTrapEnv(mode=mode) if clock is None else LedgerTrapEnv(mode=mode, clock=clock)
    obs, info = env.reset(seed, cfg)
    return env, obs, info


def submit(env, findings_obj):
    return env.step({"tool": "submit", "args": {"findings": findings_obj}})


def oracle_obj(env):
    return oracle_findings(env.privileged_episode())


def finding(t, affected, amount, cur, evidence, explanation="x"):
    return {"type": t, "affected_ids": list(affected), "claimed_amount_minor": amount, "currency": cur,
            "evidence_ids": list(evidence), "explanation": explanation}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t
