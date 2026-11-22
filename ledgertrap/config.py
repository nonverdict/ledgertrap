"""Versioned episode config: defaults, strict validation, hashing, resolution."""

import copy

from .canon import canonical_bytes, sha256_hex
from .money import CURRENCY_EXP
from .version import CONFIG_VERSION


class ConfigError(ValueError):
    pass


ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
VARIANTS = {
    "duplicate_invoice": ("document", "double_posting"),
    "payment_allocation": ("missing", "partial", "wrong_invoice"),
    "fx_conversion": ("wrong_date", "wrong_rate"),
    "period_cutoff": ("early", "late"),
}
IMPLEMENTED_TYPES = ISSUE_TYPES
SCENARIOS = ("C1", "C2")
IMPLEMENTED_SCENARIOS = SCENARIOS

TRAPS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12")
TIMEZONES = ("+05:00", "+03:00", "+01:00", "+00:00", "-05:00", "+08:00")
CHART_TEMPLATES = ("kz_standard", "ifrs_numeric", "alpha_codes")
BASE_CURRENCIES = ("KZT", "USD", "EUR")

DEFAULTS = {
    "config_version": CONFIG_VERSION,
    "name": "default",
    "company": {
        "base_currency": list(BASE_CURRENCIES),
        "foreign_currencies": ["USD", "EUR", "CNY", "RUB", "GBP", "JPY", "KWD"],
        "n_foreign_currencies": [1, 3],
        "timezone": list(TIMEZONES),
        "doc_offset_same_prob": 0.7,
        "chart_template": list(CHART_TEMPLATES),
        "rounding": ["HALF_UP", "HALF_EVEN"],
        "period_kind": ["month", "quarter"],
        "period_year": 2026,
        "window_days": 10,
    },
    "size": {"ledger_rows": [200, 800], "documents": [20, 120], "customers": [5, 25]},
    "errors": {
        "mode": "random",
        "clean_probability": 0.25,
        "count": [1, 8],
        "exact": None,
        "type_weights": {t: 1 for t in ISSUE_TYPES},
        "variant_weights": {t: {v: 1 for v in VARIANTS[t]} for t in ISSUE_TYPES},
        "interacting": {"C1": 0.3, "C2": 0.3},
    },
    "traps": {"rate": 0.6, "enabled": list(TRAPS)},
    "presentation": {"id_salt": 0, "order_seed": 0, "order": "time"},
    "limits": {"max_steps": 100, "page_size": 50, "time_limit_s": 60},
    "source": None,
}

# Keys whose dict values are free-form (validated separately), not merged recursively.
_OPAQUE = {("errors", "exact"), ("source",)}
# Two-element lists interpreted as inclusive [min, max] ranges.
RANGE_KEYS = {("size", "ledger_rows"), ("size", "documents"), ("size", "customers"),
              ("errors", "count"), ("company", "n_foreign_currencies")}


def _merge(base, override, path=()):
    if not isinstance(override, dict):
        raise ConfigError("config%s must be an object" % _p(path))
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in base:
            raise ConfigError("unknown config key %s" % _p(path + (k,)))
        sub = path + (k,)
        if isinstance(base[k], dict) and sub not in _OPAQUE and sub not in (
                ("errors", "type_weights"), ("errors", "variant_weights")):
            out[k] = _merge(base[k], v, sub)
        elif sub in (("errors", "type_weights"), ("errors", "variant_weights")):
            out[k] = _merge_weights(base[k], v, sub)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_weights(base, override, path):
    if not isinstance(override, dict):
        raise ConfigError("%s must be an object" % _p(path))
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in base:
            raise ConfigError("unknown config key %s" % _p(path + (k,)))
        if isinstance(base[k], dict):
            out[k] = _merge_weights(base[k], v, path + (k,))
        else:
            out[k] = v
    return out


def _p(path):
    return "." + ".".join(path) if path else ""


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool))


def _check_range(cfg, path, lo, hi):
    v = cfg
    for k in path:
        v = v[k]
    if _is_int(v):
        v = [v, v]
    if not (isinstance(v, list) and len(v) == 2 and all(_is_int(x) for x in v) and lo <= v[0] <= v[1] <= hi):
        raise ConfigError("%s must be [min,max] within [%d,%d]" % (_p(path), lo, hi))


def _check_choice(value, allowed, path):
    vals = value if isinstance(value, list) else [value]
    if not vals or any(v not in allowed for v in vals):
        raise ConfigError("%s must be one of %s" % (_p(path), list(allowed)))


def validate(cfg: dict) -> None:
    if cfg.get("config_version") != CONFIG_VERSION:
        raise ConfigError("config_version must be %r" % CONFIG_VERSION)
    if not isinstance(cfg.get("name"), str):
        raise ConfigError(".name must be a string")
    c = cfg["company"]
    _check_choice(c["base_currency"], BASE_CURRENCIES, ("company", "base_currency"))
    _check_choice(c["foreign_currencies"], tuple(CURRENCY_EXP), ("company", "foreign_currencies"))
    _check_range(cfg, ("company", "n_foreign_currencies"), 0, 7)
    _check_choice(c["timezone"], TIMEZONES, ("company", "timezone"))
    _check_choice(c["chart_template"], CHART_TEMPLATES, ("company", "chart_template"))
    _check_choice(c["rounding"], ("HALF_UP", "HALF_EVEN"), ("company", "rounding"))
    _check_choice(c["period_kind"], ("month", "quarter"), ("company", "period_kind"))
    if not (_is_num(c["doc_offset_same_prob"]) and 0 <= c["doc_offset_same_prob"] <= 1):
        raise ConfigError(".company.doc_offset_same_prob must be in [0,1]")
    if not (_is_int(c["period_year"]) and 2000 <= c["period_year"] <= 2100):
        raise ConfigError(".company.period_year must be an int year")
    if not (_is_int(c["window_days"]) and 3 <= c["window_days"] <= 31):
        raise ConfigError(".company.window_days must be in [3,31]")
    _check_range(cfg, ("size", "ledger_rows"), 200, 2000)
    _check_range(cfg, ("size", "documents"), 20, 200)
    _check_range(cfg, ("size", "customers"), 3, 60)
    e = cfg["errors"]
    _check_choice(e["mode"], ("random", "clean", "exact"), ("errors", "mode"))
    if isinstance(e["mode"], list):
        raise ConfigError(".errors.mode must be a single value")
    if not (_is_num(e["clean_probability"]) and 0 <= e["clean_probability"] <= 1):
        raise ConfigError(".errors.clean_probability must be in [0,1]")
    _check_range(cfg, ("errors", "count"), 0, 12)
    for t, w in e["type_weights"].items():
        if not (_is_num(w) and w >= 0):
            raise ConfigError(".errors.type_weights.%s must be >= 0" % t)
    for t, vw in e["variant_weights"].items():
        for v, w in vw.items():
            if not (_is_num(w) and w >= 0):
                raise ConfigError(".errors.variant_weights.%s.%s must be >= 0" % (t, v))
    for s, p in e["interacting"].items():
        if s not in SCENARIOS or not (_is_num(p) and 0 <= p <= 1):
            raise ConfigError(".errors.interacting.%s must be a probability" % s)
    if e["mode"] == "exact":
        ex = e["exact"]
        if not isinstance(ex, dict) or not ex:
            raise ConfigError(".errors.exact must be a non-empty object in exact mode")
        total = 0
        for k, n in ex.items():
            if not (_is_int(n) and n >= 0):
                raise ConfigError(".errors.exact.%s must be a non-negative int" % k)
            total += n
            if k in SCENARIOS:
                if k not in IMPLEMENTED_SCENARIOS:
                    raise ConfigError("scenario %s is not implemented in this version" % k)
                continue
            t, _, v = k.partition(":")
            if t not in VARIANTS or v not in VARIANTS[t]:
                raise ConfigError("unknown exact plan key %r (use type:variant)" % k)
            if t not in IMPLEMENTED_TYPES:
                raise ConfigError("issue type %s is not implemented in this version" % t)
        if total > 12:
            raise ConfigError(".errors.exact plans more than 12 issues")
    elif e["exact"] is not None:
        raise ConfigError(".errors.exact is only allowed in exact mode")
    t = cfg["traps"]
    if not (_is_num(t["rate"]) and 0 <= t["rate"] <= 1):
        raise ConfigError(".traps.rate must be in [0,1]")
    _check_choice(t["enabled"], TRAPS, ("traps", "enabled")) if t["enabled"] else None
    pr = cfg["presentation"]
    for k in ("id_salt", "order_seed"):
        if not (_is_int(pr[k]) and 0 <= pr[k] < 2**63):
            raise ConfigError(".presentation.%s must be a non-negative int" % k)
    _check_choice(pr["order"], ("time", "shuffle"), ("presentation", "order"))
    if isinstance(pr["order"], list):
        raise ConfigError(".presentation.order must be a single value")
    lim = cfg["limits"]
    if not (_is_int(lim["max_steps"]) and 1 <= lim["max_steps"] <= 100):
        raise ConfigError(".limits.max_steps must be in [1,100]")
    if not (_is_int(lim["page_size"]) and 1 <= lim["page_size"] <= 50):
        raise ConfigError(".limits.page_size must be in [1,50]")
    if not (_is_num(lim["time_limit_s"]) and 0 < lim["time_limit_s"] <= 60):
        raise ConfigError(".limits.time_limit_s must be in (0,60]")
    if cfg["source"] is not None:
        raise ConfigError(".source (loading episodes from disk) is not implemented in this version")


def load(config) -> dict:
    """Merge a user config over DEFAULTS and validate. None -> defaults."""
    cfg = _merge(DEFAULTS, {} if config is None else config)
    validate(cfg)
    return cfg


def config_core(cfg: dict) -> dict:
    core = copy.deepcopy(cfg)
    core.pop("presentation", None)
    core.pop("source", None)
    return core


def config_hash(cfg: dict) -> str:
    return sha256_hex(canonical_bytes(config_core(cfg)))


def pick(rng, cfg: dict, path):
    """Resolve a config value: [min,max] range, list of choices, or scalar."""
    v = cfg
    for k in path:
        v = v[k]
    if tuple(path) in RANGE_KEYS:
        if _is_int(v):
            return v
        return rng.randint(v[0], v[1])
    if isinstance(v, list):
        return rng.choice(v)
    return v
