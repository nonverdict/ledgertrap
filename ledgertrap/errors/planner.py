"""Plan which issues to inject and apply them (DESIGN §7.6, §8)."""

from ..config import IMPLEMENTED_TYPES, SCENARIOS, VARIANTS, pick
from . import allocation, cutoff, duplicate, fx, interacting


class GeneratorError(RuntimeError):
    pass


def make_plan(rng, cfg, n_docs):
    """Return a list of plan items: ("C1", None) / ("C2", None) for scenarios (2 issues each) and
    (type, variant) for single issues. Scenarios are applied first."""
    e = cfg["errors"]
    cap = min(12, n_docs // 4)
    if e["mode"] == "clean":
        return []
    if e["mode"] == "exact":
        plan = []
        for key in sorted(e["exact"]):
            if key in SCENARIOS:
                plan.extend([(key, None)] * e["exact"][key])
        for key in sorted(e["exact"]):
            if key not in SCENARIOS:
                t, _, v = key.partition(":")
                plan.extend([(t, v)] * e["exact"][key])
        size = sum(2 if v is None else 1 for _, v in plan)
        if size > cap:
            raise GeneratorError("exact plan has %d issues, cap for %d documents is %d" % (size, n_docs, cap))
        return plan
    if rng.fork("clean").chance(e["clean_probability"]):
        return []
    n = min(pick(rng.fork("count"), cfg, ("errors", "count")), cap)
    plan = []
    for s in SCENARIOS:
        if n >= 2 and rng.fork("scenario-" + s).chance(e["interacting"][s]):
            plan.append((s, None))
            n -= 2
    types = [(t, e["type_weights"][t]) for t in IMPLEMENTED_TYPES if e["type_weights"][t] > 0]
    if not types:
        return plan
    for i in range(n):
        r = rng.fork("issue-%d" % i)
        t = r.weighted(types)
        vs = [(v, e["variant_weights"][t][v]) for v in VARIANTS[t] if e["variant_weights"][t][v] > 0]
        if vs:
            plan.append((t, r.weighted(vs)))
    return plan


def apply_one(rng, world, used, cfg, t, v):
    same_prob = cfg["company"]["doc_offset_same_prob"]
    if t == "C1":
        return interacting.apply_c1(rng, world, used, same_prob)
    if t == "C2":
        return interacting.apply_c2(rng, world, used, same_prob)
    if t == "duplicate_invoice" and v == "document":
        return duplicate.apply_document(rng, world, used, same_prob)
    if t == "duplicate_invoice" and v == "double_posting":
        return duplicate.apply_double(rng, world, used)
    if t == "period_cutoff":
        return cutoff.apply(rng, world, used, v, cfg["company"]["window_days"])
    if t == "payment_allocation":
        return allocation.apply(rng, world, used, v)
    if t == "fx_conversion":
        return fx.apply(rng, world, used, v)
    raise GeneratorError("issue %s:%s not implemented" % (t, v))


def inject_errors(rng, cfg, world):
    n_docs = len(world.docs)
    plan = make_plan(rng.fork("plan"), cfg, n_docs)
    used = set()
    issues, unplaced = [], []
    for i, (t, v) in enumerate(plan):
        out = apply_one(rng.fork("apply-%d" % i), world, used, cfg, t, v)
        label = t if v is None else "%s:%s" % (t, v)
        if out is None:
            if cfg["errors"]["mode"] == "exact":
                raise GeneratorError("no eligible target for %s" % label)
            unplaced.append(label)
            continue
        issues.extend(out if isinstance(out, list) else [out])
    return issues, {"plan": [t if v is None else "%s:%s" % (t, v) for t, v in plan], "unplaced": unplaced,
                    "is_clean": len(issues) == 0}
