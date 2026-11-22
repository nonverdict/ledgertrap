"""Metrics and rewards exactly as case §05.

P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); zero denominators give 0.
Empty truth and empty findings -> F1 = 1; empty truth with findings -> F1 = 0.
reward_main = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1). Invalid structure -> -1.
"""

STEP_COST = 0.002


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if tp + fp + fn == 0:
        f1 = 1.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return p, r, f1


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def reward_main(f1, inspection_steps):
    return clip(2.0 * f1 - 1.0 - STEP_COST * inspection_steps)


def f1_soft(tp, fp, fn, near):
    """Experimental (bonus): each 'near' pair is removed from FP and FN and counts half."""
    fp2, fn2 = fp - near, fn - near
    denom = 2 * tp + 2 * near + fp2 + fn2
    if denom == 0:
        return 1.0
    return (2 * tp + near) / denom
