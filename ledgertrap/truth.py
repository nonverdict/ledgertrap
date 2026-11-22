"""Truth issues in public IDs, human-checkable certificates, and trap detection."""

from collections import defaultdict
from datetime import timedelta

from .money import format_minor, format_rate
from .timeutil import is_business


def detect_traps(world):
    """Which correct-but-suspicious patterns (DESIGN §8.4) exist in the clean world."""
    found = set()
    invs = [d for d in world.invoices() if d["kind"] == "invoice"]
    by_ent = defaultdict(list)
    for d in invs:
        sig = tuple((ln["sku"], ln["quantity"], ln["unit_price_minor"]) for ln in d["lines"])
        by_ent[(d["entity"], d["currency"], sig)].append(d)
    if any(len(v) >= 2 for v in by_ent.values()):
        found.add("L1")
    by_total = defaultdict(set)
    for d in invs:
        by_total[(d["currency"], d["total"])].add(d["entity"])
    if any(len(v) >= 2 for v in by_total.values()):
        found.add("L2")
    for d in world.invoices():
        pays = world.payments_for(d["key"])
        if len(pays) >= 2:
            found.add("L3")
        if pays and sum(a for _, a in pays) < d["total"] and d["kind"] == "invoice":
            found.add("L12")
        if d["kind"] == "credit_note":
            found.add("L4")
        if d["opening"]:
            found.add("L11")
    for p in world.payments():
        local = p["timestamp"].astimezone(world.tz).date()
        if p["currency"] != world.base and not is_business(local):
            found.add("L5")
        if p["refs"] is None:
            found.add("L8")
        elif p["amount"] > sum(a for _, a in p["refs"]) and p["amount"] > 0:
            found.add("L9")
    edge = timedelta(days=1)
    for d in world.docs.values():
        m = world.doc_moment(d)
        if abs(m - world.period_end) <= edge and m.date() != m.astimezone(world.tz).date():
            found.add("L6")
    for r in world.rows.values():
        if r["role"] in ("fx_gain", "fx_loss"):
            found.add("L10")
        doc = world.docs.get(r["payment"] or r["invoice"] or "")
        if doc is not None and r["ts"] != world.period_start:
            if r["ts"].date() > world.doc_moment(doc).astimezone(world.tz).date():
                found.add("L7")
    return sorted(found, key=lambda t: int(t[1:]))


def _local(dt, world):
    return dt.astimezone(world.tz).strftime("%Y-%m-%d %H:%M")


def certificate_text(issue, world, pub):
    """Russian, human-checkable computation for one issue (public IDs only)."""
    c = issue["cert"]
    doc_id = lambda k: pub["docs"][k]["doc_id"]          # noqa: E731
    num = lambda k: pub["docs"][k]["number"]             # noqa: E731
    row = lambda k: pub["rows"][k]                       # noqa: E731
    if c["kind"] == "duplicate_document":
        d1, d2 = world.docs[c["d1"]], world.docs[c["d2"]]
        return ("Счёт %s (%s) от %s и счёт %s (%s) от %s: один клиент %s, валюта %s, %d позиц. совпадают "
                "полностью, включая период услуги. Оба проведены: строки %s и %s. Сумма дубля %s." % (
                    num(c["d1"]), doc_id(c["d1"]), _local(d1["issued_at"], world),
                    num(c["d2"]), doc_id(c["d2"]), _local(d2["issued_at"], world),
                    pub["entities"][d1["entity"]], d1["currency"], len(d1["lines"]),
                    ", ".join(row(k) for k in c["rows_d1"]), ", ".join(row(k) for k in c["rows_d2"]),
                    format_minor(c["amount"], c["currency"])))
    if c["kind"] == "double_posting":
        return ("Счёт %s (%s) на %s проведён дважды: строки дебиторки %s и строки выручки %s, каждая на %s. "
                "Ожидается по одной строке каждой роли. Сумма ошибки %s." % (
                    num(c["doc"]), doc_id(c["doc"]), format_minor(c["amount"], c["currency"]),
                    " и ".join(row(k) for k in c["ar"]), " и ".join(row(k) for k in c["rev"]),
                    format_minor(c["base_amount"], world.base), format_minor(c["amount"], c["currency"])))
    if c["kind"] in ("cutoff_early", "cutoff_late"):
        d = world.docs[c["doc"]]
        m = world.doc_moment(d)
        iv = world.interval(m)
        r0 = world.rows[c["rows"][0]]
        return ("Документ %s (%s): метка %s, по времени компании %s (%s) — интервал «%s» "
                "(период [%s, %s)). Проводки %s датированы %s — интервал «%s». Сумма документа %s." % (
                    num(c["doc"]), doc_id(c["doc"]), m.isoformat(), _local(m, world), world.company["timezone"],
                    iv, _local(world.period_start, world), _local(world.period_end, world),
                    ", ".join(row(k) for k in c["rows"]), _local(r0["ts"], world), world.interval(r0["ts"]),
                    format_minor(c["amount"], c["currency"])))
    if c["kind"] in ("alloc_missing", "alloc_partial"):
        p = world.docs[c["pay"]]
        fxs = "" if c["currency"] == world.base else " по курсу %s на %s (%s:%s:%s)" % (
            format_rate(c["rate"]), world.rate_date(p).isoformat(), world.rate_date(p).isoformat(),
            c["currency"], world.base)
        return ("Платёж %s (%s) на %s распределён на счёт %s (%s) в сумме %s; нераспределённый остаток по документу "
                "%s. Строка авансов %s: %s%s, то есть %s в валюте платежа — на %s больше остатка. "
                "Эта часть не зачтена в счёт %s." % (
                    num(c["pay"]), doc_id(c["pay"]), format_minor(p["amount"], c["currency"]), num(c["inv"]),
                    doc_id(c["inv"]),
                    format_minor(c["a"], c["currency"]), format_minor(c["u"], c["currency"]), row(c["adv_row"]),
                    format_minor(c["adv"], world.base), fxs, format_minor(c["u"] + c["m"], c["currency"]),
                    format_minor(c["m"], c["currency"]), num(c["inv"])))
    if c["kind"] == "alloc_wrong_invoice":
        return ("Платёж %s (%s) по invoice_refs распределён на счёт %s (%s) в сумме %s, но строка дебиторки %s "
                "проведена по счёту %s. Не зачтено в счёт %s: %s." % (
                    num(c["pay"]), doc_id(c["pay"]), num(c["inv"]), doc_id(c["inv"]), format_minor(c["a"], c["currency"]),
                    row(c["row_y"]), num(c["y"]), num(c["inv"]), format_minor(c["a"], c["currency"])))
    if c["kind"] in ("fx_wrong_date", "fx_wrong_rate"):
        d = world.docs[c["doc"]]
        used = ("курс %s на %s" % (format_rate(c["wrong_rate"]), c["wrong_date"].isoformat())
                if c["wrong_date"] else "курс %s, которого нет в fx.csv" % format_rate(c["wrong_rate"]))
        return ("Документ %s (%s) на %s, дата по времени компании %s. По политике берётся курс последнего "
                "рабочего дня не позже этой даты: %s (%s:%s:%s), правильно %s. В строке %s проведено %s — это %s. "
                "Расхождение %s." % (
                    num(c["doc"]), doc_id(c["doc"]), format_minor(c["amount_doc"], d["currency"]),
                    world.doc_moment(d).astimezone(world.tz).date().isoformat(), format_rate(c["rate"]),
                    c["rate_date"].isoformat(), d["currency"], world.base, format_minor(c["right"], world.base),
                    row(c["row"]), format_minor(c["posted"], world.base), used,
                    format_minor(abs(c["posted"] - c["right"]), world.base)))
    raise ValueError(c["kind"])


def public_issues(issues, world, pub):
    """Truth issues and certificates in public IDs, with stable issue ids."""
    def pid(k):
        if k in pub["docs"]:
            return pub["docs"][k]["doc_id"]
        if k.startswith("fx:"):
            _, d, cur = k.split(":")
            return "%s:%s:%s" % (d, cur, world.base)
        return pub["rows"][k]

    out, certs = [], []
    for i, iss in enumerate(issues, start=1):
        issue_id = "ISS-%02d" % i
        cert_id = "CERT-%02d" % i
        out.append({
            "issue_id": issue_id, "type": iss["type"], "variant": iss["variant"], "scenario": iss["scenario"],
            "isa_assertion": iss["isa_assertion"],
            "affected_ids": sorted(pid(k) for k in iss["affected"]),
            "expected_amount_minor": iss["amount"], "currency": iss["currency"],
            "evidence_sets": sorted(sorted(pid(k) for k in s) for s in iss["evidence_sets"]),
            "certificate_id": cert_id,
        })
        certs.append({"certificate_id": cert_id, "issue_id": issue_id,
                      "text": certificate_text(iss, world, pub)})
    return out, certs
