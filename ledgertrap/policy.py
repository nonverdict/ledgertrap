"""Agent-visible policy (DESIGN §6.6) and task text. Contains no truth."""

from .money import CURRENCY_EXP

ISSUE_RULES = {
    "duplicate_invoice": {
        "description": "A customer invoice that is booked twice.",
        "cases": {
            "document": {
                "rule": "Two invoice documents of the same entity and currency with identical line_items "
                        "(including service periods), both posted to the ledger.",
                "affected_ids": "doc_ids of both invoice documents (original and duplicate)",
                "amount": "total_minor of the duplicate, in its currency",
                "evidence_rule": "both documents + one AR-or-revenue row of each document's own posting",
            },
            "double_posting": {
                "rule": "One invoice document whose own posting (invoice_id set, payment_id null) appears twice.",
                "affected_ids": "doc_id of the invoice document",
                "amount": "invoice total_minor, in invoice currency",
                "evidence_rule": "the document + two rows of the same role (two AR rows or two revenue rows)",
            },
        },
    },
    "payment_allocation": {
        "description": "A payment allocation (invoice_refs) that is missing, partial or posted to another invoice.",
        "cases": {
            "missing": {
                "rule": "Payment refers to invoice X, but no AR credit for X; the amount sits in advances.",
                "affected_ids": "doc_ids of the payment and of invoice X",
                "amount": "the allocation to X, in payment currency; equals unconv(advances row) - (amount_minor - "
                          "sum of invoice_refs amounts), see partial",
                "evidence_rule": "payment + invoice X + advances row (+ fx rate of the payment date if foreign currency)",
            },
            "partial": {
                "rule": "Payment refers to invoice X, but only part of the allocation is credited to X; "
                        "the remainder sits in advances.",
                "affected_ids": "doc_ids of the payment and of invoice X",
                "amount": "the part not credited to X, in payment currency: unconv(advances row) - (amount_minor "
                          "- sum of invoice_refs amounts), where unconv(y) = ROUND_HALF_EVEN(y / rate * "
                          "10^(exp(payment currency) - exp(base))) with the payment-date rate (identity if the "
                          "payment is in base currency); the generator guarantees this recovers the amount exactly",
                "evidence_rule": "payment + invoice X + advances row (+ fx rate of the payment date if foreign currency)",
            },
            "wrong_invoice": {
                "rule": "Payment refers to invoice X, but the AR credit is posted with another invoice_id.",
                "affected_ids": "doc_ids of the payment and of invoice X",
                "amount": "the allocation to X, in payment currency",
                "evidence_rule": "payment + invoice X + the misdirected AR row",
            },
        },
    },
    "fx_conversion": {
        "description": "A foreign-currency document converted with the wrong rate.",
        "cases": {
            "wrong_date": {
                "rule": "Rate of a different business day than the policy rate date was used.",
                "affected_ids": "doc_id of the document",
                "amount": "|posted - correct| in base currency (AR row for invoices, bank row for payments)",
                "evidence_rule": "document + the wrongly converted row + fx rate of the correct date",
            },
            "wrong_rate": {
                "rule": "A rate that does not exist in fx for that currency was used.",
                "affected_ids": "doc_id of the document",
                "amount": "|posted - correct| in base currency",
                "evidence_rule": "document + the wrongly converted row + fx rate of the correct date",
            },
        },
    },
    "period_cutoff": {
        "description": "A document posted in a different interval (period vs after-period window) than its date.",
        "cases": {
            "early": {
                "rule": "Document dated on/after period end is posted inside the period.",
                "affected_ids": "doc_id of the document",
                "amount": "|document amount| in document currency",
                "evidence_rule": "document + any one row of its posting",
            },
            "late": {
                "rule": "Document dated inside the period is posted on/after period end.",
                "affected_ids": "doc_id of the document",
                "amount": "|document amount| in document currency",
                "evidence_rule": "document + any one row of its posting",
            },
        },
    },
}

POSTING_RULES = [
    "Signed amounts: a positive amount goes to the stated side, a negative one to the opposite side.",
    "Invoice X: Dr ar conv(total, rate(X)) / Cr revenue, invoice_id=X, payment_id=null.",
    "Credit note CN (negative invoice): Dr returns conv(|total|) / Cr ar, invoice_id=CN.",
    "Opening balance of an earlier unpaid invoice X: Dr ar / Cr opening_equity at period_start, invoice_id=X.",
    "Payment P of amount A: Dr bank conv(A, rate(P)), payment_id=P.",
    "For each invoice_refs item (X, a): Cr ar with invoice_id=X, payment_id=P, amount "
    "conv(paid_before+a, rate(X)) - conv(paid_before, rate(X)); paid_before = allocations to X by payments "
    "with an earlier moment.",
    "Unallocated remainder u = A - sum(a): Cr advances conv(u, rate(P)).",
    "FX difference D = bank - sum(ar credits) - advances: D>0 Cr fx_gain, D<0 Dr fx_loss.",
    "Refund: payment with negative amount referring to a credit note.",
    "Other expenses: Dr expense / Cr bank, no references.",
    "A document's own posting is made 0-3 days after its local date and stays in the same interval "
    "(period or after-period window).",
]


def build_policy(world):
    c = world.company
    return {
        "period": {"start": world.period_start.isoformat(), "end": world.period_end.isoformat(),
                   "interval": "[start,end)", "basis": "instant; dates are taken in the company timezone"},
        "timezone": c["timezone"],
        "window_days": (world.export_end - world.period_end).days,
        "business_days": "Monday-Friday, no holidays",
        "fx_rate_date": "last business day on or before the document's local date (invoice: issued_at, "
                        "payment: timestamp); the posting date is irrelevant",
        "fx_quote": "1 unit of source_currency = rate units of target_currency (major units)",
        "rounding": c["rounding"],
        "rounding_note": "HALF_UP rounds halves away from zero; HALF_EVEN is banker's rounding",
        "conversion": "conv(x) = sign(x) * round(|x| * rate * 10^(exp(base)-exp(src))) to an integer, once per amount",
        "currency_exponents": {k: CURRENCY_EXP[k] for k in sorted(world.currencies)},
        "accounts": c["accounts"],
        "posting_rules": POSTING_RULES,
        "issue_types": ISSUE_RULES,
        "submission": {
            "max_findings": 30, "max_evidence_ids": 10, "max_explanation_chars": 1000,
            "amount_tolerance_minor": {t: 0 for t in ISSUE_RULES},
            "notes": ["ids, type and currency are compared as exact strings (case and spaces matter)",
                      "affected_ids must equal the canonical set exactly: the doc_ids named in issue_types "
                      "(invoice_id / payment_id numbers are not canonical); order and repeats inside the list "
                      "do not matter",
                      "evidence_ids: at least one complete evidence set is required; extra ids are allowed only "
                      "if they exist (row_id, doc_id, invoice_id, payment_id or fx key 'YYYY-MM-DD:SRC:TGT'); "
                      "an unknown evidence id makes the finding unmatched",
                      "claimed_amount_minor is a JSON integer in minor units of `currency`; a match needs "
                      "|claimed - expected| <= amount_tolerance_minor[type] (0 for every type: amounts are "
                      "exactly recomputable from the records with the rules above)",
                      "matching is one-to-one and maximum; a finding that matches no issue and every repeated "
                      "finding is a false positive",
                      "structure errors (wrong JSON type, missing field, > 30 findings, > 10 evidence ids before "
                      "de-duplication, explanation > 1000 characters) end the episode with reward -1"],
        },
    }


TASK_TEXT = (
    "You are auditing one company for one reporting period. Records are read-only and available only "
    "through the tools. Find misstatements of the four issue types described in policy.issue_types and "
    "submit them once with submit(findings). Each finding needs the issue type, the affected document ids, "
    "the amount in minor units with its currency, and evidence ids (documents, ledger row_ids, fx ids "
    "'YYYY-MM-DD:SRC:TGT') that form a complete evidence set. Correct-but-suspicious records exist; "
    "submitting nothing is the right answer for a clean company. Exploration earns no reward; each "
    "non-submit step costs 0.002 of the final reward. Budget: 100 steps including submit."
)
