"""Read-only record store behind the tools. Returns copies, never internal objects.

Records are returned exactly as in the episode files (case §03 fields, file order): ledger rows with integer
minor amounts and null for empty references, full invoices (with line_items), payments, fx rows. Ids are
looked up as exact strings. Lookup keys: ledger row_id; invoices doc_id or invoice_id; payments doc_id or
payment_id; fx 'YYYY-MM-DD:SRC:TGT' (date:source_currency:target_currency, unique per file).
"""

import base64

from .canon import deep_copy_json

TABLES = ("ledger", "invoices", "payments", "fx")


class StoreError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class RecordStore:
    def __init__(self, episode, page_size=50):
        rec = episode.records
        self.page_size = page_size
        self._tables = {t: rec[t] for t in TABLES}
        self._index = {
            "ledger": {r["row_id"]: r for r in rec["ledger"]},
            "invoices": {}, "payments": {},
            "fx": {"%s:%s:%s" % (r["date"], r["source_currency"], r["target_currency"]): r for r in rec["fx"]},
        }
        for h in rec["invoices"]:
            self._index["invoices"][h["invoice_id"]] = h
            self._index["invoices"][h["doc_id"]] = h
        for p in rec["payments"]:
            self._index["payments"][p["payment_id"]] = p
            self._index["payments"][p["doc_id"]] = p
        self._documents = {}
        for d in rec["invoices"] + rec["payments"]:
            self._documents[d["doc_id"]] = d

    # cursors -----------------------------------------------------------------------------
    @staticmethod
    def make_cursor(table, offset):
        raw = ("%s:%d" % (table, offset)).encode("ascii")
        return base64.b32encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def parse_cursor(cursor, table):
        if not isinstance(cursor, str) or not cursor or len(cursor) > 64:
            raise StoreError("BAD_CURSOR")
        try:
            pad = "=" * (-len(cursor) % 8)
            raw = base64.b32decode(cursor + pad).decode("ascii")
            t, _, off = raw.partition(":")
            if t != table or not off.isdigit():
                raise ValueError
            return int(off)
        except (ValueError, UnicodeDecodeError, base64.binascii.Error):
            raise StoreError("BAD_CURSOR") from None

    # tools -------------------------------------------------------------------------------
    def list_records(self, table, cursor=None, limit=None):
        if table not in TABLES:
            raise StoreError("UNKNOWN_TABLE")
        rows = self._tables[table]
        offset = 0 if cursor is None else self.parse_cursor(cursor, table)
        if offset > len(rows):
            raise StoreError("BAD_CURSOR")
        n = self.page_size if limit is None else limit
        page = rows[offset:offset + n]
        nxt = offset + len(page)
        return {"rows": deep_copy_json(page),
                "next_cursor": self.make_cursor(table, nxt) if nxt < len(rows) else None,
                "total": len(rows)}

    def get_record(self, table, rid):
        if table not in TABLES:
            raise StoreError("UNKNOWN_TABLE")
        rec = self._index[table].get(rid)
        if rec is None:
            raise StoreError("UNKNOWN_ID")
        return {"record": deep_copy_json(rec)}

    def get_document(self, doc_id):
        doc = self._documents.get(doc_id)
        if doc is None:
            raise StoreError("UNKNOWN_ID")
        return {"document": deep_copy_json(doc)}
