"""Canonical JSON and hashing helpers (byte-stable across Python versions)."""

import hashlib
import hmac
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def deep_copy_json(obj):
    """Copy a JSON-compatible value so callers cannot mutate internal state."""
    return json.loads(json.dumps(obj))
