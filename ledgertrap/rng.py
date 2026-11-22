"""Deterministic PRNG based on SHA-256 in counter mode.

`random.Random` does not promise identical `choice`/`shuffle`/`sample` across
Python versions, so all generation randomness goes through `DetRng`.
"""

import hashlib

from .canon import hmac_sha256

_SCALE = 10**6  # probabilities/weights are quantised to 1e-6


class DetRng:
    __slots__ = ("_key", "_ctr", "_buf")

    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key must be bytes")
        self._key = bytes(key)
        self._ctr = 0
        self._buf = b""

    def fork(self, label: str) -> "DetRng":
        return DetRng(hmac_sha256(self._key, label.encode("utf-8")))

    def _take(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha256(self._key + self._ctr.to_bytes(8, "big")).digest()
            self._ctr += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def u64(self) -> int:
        return int.from_bytes(self._take(8), "big")

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.u64()
            if x < limit:
                return x % n

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("empty range")
        return a + self.below(b - a + 1)

    def choice(self, seq):
        if not seq:
            raise ValueError("empty sequence")
        return seq[self.below(len(seq))]

    def shuffle(self, lst: list) -> None:
        for i in range(len(lst) - 1, 0, -1):
            j = self.below(i + 1)
            lst[i], lst[j] = lst[j], lst[i]

    def chance(self, p) -> bool:
        threshold = int(round(float(p) * _SCALE))
        return self.below(_SCALE) < threshold

    def weighted(self, items):
        """items: list of (value, weight). Weights quantised to 1e-6."""
        ints = [(v, int(round(float(w) * _SCALE))) for v, w in items]
        total = sum(w for _, w in ints)
        if total <= 0:
            raise ValueError("weights sum to zero")
        r = self.below(total)
        for v, w in ints:
            if r < w:
                return v
            r -= w
        raise AssertionError("unreachable")

    def token(self, alphabet: str, length: int) -> str:
        return "".join(alphabet[self.below(len(alphabet))] for _ in range(length))
