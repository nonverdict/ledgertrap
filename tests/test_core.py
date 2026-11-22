"""Deterministic RNG, money conversion, tolerance, time rules, calculator."""

import unittest
from datetime import date
from decimal import Decimal, getcontext

from ledgertrap.calculator import CalcError, evaluate
from ledgertrap.money import conv, format_minor, unconv
from ledgertrap.rng import DetRng
from ledgertrap.timeutil import last_business_on_or_before, local_date, parse, parse_offset


class TestRng(unittest.TestCase):
    def test_same_key_same_stream(self):
        a, b = DetRng(b"k"), DetRng(b"k")
        self.assertEqual([a.u64() for _ in range(50)], [b.u64() for _ in range(50)])

    def test_known_first_value_is_stable(self):
        # Pins the algorithm: sha256(key || counter) big-endian. Changing it breaks reproducibility.
        import hashlib
        expected = int.from_bytes(hashlib.sha256(b"k" + (0).to_bytes(8, "big")).digest()[:8], "big")
        self.assertEqual(DetRng(b"k").u64(), expected)

    def test_fork_independent(self):
        r = DetRng(b"k")
        self.assertNotEqual(r.fork("a").u64(), r.fork("b").u64())
        self.assertEqual(r.fork("a").u64(), DetRng(b"k").fork("a").u64())

    def test_below_range_and_coverage(self):
        r = DetRng(b"x")
        seen = {r.below(7) for _ in range(500)}
        self.assertEqual(seen, set(range(7)))

    def test_shuffle_is_permutation(self):
        r = DetRng(b"s")
        x = list(range(20))
        r.shuffle(x)
        self.assertEqual(sorted(x), list(range(20)))


class TestMoney(unittest.TestCase):
    def test_usd_to_kzt(self):
        self.assertEqual(conv(100000, "USD", "KZT", "512.34", "HALF_UP"), 51234000)

    def test_jpy_zero_decimals(self):
        # 1500 JPY * 3.4567 = 5185.05 KZT -> 518505 minor
        self.assertEqual(conv(1500, "JPY", "KZT", "3.4567", "HALF_UP"), 518505)

    def test_kwd_three_decimals(self):
        # 1.234 KWD * 1600.5 = 1975.017 KZT -> 197501.7 minor -> 197502
        self.assertEqual(conv(1234, "KWD", "KZT", "1600.5", "HALF_UP"), 197502)

    def test_half_up_vs_half_even(self):
        self.assertEqual(conv(5, "USD", "KZT", "0.1", "HALF_UP"), 1)    # 0.5 -> 1
        self.assertEqual(conv(5, "USD", "KZT", "0.1", "HALF_EVEN"), 0)  # 0.5 -> 0
        self.assertEqual(conv(15, "USD", "KZT", "0.1", "HALF_EVEN"), 2)  # 1.5 -> 2

    def test_negative_symmetric(self):
        for mode in ("HALF_UP", "HALF_EVEN"):
            for x in (5, 15, 12345, 999):
                self.assertEqual(conv(-x, "USD", "KZT", "0.1", mode), -conv(x, "USD", "KZT", "0.1", mode))

    def test_same_currency_identity(self):
        self.assertEqual(conv(12345, "KZT", "KZT", None, "HALF_UP"), 12345)

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            conv(1.5, "USD", "KZT", "1", "HALF_UP")

    def test_unconv_inverts_conv_when_base_is_finer(self):
        for x in (1, 99, 12345, 10**9):
            self.assertEqual(unconv(conv(x, "USD", "KZT", "521.821", "HALF_UP"), "USD", "KZT", "521.821"), x)

    def test_format(self):
        self.assertEqual(format_minor(286000, "KZT"), "2 860,00 KZT")
        self.assertEqual(format_minor(1500, "JPY"), "1 500 JPY")
        self.assertEqual(format_minor(1234, "KWD"), "1,234 KWD")


class TestTime(unittest.TestCase):
    def test_weekend_rate_date_is_friday(self):
        self.assertEqual(last_business_on_or_before(date(2026, 3, 28)), date(2026, 3, 27))  # Sat -> Fri
        self.assertEqual(last_business_on_or_before(date(2026, 3, 29)), date(2026, 3, 27))  # Sun -> Fri
        self.assertEqual(last_business_on_or_before(date(2026, 3, 31)), date(2026, 3, 31))

    def test_offsets_change_local_date(self):
        tz = parse_offset("+05:00")
        self.assertEqual(local_date(parse("2026-04-01T02:00:00+08:00"), tz), date(2026, 3, 31))
        self.assertEqual(local_date(parse("2026-03-31T20:10:00+00:00"), tz), date(2026, 4, 1))
        self.assertEqual(local_date(parse("2026-03-31T23:00:00-12:00"), parse_offset("+14:00")), date(2026, 4, 2))


class TestCalculator(unittest.TestCase):
    def test_exact_decimal(self):
        self.assertEqual(evaluate("(1.5+2)*3/7"), "1.5")
        self.assertEqual(evaluate("0.1+0.2"), "0.3")
        self.assertEqual(evaluate("-3 - -2"), "-1")
        self.assertEqual(evaluate("1000.00 * 512.34"), "512340")
        self.assertTrue(evaluate("1/3").startswith("0.3333333333"))

    def test_rejects_code_and_bad_input(self):
        for bad in ("__import__('os')", "2**3", "abs(1)", "1/0", "1,5", "x+1", "1 % 2", "4//2", "",
                    "1" * 201, "(" * 60 + "1" + ")" * 60, "1e5", "()", "1 2", None, 5):
            with self.assertRaises(CalcError, msg=repr(bad)):
                evaluate(bad)

    def test_global_decimal_context_untouched(self):
        prec = getcontext().prec
        evaluate("1/7")
        self.assertEqual(getcontext().prec, prec)
        self.assertEqual(Decimal(1) / Decimal(7), Decimal(1) / Decimal(7))


if __name__ == "__main__":
    unittest.main()
