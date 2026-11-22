import unittest

from ledgertrap.config import ConfigError, config_hash, load


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = load(None)
        self.assertEqual(cfg["limits"]["max_steps"], 100)
        self.assertEqual(cfg["limits"]["page_size"], 50)

    def test_unknown_keys_rejected(self):
        for bad in ({"nope": 1}, {"company": {"nope": 1}}, {"errors": {"type_weights": {"bogus": 1}}},
                    {"limits": {"max_step": 5}}):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                load(bad)

    def test_limits_bounded_by_case(self):
        for bad in ({"limits": {"max_steps": 101}}, {"limits": {"page_size": 51}},
                    {"limits": {"time_limit_s": 61}}, {"size": {"ledger_rows": [100, 300]}},
                    {"size": {"documents": [20, 300]}}, {"errors": {"count": [0, 13]}}):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                load(bad)

    def test_hash_ignores_presentation_only(self):
        a = config_hash(load(None))
        b = config_hash(load({"presentation": {"id_salt": 7, "order_seed": 3, "order": "shuffle"}}))
        c = config_hash(load({"size": {"documents": [30, 40]}}))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_exact_mode_validation(self):
        load({"errors": {"mode": "exact", "exact": {"period_cutoff:late": 2}}})
        for bad in ({"errors": {"mode": "exact", "exact": {"period_cutoff:sideways": 1}}},
                    {"errors": {"mode": "exact", "exact": {"C3": 1}}},  # unknown scenario
                    {"errors": {"mode": "exact", "exact": None}},
                    {"errors": {"mode": "random", "exact": {"period_cutoff:late": 1}}},
                    {"errors": {"mode": "exact", "exact": {"period_cutoff:late": 13}}}):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                load(bad)

    def test_version_required(self):
        with self.assertRaises(ConfigError):
            load({"config_version": "0.9"})


if __name__ == "__main__":
    unittest.main()
