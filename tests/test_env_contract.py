"""reset/step contract, tools, limits, invalid input, modes, isolation, reproducibility."""

import json
import unittest

from ledgertrap.env import LedgerTrapEnv
from ledgertrap.policies.baselines import silent_action
from tests.helpers import DIRTY, FakeClock, env_for, oracle_obj, submit

TRUTH_KEYS = ("issue_id", "evidence_sets", "expected_amount_minor", "is_clean", "scenario", "variant",
              "certificate_id", "tolerance_minor")


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_keys(v)


class TestReset(unittest.TestCase):
    def test_reset_shape_json(self):
        env, obs, info = env_for(1)
        json.dumps(obs), json.dumps(info)
        for k in ("task", "manifest", "policy", "tools", "budget"):
            self.assertIn(k, obs)
        self.assertEqual(obs["budget"]["steps_left"], 100)
        self.assertEqual(obs["budget"]["page_size"], 50)
        self.assertEqual(obs["budget"]["time_limit_s"], 60)
        self.assertEqual({t["name"] for t in obs["tools"]},
                         {"list_records", "get_record", "get_document", "calculator", "submit"})
        for cases in obs["policy"]["issue_types"].values():
            for case in cases["cases"].values():
                self.assertIn("evidence_rule", case)
        self.assertEqual(info["episode_id"], obs["manifest"]["episode_id"])

    def test_eval_info_empty(self):
        env, obs, info = env_for(1, mode="eval")
        self.assertEqual(info, {})

    def test_reset_isolates_episodes(self):
        env, obs, _ = env_for(1)
        first_row = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]
        row_id = first_row["result"]["rows"][0]["row_id"]
        env.reset(2, DIRTY)
        o = env.step({"tool": "get_record", "args": {"table": "ledger", "id": row_id}})[0]
        self.assertEqual(o["error"]["code"], "UNKNOWN_ID")
        self.assertEqual(o["steps_left"], 99)

    def test_step_before_reset(self):
        with self.assertRaises(RuntimeError):
            LedgerTrapEnv().step({"tool": "calculator", "args": {"expression": "1"}})

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            LedgerTrapEnv(mode="debug")


class TestTools(unittest.TestCase):
    def test_step_shape(self):
        env, obs, _ = env_for(3)
        out = env.step({"tool": "list_records", "args": {"table": "invoices"}})
        self.assertEqual(len(out), 5)
        o, r, term, trunc, info = out
        self.assertIsInstance(o, dict)
        self.assertIsInstance(r, float)
        self.assertIs(term, False)
        self.assertIs(trunc, False)
        self.assertEqual(info, {})
        self.assertEqual(r, 0.0)
        json.dumps(out)
        # records are returned exactly as in invoices.jsonl (case §03 fields, line_items included)
        self.assertEqual(o["result"]["rows"][0], env.privileged_episode().records["invoices"][0])

    def test_pagination_complete_and_stable(self):
        env, obs, _ = env_for(4)
        for table in ("ledger", "invoices", "payments", "fx"):
            got, cursor, cursors = [], None, []
            while True:
                o = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursor}})[0]
                self.assertTrue(o["ok"], o)
                self.assertLessEqual(len(o["result"]["rows"]), 50)
                got.extend(o["result"]["rows"])
                cursor = o["result"]["next_cursor"]
                if cursor is None:
                    break
                cursors.append(cursor)
            self.assertEqual(len(got), o["result"]["total"])
            self.assertEqual(got, env.privileged_episode().records[table])  # every file row, file order
            if table == "fx":
                keys = {"%s:%s:%s" % (g["date"], g["source_currency"], g["target_currency"]) for g in got}
            else:
                keys = {g["row_id" if table == "ledger" else "doc_id"] for g in got}
            self.assertEqual(len(keys), len(got))
            if cursors:
                again = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursors[0]}})[0]
                first = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursors[0]}})[0]
                self.assertEqual(again["result"], first["result"])
            env.reset(4, DIRTY)

    def test_cursor_errors_and_limits(self):
        env, obs, _ = env_for(4)
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})[0]
        self.assertEqual(o["error"]["code"], "BAD_ARGS")
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 0}})[0]
        self.assertEqual(o["error"]["code"], "BAD_ARGS")
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": True}})[0]
        self.assertEqual(o["error"]["code"], "BAD_ARGS")
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": "garbage!!"}})[0]
        self.assertEqual(o["error"]["code"], "BAD_CURSOR")
        fx_cursor = env.step({"tool": "list_records", "args": {"table": "fx", "limit": 1}})[0]["result"]["next_cursor"]
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": fx_cursor}})[0]
        self.assertEqual(o["error"]["code"], "BAD_CURSOR")
        total = obs["manifest"]["counts"]["ledger"]
        from ledgertrap.store import RecordStore
        end = env.step({"tool": "list_records",
                        "args": {"table": "ledger", "cursor": RecordStore.make_cursor("ledger", total)}})[0]
        self.assertEqual(end["result"]["rows"], [])
        self.assertIsNone(end["result"]["next_cursor"])
        o = env.step({"tool": "list_records",
                      "args": {"table": "ledger", "cursor": RecordStore.make_cursor("ledger", total + 1)}})[0]
        self.assertEqual(o["error"]["code"], "BAD_CURSOR")

    def test_get_record_and_document(self):
        env, obs, _ = env_for(5)
        inv = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]["result"]["rows"][0]
        for table, rid in (("invoices", inv["invoice_id"]), ("invoices", inv["doc_id"])):
            o = env.step({"tool": "get_record", "args": {"table": table, "id": rid}})[0]
            self.assertTrue(o["ok"], o)
            self.assertEqual(o["result"]["record"], inv)
        for near_miss in (" " + inv["doc_id"], inv["doc_id"].lower(), inv["doc_id"] + " "):  # ids are exact strings
            o = env.step({"tool": "get_record", "args": {"table": "invoices", "id": near_miss}})[0]
            self.assertEqual(o["error"]["code"], "UNKNOWN_ID")
        doc = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]
        self.assertEqual(doc["result"]["document"], inv)
        fx = env.step({"tool": "list_records", "args": {"table": "fx", "limit": 1}})[0]["result"]["rows"]
        if fx:
            key = "%s:%s:%s" % (fx[0]["date"], fx[0]["source_currency"], fx[0]["target_currency"])
            o = env.step({"tool": "get_record", "args": {"table": "fx", "id": key}})[0]
            self.assertEqual(o["result"]["record"], fx[0])
        secret_id = "NO-SUCH-ID-123"
        o = env.step({"tool": "get_document", "args": {"doc_id": secret_id}})[0]
        self.assertEqual(o["error"]["code"], "UNKNOWN_ID")
        self.assertNotIn(secret_id, json.dumps(o))

    def test_returned_records_are_copies(self):
        env, obs, _ = env_for(5)
        o = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]
        o["result"]["rows"][0]["debit_minor"] = -999
        o2 = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]
        self.assertNotEqual(o2["result"]["rows"][0]["debit_minor"], -999)

    def test_invalid_actions_spend_step_and_do_not_crash(self):
        env, obs, _ = env_for(6)
        cases = [
            ("not a dict", "BAD_ACTION"), ({"tool": "calculator"}, "BAD_ACTION"),
            ({"tool": 5, "args": {}}, "BAD_ACTION"), ({"tool": "calculator", "args": [], "x": 1}, "BAD_ACTION"),
            ({"tool": "rm -rf", "args": {}}, "UNKNOWN_TOOL"), ({"tool": "shell", "args": {"cmd": "ls"}}, "UNKNOWN_TOOL"),
            ({"tool": "calculator", "args": {}}, "BAD_ARGS"),
            ({"tool": "calculator", "args": {"expression": "1", "extra": 2}}, "BAD_ARGS"),
            ({"tool": "calculator", "args": {"expression": "2**8"}}, "CALC_ERROR"),
            ({"tool": "get_record", "args": {"table": "truth", "id": "ISS-01"}}, "UNKNOWN_TABLE"),
            ({"tool": "get_record", "args": {"table": "provenance", "id": "x"}}, "UNKNOWN_TABLE"),
            ({"tool": "list_records", "args": {"table": "policy"}}, "UNKNOWN_TABLE"),
            ({"tool": "get_document", "args": {"doc_id": "../grader/truth.json"}}, "UNKNOWN_ID"),
            ({"tool": "get_document", "args": {"doc_id": 7}}, "BAD_ARGS"),
            ({"tool": "calculator", "args": {"expression": "x" * 2_000_000}}, "BAD_ACTION"),
        ]
        for i, (action, code) in enumerate(cases, start=1):
            o, r, term, trunc, info = env.step(action)
            self.assertFalse(o["ok"], action)
            self.assertEqual(o["error"]["code"], code, action)
            self.assertEqual(o["steps_left"], 100 - i)
            self.assertEqual((r, term, trunc, info), (0.0, False, False, {}))

    def test_internal_error_is_opaque(self):
        env, obs, _ = env_for(6)

        def boom(*a, **k):
            raise RuntimeError("/home/secret/path truth.json")
        env._store.list_records = boom
        o = env.step({"tool": "list_records", "args": {"table": "ledger"}})[0]
        self.assertEqual(o["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("truth", json.dumps(o))
        self.assertTrue(env.internal_log)


class TestTermination(unittest.TestCase):
    def test_submit_terminates(self):
        env, obs, _ = env_for(7)
        o, r, term, trunc, info = submit(env, oracle_obj(env))
        self.assertEqual((term, trunc), (True, False))
        self.assertEqual(o["type"], "episode_end")
        self.assertEqual(o["reason"], "submitted")
        self.assertEqual(r, 1.0)
        self.assertEqual(info["metrics"]["tp"], len(env.privileged_episode().truth["issues"]))

    def test_step_after_end(self):
        env, obs, _ = env_for(7)
        submit(env, oracle_obj(env))
        for a in ({"tool": "list_records", "args": {"table": "ledger"}},
                  {"tool": "submit", "args": {"findings": oracle_obj(env)}}):
            o, r, term, trunc, info = env.step(a)
            self.assertEqual(o["error"]["code"], "EPISODE_DONE")
            self.assertEqual((r, term, trunc, info), (0.0, True, False, {}))
        self.assertEqual(len(env.journal), 1)

    def test_max_steps_truncates_with_empty_answer(self):
        env, obs, _ = env_for(8, kind="clean")
        for i in range(99):
            o, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
            self.assertFalse(term or trunc)
        o, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
        self.assertEqual((term, trunc), (False, True))
        self.assertEqual(o["reason"], "max_steps")
        self.assertAlmostEqual(r, 1.0 - 0.002 * 100, places=9)  # clean company, empty answer, 100 steps

    def test_submit_allowed_as_hundredth_step(self):
        env, obs, _ = env_for(8)
        for _ in range(99):
            env.step({"tool": "calculator", "args": {"expression": "1"}})
        o, r, term, trunc, info = submit(env, oracle_obj(env))
        self.assertEqual((term, trunc), (True, False))
        self.assertAlmostEqual(r, 1.0 - 0.002 * 99, places=9)

    def test_time_limit_truncates(self):
        clock = FakeClock()
        env, obs, _ = env_for(9, clock=clock)
        env.step({"tool": "calculator", "args": {"expression": "1"}})
        clock.t += 61
        o, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1"}})
        self.assertEqual((term, trunc, o["reason"]), (False, True, "time_limit"))
        self.assertAlmostEqual(r, -1.0, places=9)  # dirty company, empty answer
        self.assertEqual(info["metrics"]["inspection_steps"], 1)

    def test_time_limit_also_blocks_submit(self):
        clock = FakeClock()
        env, obs, _ = env_for(9, clock=clock)
        clock.t += 60.5
        o, r, term, trunc, info = submit(env, oracle_obj(env))
        self.assertEqual((term, trunc), (False, True))


class TestEvalMode(unittest.TestCase):
    def test_eval_is_silent(self):
        env, obs, info = env_for(10, mode="eval")
        self.assertEqual(info, {})
        o, r, term, trunc, info = env.step({"tool": "list_records", "args": {"table": "ledger"}})
        self.assertEqual((r, info), (0.0, {}))
        o, r, term, trunc, info = submit(env, oracle_obj(env))
        self.assertEqual((r, term, info), (0.0, True, {}))
        self.assertNotIn("metrics", json.dumps(o))
        self.assertAlmostEqual(env.journal[-1]["metrics"]["reward_main"], 1.0 - 0.002 * 1, places=9)

    def test_eval_invalid_submission_still_silent(self):
        env, obs, info = env_for(10, mode="eval")
        o, r, term, trunc, info = env.step({"tool": "submit", "args": {"findings": "garbage"}})
        self.assertEqual((r, term, info), (0.0, True, {}))
        self.assertEqual(env.journal[-1]["metrics"]["reward_main"], -1.0)


class TestNoLeak(unittest.TestCase):
    def test_no_truth_keys_or_paths_in_observations(self):
        for mode in ("train", "eval"):
            env, obs, info = env_for(11, mode=mode)
            seen = [obs, info]
            for table in ("ledger", "invoices", "payments", "fx"):
                o = env.step({"tool": "list_records", "args": {"table": table}})[0]
                seen.append(o)
                for row in o["result"]["rows"][:3]:
                    if "doc_id" in row:
                        seen.append(env.step({"tool": "get_document", "args": {"doc_id": row["doc_id"]}})[0])
            seen.extend(env.step(submit_args)[0:5:4] for submit_args in [
                {"tool": "submit", "args": {"findings": oracle_obj(env)}}])
            ep = env.privileged_episode()
            blob = json.dumps(seen, ensure_ascii=False)
            keys = set(walk_keys(seen))
            for k in TRUTH_KEYS:
                self.assertNotIn(k, keys)
            for frag in ("/home", "/tmp", "truth.json", "provenance", "grader/", "ledgertrap-open-v1"):
                self.assertNotIn(frag, blob)
            if mode == "eval":  # secret_id (a hash) is train-mode info only
                self.assertNotIn(ep.secret_id, blob)
            for g in ep.truth["issues"]:
                self.assertNotIn(g["issue_id"], blob)
                self.assertNotIn(g["certificate_id"], blob)

    def test_silent_policy_helper(self):
        env, obs, _ = env_for(12, kind="clean")
        o, r, term, trunc, info = env.step(silent_action(obs))
        self.assertEqual(r, 1.0)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_actions_same_transitions(self):
        actions = [{"tool": "list_records", "args": {"table": "ledger"}},
                   {"tool": "list_records", "args": {"table": "invoices", "limit": 7}},
                   {"tool": "calculator", "args": {"expression": "10/4"}},
                   {"tool": "get_record", "args": {"table": "fx", "id": "nope"}}]

        def run():
            env, obs, info = env_for(13)
            out = [json.dumps([obs, info], sort_keys=True)]
            for a in actions:
                out.append(json.dumps(env.step(a), sort_keys=True))
            out.append(json.dumps(submit(env, oracle_obj(env)), sort_keys=True))
            return out
        self.assertEqual(run(), run())


class TestServeCommand(unittest.TestCase):
    def test_serve_json_lines(self):
        import os
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lines = ['{"tool": "list_records", "args": {"table": "fx", "limit": 2}}', "not json",
                 '{"tool": "submit", "args": {"findings": {"episode_id": "x", "findings": []}}}',
                 '{"tool": "calculator", "args": {"expression": "1"}}']
        out = subprocess.run([sys.executable, "-m", "ledgertrap", "serve", "--seed", "5", "--mode", "eval"],
                             input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=root, timeout=120)
        recs = [json.loads(x) for x in out.stdout.splitlines()]
        self.assertEqual(len(recs), 4)  # reset + 3 steps; the episode ended at submit
        self.assertEqual(recs[0]["info"], {})
        self.assertEqual(recs[2]["observation"]["error"]["code"], "BAD_ACTION")
        self.assertTrue(recs[3]["terminated"])
        self.assertEqual(recs[3]["reward"], 0.0)  # eval: no reward to the policy


if __name__ == "__main__":
    unittest.main()
