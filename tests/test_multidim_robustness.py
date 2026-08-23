"""Regression tests for five store/learn robustness defects (audit 2026-07-25).

Each test reproduces, through the real entry points (``call_tool`` /
``handle_message`` / ``load``), a defect that existed on the previous tree:

* D1 -- a stored context WITHOUT ``name`` crashed ``multidim_analyze`` with
  ``KeyError`` once selected by detection;
* D2 -- a stored context WITHOUT ``keywords``/``axes`` crashed
  ``multidim_learn`` with ``KeyError`` (``_valid_context`` accepted what the
  consumers could not handle);
* D3 -- re-sending the same ``axes`` to ``multidim_learn`` appended duplicates
  forever (learn was not idempotent for axes, unlike traps);
* D4 -- learning ``keywords`` on ``generic`` reported success although the
  detector never keyword-matches ``generic`` (silently dead data);
* D5 -- the in-memory reset markers ``_reset_reason``/``_backup`` leaked into
  ``store.json`` when a mutation followed a corrupt-store reset.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from multidim_mcp import server, store


class RobustnessTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="phases-mdrobust-")
        self._prev = os.environ.get("MULTIDIM_MCP_HOME")
        os.environ["MULTIDIM_MCP_HOME"] = self._tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("MULTIDIM_MCP_HOME", None)
        else:
            os.environ["MULTIDIM_MCP_HOME"] = self._prev
        self._tmp.cleanup()

    def _write_store(self, mutate):
        """Load the seeded store, apply ``mutate`` to it, write it back raw
        (bypassing save-side sanitisation on purpose: these tests simulate a
        hand-edited store.json)."""
        st = store.load()
        mutate(st)
        path = store.paths.store_path()
        path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        return path


class TestD1NamelessContext(RobustnessTestBase):
    def test_nameless_context_is_invalid(self):
        # The consumers all rely on context["name"]; a context without a
        # non-empty name must not pass validation.
        self.assertFalse(store._valid_context({"description": "d", "keywords": [],
                                               "axes": [], "traps": []}))
        self.assertFalse(store._valid_context({"name": "", "description": "d",
                                               "keywords": [], "axes": [], "traps": []}))

    def test_store_with_nameless_context_is_backed_up_and_reset(self):
        path = self._write_store(lambda st: st["contexts"].append(
            {"description": "d", "keywords": ["zorglubxyz"], "axes": [], "traps": []}))
        st = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertTrue(all(c.get("name") for c in st["contexts"]))
        # analysis works again after the reset (no KeyError)
        text, is_error = server.call_tool(st, "multidim_analyze",
                                          {"subject": "subject zorglubxyz", "format": "v2"})
        self.assertFalse(is_error, text)


class TestD2MissingKeywordsAxes(RobustnessTestBase):
    def test_missing_keywords_and_axes_are_migrated_additively(self):
        # Recoverable gaps: completed with [] on load, NO reset, data kept.
        path = self._write_store(lambda st: st["contexts"].append(
            {"name": "nokw", "description": "x", "traps": []}))
        st = store.load()
        ctx = store.find_context(st, "nokw")
        self.assertIsNotNone(ctx, "context must survive the load (no reset)")
        self.assertEqual(ctx["keywords"], [])
        self.assertEqual(ctx["axes"], [])
        self.assertFalse(path.with_name(path.name + ".bak").exists())

    def test_learn_on_migrated_context_no_longer_crashes(self):
        self._write_store(lambda st: st["contexts"].append(
            {"name": "nokw", "description": "x", "traps": []}))
        st = store.load()
        text, is_error = server.call_tool(st, "multidim_learn",
                                          {"context": "nokw", "keywords": ["abc"],
                                           "axes": [{"name": "A", "question": "q"}]})
        self.assertFalse(is_error, text)
        fresh = store.load()
        ctx = store.find_context(fresh, "nokw")
        self.assertIn("abc", ctx["keywords"])
        self.assertEqual([a["name"] for a in ctx["axes"]], ["A"])


class TestD3AxisDeduplication(RobustnessTestBase):
    def test_same_learn_three_times_keeps_one_axis(self):
        ax = [{"name": "AxeA", "question": "q?", "sublenses": ["s1"]}]
        for _ in range(3):
            text, is_error = server.call_tool(store.load(), "multidim_learn",
                                              {"context": "newctx", "axes": ax})
            self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "newctx")
        self.assertEqual([a["name"] for a in ctx["axes"]], ["AxeA"])

    def test_resent_axis_updates_in_place(self):
        # Same axis name = same axis: question/sublenses are refreshed,
        # never appended as a sibling duplicate.
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "AxeA", "question": "old"}]})
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "AxeA", "question": "new",
                                                    "sublenses": ["s"]}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(len(ctx["axes"]), 1)
        self.assertEqual(ctx["axes"][0]["question"], "new")
        self.assertEqual(ctx["axes"][0]["sublenses"], ["s"])

    def test_partial_resend_preserves_omitted_fields(self):
        # Review finding (2026-07-25): re-sending {name, question} only must
        # NOT erase the stored sublenses (nor question, when only sublenses
        # are re-sent). Omitted = untouched, provided = refreshed.
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "A", "question": "q0",
                                                    "sublenses": ["s1", "s2"]}]})
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "A", "question": "q1"}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(len(ctx["axes"]), 1)
        self.assertEqual(ctx["axes"][0]["question"], "q1")
        self.assertEqual(ctx["axes"][0]["sublenses"], ["s1", "s2"])
        # symmetric: re-send sublenses only, question survives
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "A", "sublenses": ["s3"]}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(ctx["axes"][0]["question"], "q1")
        self.assertEqual(ctx["axes"][0]["sublenses"], ["s3"])

    def test_legacy_duplicate_axes_are_collapsed_on_learn(self):
        # a store written by the OLD
        # buggy extend may already hold two axes named 'A'. by_name indexed
        # only one of them, so a re-learn updated one twin and left the other
        # forever. merge_axes now collapses legacy duplicates at the write
        # door: first occurrence keeps its position, later twins fill the
        # fields the survivor lacks, then disappear.
        self._write_store(lambda st: st["contexts"].append(
            {"name": "legacy", "description": "d", "keywords": [], "traps": [],
             "axes": [{"name": "A", "question": "q1", "sublenses": []},
                      {"name": "B", "question": "qb", "sublenses": []},
                      {"name": "A", "question": "", "sublenses": ["s1"]}]}))
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy",
                                           "axes": [{"name": "A", "question": "q2"}]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "legacy")
        self.assertEqual([a["name"] for a in ctx["axes"]], ["A", "B"])
        # survivor kept its position, took the update, recovered the twin's
        # sublenses it lacked
        self.assertEqual(ctx["axes"][0]["question"], "q2")
        self.assertEqual(ctx["axes"][0]["sublenses"], ["s1"])

    def test_whitespace_only_axis_name_is_refused(self):
        # '   ' passed the non-empty
        # check unstripped, dodged the dedup (blank names are not identities)
        # and accumulated a duplicate per learn. Refused at the door now.
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "axes": [{"name": "   "}]})
        self.assertTrue(is_error)
        self.assertIn("name", text)

    def test_axis_name_is_stored_stripped(self):
        # ' A ' and 'A' are one identity: stored stripped, deduped together.
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": " A ", "question": "q1"}]})
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "A", "question": "q2"}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual([a["name"] for a in ctx["axes"]], ["A"])
        self.assertEqual(ctx["axes"][0]["question"], "q2")

    def test_legacy_unstripped_axis_name_merges_with_stripped(self):
        # a historical axis stored as
        # ' A ' plus a re-learn of 'A' produced two axes (dedup indexed the
        # unstripped name). One identity now: survivor renamed 'A', updated.
        self._write_store(lambda st: st["contexts"].append(
            {"name": "legacy2", "description": "d", "keywords": [], "traps": [],
             "axes": [{"name": " A ", "question": "q1", "sublenses": ["s"]}]}))
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy2",
                                           "axes": [{"name": "A", "question": "q2"}]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "legacy2")
        self.assertEqual([a["name"] for a in ctx["axes"]], ["A"])
        self.assertEqual(ctx["axes"][0]["question"], "q2")
        self.assertEqual(ctx["axes"][0]["sublenses"], ["s"])

    def test_anonymous_axes_are_preserved_by_dedup(self):
        # two axes with an empty name in
        # a hand-edited store were both indexed under the same key by the
        # legacy dedup, destroying all but the first. Anonymous axes are not
        # identities: they must survive any learn untouched.
        self._write_store(lambda st: st["contexts"].append(
            {"name": "anon", "description": "d", "keywords": [], "traps": [],
             "axes": [{"name": "", "question": "q1", "sublenses": []},
                      {"name": "", "question": "q2", "sublenses": []}]}))
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "anon", "keywords": ["k"]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "anon")
        self.assertEqual([a["question"] for a in ctx["axes"]], ["q1", "q2"])

    def test_created_axis_carries_full_shape(self):
        # Defaults are filled at creation: a stored axis always has question
        # and sublenses even when the learn call omitted them.
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "Bare"}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(ctx["axes"][0], {"name": "Bare", "question": "",
                                          "sublenses": []})

    def test_distinct_axes_are_kept(self):
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "axes": [{"name": "A", "question": "qa"},
                                                   {"name": "B", "question": "qb"}]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual([a["name"] for a in ctx["axes"]], ["A", "B"])

    def test_creation_message_counts_stored_axes_not_sent(self):
        # two same-name axes in ONE
        # create call collapse to one; the message must report the stored
        # count, not "created with 2 axes".
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c2",
                                           "axes": [{"name": "A", "question": "q1"},
                                                    {"name": "A", "question": "q2"}]})
        self.assertFalse(is_error, text)
        # exact up to the period: "created with 11 axes." must NOT match
        self.assertIn("created with 1 axes.", text)
        ctx = store.find_context(store.load(), "c2")
        self.assertEqual(len(ctx["axes"]), 1)


class TestD4GenericKeywords(RobustnessTestBase):
    def test_keywords_only_on_generic_is_an_actionable_error(self):
        # generic is never keyword-matched: storing keywords there is dead
        # data, so a keywords-only learn must fail loudly, not "succeed".
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "generic",
                                           "keywords": ["motrarissime"]})
        self.assertTrue(is_error)
        self.assertIn("generic", text)
        fresh = store.load()
        self.assertNotIn("motrarissime", store.find_context(fresh, "generic")["keywords"])

    def test_keywords_plus_invalid_trap_on_generic_is_still_an_error(self):
        # keywords + traps=[{}] slipped
        # past the keywords-only gate as "mixed", the invalid trap was then
        # refused by upsert_traps, and the call reported success while
        # learning NOTHING. Only a valid trap makes the call mixed.
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "generic",
                                           "keywords": ["mort"], "traps": [{}]})
        self.assertTrue(is_error)
        self.assertIn("generic", text)
        fresh = store.load()
        self.assertNotIn("mort", store.find_context(fresh, "generic")["keywords"])

    def test_keywords_plus_collision_refused_trap_on_generic_is_an_error(self):
        # a trap VALID in shape can
        # still be refused by upsert_traps (id/statement collision). If the
        # keywords were dropped and every trap failed, the call learned
        # nothing: the decision must fall AFTER upsert, under the lock.
        base = {"mandatory_question": "q?", "triggers": ["zz"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "generic",
                          "traps": [dict(base, trap_id="id1", statement="lesson one"),
                                    dict(base, trap_id="id2", statement="lesson two")]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "generic",
                                           "keywords": ["mort"],
                                           "traps": [dict(base, trap_id="id1",
                                                          statement="lesson two")]})
        self.assertTrue(is_error)
        self.assertIn("nothing learned", text)
        fresh = store.load()
        generic = store.find_context(fresh, "generic")
        self.assertNotIn("mort", generic["keywords"])
        stmts = sorted(t["statement"] for t in generic["traps"])
        self.assertEqual(stmts, ["lesson one", "lesson two"])

    def test_keyword_plus_identical_trap_resend_on_generic_is_a_noop_success(self):
        # re-sending an IDENTICAL trap
        # is a legitimate idempotent no-op (0 added, 0 updated, 0 refused),
        # not a failure: with a dropped keyword it must stay a success with
        # the ignored-keywords note, not flip to isError.
        trap = {"trap_id": "tid", "statement": "same lesson",
                "mandatory_question": "q?", "triggers": ["zz"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "generic", "traps": [trap]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "generic",
                                           "keywords": ["mort"], "traps": [trap]})
        self.assertFalse(is_error, text)
        self.assertIn("ignored", text)
        fresh = store.load()
        generic = store.find_context(fresh, "generic")
        self.assertNotIn("mort", generic["keywords"])
        self.assertEqual(len([t for t in generic["traps"]
                              if t["trap_id"] == "tid"]), 1)

    def test_mixed_learn_on_generic_succeeds_with_note_and_drops_keywords(self):
        # Axes on generic are legitimate; only the keywords are dead. The call
        # succeeds (scripted callers keep working) but says what was ignored.
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "generic",
                                           "keywords": ["motrarissime"],
                                           "axes": [{"name": "AxeG", "question": "q"}]})
        self.assertFalse(is_error, text)
        self.assertIn("ignored", text)
        fresh = store.load()
        generic = store.find_context(fresh, "generic")
        self.assertNotIn("motrarissime", generic["keywords"])
        self.assertIn("AxeG", [a["name"] for a in generic["axes"]])


class TestFastPathTrapMigration(RobustnessTestBase):
    def test_trap_without_active_is_migrated_despite_fast_path(self):
        # a FULLY migrated store plus a
        # valid trap lacking 'active' slipped through the lock-free fast path
        # ("traps" key present), the active=True migration never ran, and
        # select_traps -- which requires active is True -- silently never
        # injected the lesson. The fast path must defer to the locked
        # migration whenever any trap misses 'active'.
        from multidim_mcp import frames
        trap = {"trap_id": "t1", "statement": "lesson", "mandatory_question": "asked?",
                "triggers": ["zorglubxyz"]}  # no 'active'
        self._write_store(lambda st: st["contexts"].append(
            {"name": "tctx", "description": "d", "keywords": [], "axes": [],
             "traps": [trap]}))
        st = store.load()
        loaded = store.find_context(st, "tctx")["traps"][0]
        self.assertIs(loaded.get("active"), True)
        selected = frames.select_traps(store.find_context(st, "tctx"),
                                       "subject zorglubxyz")
        self.assertEqual([t["trap_id"] for t in selected], ["t1"])


class TestD5PrivateKeysNeverPersisted(RobustnessTestBase):
    def test_reset_markers_do_not_leak_to_disk_through_mutate(self):
        path = store.paths.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        # learn right after the corrupt-store reset: mutate() persists the
        # reset store; the in-memory markers must not follow it to disk
        text, is_error = server.call_tool({"version": 1, "contexts": []},
                                          "multidim_learn",
                                          {"context": "c1", "keywords": ["k"]})
        self.assertFalse(is_error, text)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        leaked = [k for k in on_disk if k in store.PRIVATE_MARKERS]
        self.assertEqual(leaked, [])

    def test_markers_persisted_by_prefix_version_are_scrubbed_on_load(self):
        # markers written to disk by a
        # PRE-FIX save() passed the fast path forever. load() must fall to the
        # locked path, scrub them from memory AND re-write the clean copy.
        path = self._write_store(lambda st: st.update(
            {"_reset_reason": "old", "_backup": "C:/somewhere/store.json.bak"}))
        st = store.load()
        self.assertNotIn("_reset_reason", st)
        self.assertNotIn("_backup", st)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("_reset_reason", on_disk)
        self.assertNotIn("_backup", on_disk)

    def test_unknown_underscore_key_is_not_silently_dropped(self):
        # only the KNOWN internal
        # markers are filtered; a caller extension key must survive save().
        st = store.load()
        st["_vendor_extension"] = {"kept": True}
        store.save(st)
        on_disk = json.loads(store.paths.store_path().read_text(encoding="utf-8"))
        self.assertEqual(on_disk.get("_vendor_extension"), {"kept": True})
        self.assertNotIn("_reset_reason", on_disk)
        self.assertNotIn("_backup", on_disk)

    def test_reset_markers_still_reported_in_memory(self):
        # The markers stay useful to the caller of load(): only the DISK copy
        # must be clean.
        path = store.paths.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        st = store.load()
        self.assertIn("_reset_reason", st)
        self.assertIn("_backup", st)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("_reset_reason", on_disk)
        self.assertNotIn("_backup", on_disk)


class TestOperatorAwareNormalization(RobustnessTestBase):
    def test_opposite_comparison_traps_stay_distinct(self):
        # dropping operator
        # characters from the dedup key made 'retry_count < 3' and
        # 'retry_count > 3' one key -- the second lesson silently replaced
        # the first. Both must coexist.
        base = {"mandatory_question": "checked?", "triggers": ["retry"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement="retry_count < 3")]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c",
                                           "traps": [dict(base, statement="retry_count > 3")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        stmts = sorted(t["statement"] for t in ctx["traps"])
        self.assertEqual(stmts, ["retry_count < 3", "retry_count > 3"])

    def test_opposite_comparisons_are_not_normalized_equal(self):
        # same bug class in the validator: 'x < 3' vs 'x > 3' must never be
        # judged the SAME idea (normalized equality drives REJECT verdicts)
        from multidim_mcp import validate
        self.assertFalse(validate.normalized_equal("x < 3", "x > 3"))
        self.assertTrue(validate.normalized_equal("x < 3", "x  <  3"))

    def test_numeric_sign_is_preserved(self):
        # the minus sign was
        # dropped, so 'x < 3' and 'x < -3' collapsed to one key. A sign glued
        # to a number survives; a word-internal hyphen stays a separator.
        from multidim_mcp import frames, validate
        self.assertNotEqual(frames.normalize_statement("x < 3"),
                            frames.normalize_statement("x < -3"))
        self.assertFalse(validate.normalized_equal("x < 3", "x < -3"))
        self.assertEqual(frames.normalize_statement("fail-closed design"),
                         frames.normalize_statement("fail closed design"))

    def test_sign_glued_to_a_word_is_preserved(self):
        # with no space,
        # 'limit+3' and 'limit-3' both folded to 'limit 3' -- the sign was
        # only kept when it started a token. Opposite lessons again.
        from multidim_mcp import frames, validate
        self.assertNotEqual(frames.normalize_statement("limit+3"),
                            frames.normalize_statement("limit-3"))
        self.assertFalse(validate.normalized_equal("limit+3", "limit-3"))
        # a hyphen NOT followed by a digit remains a separator
        self.assertEqual(frames.normalize_statement("fail-closed"),
                         frames.normalize_statement("fail closed"))

    def test_opposite_formulas_stay_distinct(self):
        # a spaced '+' and a
        # spaced '-' both fell through as separators, so 'price = cost + tax'
        # and 'price = cost - tax' shared one key. The validator called the
        # second an ALTERNATIVE_DUPLICATES_PRIMARY, and upsert_traps silently
        # replaced the first trap with its opposite.
        from multidim_mcp import frames, validate
        plus, minus = "price = cost + tax", "price = cost - tax"
        self.assertNotEqual(frames.normalize_statement(plus),
                            frames.normalize_statement(minus))
        self.assertFalse(validate.normalized_equal(plus, minus))
        # times vs divided-by too, in ASCII and in their Unicode look-alikes
        self.assertFalse(validate.normalized_equal("rate = a * b", "rate = a / b"))
        self.assertFalse(validate.normalized_equal("rate = a " + chr(0x00D7) + " b",
                                                   "rate = a " + chr(0x00F7) + " b"))
        # ...and the hyphenated word is still ONE idea, not a subtraction
        self.assertTrue(validate.normalized_equal("fail-closed design",
                                                  "fail closed design"))

    def test_opposite_formula_traps_both_survive(self):
        # the same defect, seen through the real learn door
        base = {"mandatory_question": "checked?", "triggers": ["price"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c",
                          "traps": [dict(base, statement="price = cost + tax")]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement="price = cost - tax")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        stmts = sorted(t["statement"] for t in ctx["traps"])
        self.assertEqual(stmts, ["price = cost + tax", "price = cost - tax"])

    def test_spacing_around_an_operator_does_not_change_the_key(self):
        # a glued '+' was
        # claimed by the technical-identifier rule, so 'cost+tax' keyed as
        # 'cost+ tax' and never matched 'cost + tax' -- a plain duplicate came
        # back as a WARNING instead of a REJECT. '+' needs to be DOUBLED to be
        # part of a name.
        from multidim_mcp import frames, validate
        self.assertTrue(validate.normalized_equal("price=cost+tax",
                                                  "price = cost + tax"))
        self.assertTrue(validate.normalized_equal("a*b", "a * b"))
        # named identifiers keep their punctuation, and stay distinct
        self.assertEqual(frames.normalize_statement("use c++ rules"),
                         "use c++ rules")
        self.assertFalse(validate.normalized_equal("use c++ rules",
                                                   "use c# rules"))
        self.assertEqual(frames.normalize_statement("notepad++ crashes"),
                         "notepad++ crashes")
        # a lone hyphen stays lexical: 'cost-tax' is not read as a subtraction
        self.assertFalse(validate.normalized_equal("cost-tax", "cost - tax"))

    def test_sign_after_an_operand_reads_as_an_addition(self):
        # '+3' was always
        # lexed as a signed number, so 'x=x+3' keyed as 'x = x +3' and never
        # matched 'x = x + 3'. After an operand a sign is an operator.
        from multidim_mcp import validate
        for variant in ("x=x+3", "x = x +3", "x = x + 3"):
            self.assertTrue(validate.normalized_equal(variant, "x = x + 3"),
                            variant)
        self.assertFalse(validate.normalized_equal("x = x + 3", "x = x - 3"))
        # after an OPERATOR the sign still belongs to the number, so a
        # threshold and its negative stay opposite lessons
        self.assertFalse(validate.normalized_equal("limit < -3", "limit < 3"))
        self.assertFalse(validate.normalized_equal("limit < -.5", "limit < .5"))

    def test_unary_minus_on_a_word_is_not_dropped(self):
        # a '-' glued to a
        # WORD matched no rule, so it was discarded as a separator and
        # 'result = -input' keyed exactly like 'result = input'. The negation
        # vanished, and learning one lesson overwrote its opposite.
        from multidim_mcp import validate
        self.assertFalse(validate.normalized_equal("result = -input",
                                                   "result = input"))
        self.assertFalse(validate.normalized_equal("-input is rejected",
                                                   "input is rejected"))
        # a hyphen WITH a letter before it is still a compound word
        self.assertTrue(validate.normalized_equal("fail-closed", "fail closed"))
        self.assertTrue(validate.normalized_equal("e-mail sent", "e mail sent"))

    def test_unary_minus_traps_both_survive(self):
        # the same defect through the real learn door
        base = {"mandatory_question": "checked?", "triggers": ["result"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c",
                          "traps": [dict(base, statement="result = input")]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement="result = -input")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        stmts = sorted(t["statement"] for t in ctx["traps"])
        self.assertEqual(stmts, ["result = -input", "result = input"])

    def test_opposite_logical_rules_stay_distinct(self):
        # logical operators
        # fell through as separators, so 'admin && owner' and 'admin || owner'
        # -- one demands both, the other either -- shared a key and the
        # validator called the alternative a duplicate of the primary.
        from multidim_mcp import frames, validate
        self.assertNotEqual(frames.normalize_statement("admin && owner"),
                            frames.normalize_statement("admin || owner"))
        self.assertFalse(validate.normalized_equal("admin && owner",
                                                   "admin || owner"))
        self.assertFalse(validate.normalized_equal("admin & owner",
                                                   "admin | owner"))
        # a negation glued to a word inverts the rule just as plainly
        self.assertFalse(validate.normalized_equal("!admin can write",
                                                   "admin can write"))
        # ...while a trailing '!' is still just punctuation
        self.assertTrue(validate.normalized_equal("disk is full!", "disk is full"))

    def test_every_negated_relation_differs_from_its_positive(self):
        # NFKD turns EVERY
        # negated relation into its positive form plus a combining stroke, and
        # fold() drops combining marks, so 'limit NOT-LESS-THAN 3' keyed as
        # 'limit < 3' -- the opposite lesson, silently overwriting it. 45 code
        # points carry that stroke, so they are detected by decomposition
        # instead of listed one by one.
        import unicodedata
        from multidim_mcp import validate
        overlays = (chr(0x0338), chr(0x0337))
        checked = 0
        for code in range(0x2000, 0x2C00):
            ch = chr(code)
            decomposed = unicodedata.normalize("NFKD", ch)
            if not any(o in decomposed for o in overlays):
                continue
            positive = "".join(c for c in decomposed
                               if not unicodedata.combining(c))
            self.assertFalse(
                validate.normalized_equal("limit " + ch + " 3",
                                          "limit " + positive + " 3"),
                "%s collides with its positive form" % hex(code))
            checked += 1
        self.assertGreater(checked, 40, "the sweep found almost nothing")

    def test_unicode_not_equal_still_matches_its_ascii_spelling(self):
        # the general rule must not break the ASCII equivalences: '!=' and its
        # Unicode twin are ONE lesson, and folding stays explicit for those
        from multidim_mcp import validate
        self.assertTrue(validate.normalized_equal("x != 3",
                                                  "x " + chr(0x2260) + " 3"))
        self.assertTrue(validate.normalized_equal("x <= 3",
                                                  "x " + chr(0x2264) + " 3"))

    def test_unicode_logic_symbols_are_kept(self):
        # logic symbols
        # written in Unicode were dropped as separators, so 'user AND admins'
        # and 'user OR admins' keyed alike and one trap replaced the other.
        # The NEGATED relations need folding BEFORE fold(), since NFKD turns
        # them into their positive form plus a combining slash.
        from multidim_mcp import validate
        AND, OR = chr(0x2227), chr(0x2228)
        self.assertFalse(validate.normalized_equal("user " + AND + " admins",
                                                   "user " + OR + " admins"))
        # the Unicode spelling and its ASCII twin are the SAME lesson
        self.assertTrue(validate.normalized_equal("user " + AND + " admins",
                                                  "user && admins"))
        for positive, negative in ((0x2208, 0x2209), (0x2261, 0x2262),
                                   (0x2282, 0x2284), (0x2203, 0x2204)):
            yes = "x " + chr(positive) + " y"
            no = "x " + chr(negative) + " y"
            self.assertFalse(validate.normalized_equal(yes, no),
                             "%s vs %s" % (hex(positive), hex(negative)))
        # a bare NOT SIGN is not punctuation either
        self.assertFalse(validate.normalized_equal(chr(0x00AC) + " admin",
                                                   "admin"))

    def test_unicode_logic_traps_both_survive(self):
        base = {"mandatory_question": "checked?", "triggers": ["user"]}
        both = "user " + chr(0x2227) + " admins"
        either = "user " + chr(0x2228) + " admins"
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement=both)]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement=either)]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(sorted(t["statement"] for t in ctx["traps"]),
                         sorted([both, either]))

    def test_unicode_symbols_of_category_s_are_kept(self):
        # the pattern is a
        # whitelist, so any symbol it did not list was dropped -- 'status OK'
        # and 'status KO' written with check and cross marks keyed alike, and
        # one trap replaced the other. Whole category S is kept now, rather
        # than an enumeration that keeps leaking.
        from multidim_mcp import validate
        for yes, no in ((0x2713, 0x2717),    # CHECK MARK / BALLOT X
                        (0x2611, 0x2612),    # BALLOT BOX WITH CHECK / WITH X
                        (0x2191, 0x2193)):   # UPWARDS / DOWNWARDS ARROW
            self.assertFalse(
                validate.normalized_equal("status " + chr(yes),
                                          "status " + chr(no)),
                "%s vs %s" % (hex(yes), hex(no)))
            self.assertFalse(validate.normalized_equal("status " + chr(yes),
                                                       "status"))
        # Unicode PUNCTUATION stays ignored: typographic quotes and an em dash
        # are ornaments, not meaning, and folding them away is what lets two
        # spellings of one lesson still match.
        self.assertTrue(validate.normalized_equal(
            "the " + chr(0x00AB) + " final status " + chr(0x00BB),
            "the final status"))

    def test_symbol_traps_both_survive(self):
        base = {"mandatory_question": "checked?", "triggers": ["status"]}
        ok, ko = "status " + chr(0x2713), "status " + chr(0x2717)
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement=ok)]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement=ko)]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(sorted(t["statement"] for t in ctx["traps"]),
                         sorted([ok, ko]))

    def test_a_run_of_minuses_is_not_one_minus(self):
        # in 'x--y' the first
        # minus had a letter before it, so no rule claimed it and it was
        # dropped -- the key collapsed onto 'x - y', a different operation.
        # A RUN of minuses is never a compound word, so it is kept whole.
        from multidim_mcp import frames, validate
        self.assertFalse(validate.normalized_equal("result = x--y",
                                                   "result = x - y"))
        # spacing around the run does not matter, it is the same expression
        self.assertTrue(validate.normalized_equal("result = x--y",
                                                  "result = x -- y"))
        # and a SINGLE hyphen between words is still lexical
        self.assertTrue(validate.normalized_equal("fail-closed", "fail closed"))
        # a signed number after an operator keeps its sign, as before
        self.assertEqual(frames.normalize_statement("limit < -3"), "limit < -3")

    def test_negation_before_a_group_or_a_relation_is_kept(self):
        # the negation rule
        # required a LETTER after the '!', so it was dropped in
        # '!(admin || owner)' -- which then keyed like the un-negated rule and
        # merged two opposite lessons. What follows a negation may be a group
        # or a relation just as well as a word.
        from multidim_mcp import validate
        self.assertFalse(validate.normalized_equal("!(admin || owner)",
                                                   "(admin || owner)"))
        self.assertFalse(validate.normalized_equal("!" + chr(0x2208) + " set",
                                                   chr(0x2208) + " set"))
        self.assertFalse(validate.normalized_equal("!admin", "admin"))
        # ...while a '!' that ends a sentence is still just punctuation
        self.assertTrue(validate.normalized_equal("disk is full!", "disk is full"))
        self.assertTrue(validate.normalized_equal("wow!!!", "wow"))
        self.assertTrue(validate.normalized_equal("hello! world", "hello world"))

    def test_trailing_punctuation_is_not_an_operator(self):
        # a bare '!' was
        # tokenized as an operator, so 'disk is full' and 'disk is full!'
        # were NOT normalized-equal -- a manifest duplicate was downgraded
        # from REJECT to WARNING. Only real comparison operators count.
        from multidim_mcp import frames, validate
        self.assertTrue(validate.normalized_equal("disk is full", "disk is full!"))
        self.assertEqual(frames.normalize_statement("disk is full!"),
                         "disk is full")
        # real operators still survive, in every spelling
        self.assertTrue(validate.normalized_equal("x!=3", "x != 3"))
        self.assertFalse(validate.normalized_equal("x < 3", "x > 3"))
        self.assertFalse(validate.normalized_equal("x <= 3", "x < 3"))

    def test_duplicate_with_trailing_bang_is_rejected(self):
        from multidim_mcp import frames, validate
        st = store.load()
        frame = frames.build_frame(st, store.find_context(st, "generic"),
                                   "subject", 0, "core")
        analysis = {
            "hypotheses": [{"hypothesis_id": "H1", "statement": "disk is full"}],
            "alternatives": [{"alternative_id": "A1", "statement": "disk is full!"}],
        }
        verdict = validate.validate_analysis(frame, analysis)
        alts = [r for r in verdict["section_results"]
                if r["section"] == "alternatives"][0]
        self.assertEqual(alts["verdict"], "REJECT")
        self.assertIn("ALTERNATIVE_DUPLICATES_PRIMARY", alts["error_codes"])

    def test_technical_identifiers_keep_their_punctuation(self):
        # 'C++' and 'C#'
        # both reduced to 'c', so a legitimate alternative was REJECTED as a
        # duplicate of the primary hypothesis. The punctuation IS the name.
        from multidim_mcp import frames, validate
        self.assertNotEqual(frames.normalize_statement("Use C++ compiler rules"),
                            frames.normalize_statement("Use C# compiler rules"))
        self.assertFalse(validate.normalized_equal("Use C++ compiler rules",
                                                   "Use C# compiler rules"))
        self.assertIn("node.js", frames.normalize_statement("deploy node.js worker"))

    def test_technical_identifier_traps_stay_distinct(self):
        base = {"mandatory_question": "checked?", "triggers": ["compiler"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c",
                          "traps": [dict(base, statement="use C++ compiler rules")]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement="use C# compiler rules")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(len(ctx["traps"]), 2)

    def test_unicode_math_signs_are_normalized(self):
        # U+2212 MINUS SIGN
        # is not '-', so 'x < MINUS 3' dropped its sign and collapsed onto
        # 'x < 3' -- the opposite lesson. Unicode look-alikes now fold to
        # their ASCII meaning. Source stays pure ASCII (code points).
        from multidim_mcp import frames, validate
        minus, le, ge, ne = chr(0x2212), chr(0x2264), chr(0x2265), chr(0x2260)
        self.assertNotEqual(frames.normalize_statement("x < 3"),
                            frames.normalize_statement("x < " + minus + "3"))
        self.assertFalse(validate.normalized_equal("x < 3", "x < " + minus + "3"))
        # equivalent spellings converge
        self.assertEqual(frames.normalize_statement("x <= 3"),
                         frames.normalize_statement("x " + le + " 3"))
        self.assertEqual(frames.normalize_statement("x >= 3"),
                         frames.normalize_statement("x " + ge + " 3"))
        self.assertEqual(frames.normalize_statement("x != 3"),
                         frames.normalize_statement("x " + ne + " 3"))

    def test_unicode_minus_traps_stay_distinct(self):
        minus = chr(0x2212)
        base = {"mandatory_question": "checked?", "triggers": ["seuil"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement="seuil < 3")]})
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [dict(base, statement="seuil < " + minus + "3")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(len(ctx["traps"]), 2)

    def test_leading_dot_decimals_stay_distinct(self):
        # '.5' lost its dot
        # and '-.5' lost its sign too, so 'limit < .5' and 'limit < -.5' both
        # folded to 'limit < 5'. The tokenizer now reads numbers, not chars.
        from multidim_mcp import frames, validate
        self.assertNotEqual(frames.normalize_statement("limit < .5"),
                            frames.normalize_statement("limit < -.5"))
        self.assertFalse(validate.normalized_equal("limit < .5", "limit < -.5"))

    def test_leading_dot_decimal_traps_stay_distinct(self):
        base = {"mandatory_question": "checked?", "triggers": ["limit"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement="limit < .5")]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c",
                                           "traps": [dict(base, statement="limit < -.5")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(sorted(t["statement"] for t in ctx["traps"]),
                         ["limit < -.5", "limit < .5"])

    def test_glued_sign_traps_stay_distinct(self):
        base = {"mandatory_question": "checked?", "triggers": ["limit"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement="limit+3")]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c",
                                           "traps": [dict(base, statement="limit-3")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(sorted(t["statement"] for t in ctx["traps"]),
                         ["limit+3", "limit-3"])

    def test_opposite_sign_traps_stay_distinct(self):
        base = {"mandatory_question": "checked?", "triggers": ["seuil"]}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(base, statement="seuil < 3")]})
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c",
                                           "traps": [dict(base, statement="seuil < -3")]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(sorted(t["statement"] for t in ctx["traps"]),
                         ["seuil < -3", "seuil < 3"])


class TestPunctuatedKeywords(RobustnessTestBase):
    def test_punctuated_keywords_match(self):
        # 'c++', 'c#' and
        # 'node.js' took the whole-token branch, which compares against
        # ALPHANUMERIC runs -- so they could never match. Both detection and
        # trap triggers now share one rule (bounded substring).
        for kw, subject in (("c++", "C++ templates"),
                            ("c#", "a C# service"),
                            ("node.js", "our node.js worker")):
            with self.subTest(kw=kw):
                self._write_store(lambda st, kw=kw: st["contexts"].append(
                    {"name": "lang_" + kw.replace("+", "p").replace("#", "s").replace(".", "_"),
                     "description": "d", "keywords": [kw], "axes": [], "traps": []}))
                c, score = server.detect_context(store.load(), subject)
                self.assertEqual(score, 1, "%r should match %r" % (kw, subject))
                self.assertTrue(c["name"].startswith("lang_"))
                # reset the store between sub-cases
                store.paths.store_path().unlink()

    def test_punctuated_trap_trigger_matches(self):
        from multidim_mcp import frames
        trap = {"trap_id": "t1", "statement": "s", "mandatory_question": "q?",
                "triggers": ["c++"], "active": True}
        ctx = {"name": "c", "traps": [trap]}
        self.assertEqual([t["trap_id"] for t in frames.select_traps(ctx, "C++ templates")],
                         ["t1"])

    def test_whole_token_rule_still_holds(self):
        # the alphanumeric branch is unchanged: no substring hits
        self._write_store(lambda st: st["contexts"].append(
            {"name": "postal_ctx", "description": "d", "keywords": ["postal"],
             "axes": [], "traps": []}))
        c, score = server.detect_context(store.load(), "a post about nothing")
        self.assertEqual(score, 0)
        self.assertEqual(c["name"], "generic")

    def test_seed_contexts_catch_their_ordinary_phrasings(self):
        # the whole-token rule means every inflection has to be a keyword of
        # its own: 'option' never matches 'options'. These subjects all fell
        # through to the generic grid until the seed keywords listed them.
        cases = (
            ("decision", "Should we migrate the billing service to PostgreSQL?"),
            ("decision", "What are our options for the database?"),
            ("decision", "We need to weigh the risks of this migration"),
            ("decision", "Comparing three alternatives for the payment provider"),
            ("decision", "Whether to rewrite or patch"),
            ("code_review", "This refactoring breaks two tests"),
            ("code_review", "Review these code changes for regressions"),
        )
        for expected, subject in cases:
            with self.subTest(subject=subject):
                c, score = server.detect_context(store.load(), subject)
                self.assertEqual(c["name"], expected, subject)
                self.assertGreater(score, 0)


class TestContextLookupNormalisation(RobustnessTestBase):
    def test_stored_name_with_trailing_space_is_found(self):
        # a hand-edited
        # 'legacy_ctx ' passed validation but find_context('legacy_ctx')
        # missed it -- learn then appended a DUPLICATE context.
        self._write_store(lambda st: st["contexts"].append(
            {"name": "legacy_ctx ", "description": "d", "keywords": [],
             "axes": [], "traps": []}))
        st = store.load()
        self.assertIsNotNone(store.find_context(st, "legacy_ctx"))
        text, is_error = server.call_tool(st, "multidim_learn",
                                          {"context": "legacy_ctx",
                                           "keywords": ["kw"]})
        self.assertFalse(is_error, text)
        names = [n.strip() for n in store.list_context_names(store.load())]
        self.assertEqual(names.count("legacy_ctx"), 1)

    def test_duplicate_normalised_names_are_corruption(self):
        # two contexts
        # sharing a normalised name made find_context (first match) and
        # detect_context (best score) disagree -- validate then rejected an
        # untouched frame. Duplicate identities = backup + reset.
        path = store.paths.store_path()
        self._write_store(lambda st: st["contexts"].extend([
            {"name": "dup", "description": "d", "keywords": ["alpha"],
             "axes": [], "traps": []},
            {"name": "DUP ", "description": "d", "keywords": ["beta"],
             "axes": [], "traps": []}]))
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIsNone(store.find_context(reloaded, "dup"))
        self.assertIn("generic", store.list_context_names(reloaded))

    def test_case_variant_generic_does_not_trigger_a_reset(self):
        # a store whose
        # seed was renamed ' Generic ' was FOUND by find_context but failed
        # the exact 'generic' presence check -- the reset destroyed the
        # user's own contexts. Both sides normalise now.
        path = store.paths.store_path()
        def rename(st):
            for c in st["contexts"]:
                if c["name"] == "generic":
                    c["name"] = " Generic "
            st["contexts"].append({"name": "mine", "description": "d",
                                   "keywords": ["mykw"], "axes": [], "traps": []})
        self._write_store(rename)
        reloaded = store.load()
        self.assertFalse(path.with_name(path.name + ".bak").exists())
        self.assertIsNotNone(store.find_context(reloaded, "mine"))
        self.assertIsNotNone(store.find_context(reloaded, "generic"))
        self.assertNotIn("_reset_reason", reloaded)

    def test_lookup_is_case_folded(self):
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "MyCtx", "keywords": ["kw"]})
        st = store.load()
        for spelling in ("myctx", "MYCTX", " MyCtx "):
            with self.subTest(spelling=spelling):
                self.assertIsNotNone(store.find_context(st, spelling))


class TestFalsyParameterDefaults(RobustnessTestBase):
    def test_non_string_depth_is_refused_not_defaulted(self):
        # `args.get("depth")
        # or "deep"` swallowed every falsy value, so depth=false silently
        # produced a DEEP grid instead of an error. Only a missing key
        # defaults.
        for bad in (False, 0, "", [], {}):
            with self.subTest(depth=bad):
                text, is_error = server.call_tool(
                    store.load(), "multidim_analyze",
                    {"subject": "subject", "depth": bad})
                self.assertTrue(is_error, "depth=%r must be refused" % (bad,))
                self.assertIn("depth", text)

    def test_missing_depth_still_defaults_to_deep(self):
        text, is_error = server.call_tool(store.load(), "multidim_analyze",
                                          {"subject": "subject", "format": "v2"})
        self.assertFalse(is_error, text)
        self.assertEqual(json.loads(text)["depth"], "deep")

    def test_non_string_context_is_refused_not_auto_detected(self):
        # context=false /
        # 0 / [] / {} fell through to auto-detection, so a schema violation
        # quietly produced a grid for a context the caller never chose.
        for bad in (False, 0, [], {}, "   "):
            with self.subTest(context=bad):
                text, is_error = server.call_tool(
                    store.load(), "multidim_analyze",
                    {"subject": "review a code diff", "context": bad})
                self.assertTrue(is_error, "context=%r must be refused" % (bad,))
                self.assertIn("context", text)

    def test_omitted_context_still_auto_detects(self):
        text, is_error = server.call_tool(store.load(), "multidim_analyze",
                                          {"subject": "review a code diff",
                                           "format": "v2"})
        self.assertFalse(is_error, text)
        self.assertEqual(json.loads(text)["context"]["name"], "code_review")

    def test_non_string_tool_name_is_refused(self):
        resp = server.handle_message(store.load(),
                                     {"jsonrpc": "2.0", "id": 1,
                                      "method": "tools/call",
                                      "params": {"name": 123, "arguments": {}}})
        self.assertEqual(resp["error"]["code"], -32602)


class TestStdoutOwnership(RobustnessTestBase):
    def test_serve_does_not_detach_the_host_stdout(self):
        # serve() wrapped
        # sys.stdout.buffer in a NEW TextIOWrapper, which took ownership and
        # left sys.stdout detached -- any later print() in the host process
        # raised ValueError. Proven in a child process: after serve() returns,
        # print() must still work.
        import subprocess as _sp
        import sys as _sys
        code = (
            "import sys, io;"
            "sys.path.insert(0, r'src');"
            "from multidim_mcp import server;"
            "server.serve(stdin=io.StringIO(''), stdout=None,"
            "             log_stream=io.StringIO());"
            "print('PRINT_STILL_WORKS')"
        )
        out = _sp.run([_sys.executable, "-c", code], capture_output=True,
                      text=True, timeout=60,
                      cwd=str(store.paths.Path(__file__).resolve().parents[1]),
                      env={**os.environ,
                           "MULTIDIM_MCP_HOME": self._tmp.name})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("PRINT_STILL_WORKS", out.stdout)


class TestSurrogateInLearnedText(RobustnessTestBase):
    def test_lone_surrogate_in_a_description_is_persisted(self):
        # a learned
        # description carrying a lone surrogate is valid JSON input but not
        # encodable in UTF-8 -- the store write raised UnicodeEncodeError and
        # the tool answered an internal error. The store is ASCII-escaped now.
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "description": "bad " + "\ud800" + " text",
             "keywords": ["kw"]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertIn("bad", ctx["description"])
        # the file itself is pure ASCII: readable by any strict decoder
        store.paths.store_path().read_text(encoding="ascii")


class TestServerVersion(RobustnessTestBase):
    def test_announced_version_matches_the_packaging_metadata(self):
        # initialize
        # announced a hardcoded 1.1.0 while the package shipped 0.1.0 --
        # a client trusting serverInfo was simply misinformed.
        import re as _re
        import pathlib as _pathlib
        pyproject = (_pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
                     ).read_text(encoding="utf-8")
        declared = _re.search(r'^version\s*=\s*"([^"]+)"', pyproject,
                              _re.MULTILINE).group(1)
        resp = server.handle_message(store.load(),
                                     {"jsonrpc": "2.0", "id": 1,
                                      "method": "initialize", "params": {}})
        announced = resp["result"]["serverInfo"]["version"]
        # installed -> exact match; in-tree checkout -> explicit marker,
        # never a stale literal pretending to be a real release
        self.assertTrue(announced == declared or announced.endswith("+unpackaged"),
                        "announced %r vs declared %r" % (announced, declared))


class TestRequestIdShape(RobustnessTestBase):
    def test_structured_id_is_refused(self):
        # JSON-RPC 2.0
        # allows only String / Number / Null as an id. An object or array id
        # was accepted and echoed back, leaving a strict client unable to
        # correlate the response.
        for bad_id in ({}, [], [1, 2], True, False):
            with self.subTest(id=bad_id):
                resp = server.handle_message(
                    store.load(), {"jsonrpc": "2.0", "id": bad_id,
                                   "method": "ping", "params": {}})
                self.assertIn("error", resp)
                self.assertEqual(resp["error"]["code"], -32600)
                self.assertIsNone(resp["id"])

    def test_valid_id_shapes_still_work(self):
        for good_id in ("abc", 1, 1.5, None):
            with self.subTest(id=good_id):
                resp = server.handle_message(
                    store.load(), {"jsonrpc": "2.0", "id": good_id,
                                   "method": "ping", "params": {}})
                self.assertIn("result", resp)
                self.assertEqual(resp["id"], good_id)


class TestBackupNeverFollowsAnExistingPath(RobustnessTestBase):
    def test_existing_backup_path_is_never_written_through(self):
        # the corrupt-store
        # backup wrote to store.json.bak with plain write_bytes -- if that
        # path already existed (a symlink to a personal store, say), the
        # evidence dump overwrote the target. O_EXCL + the personal guard
        # make that impossible: the load fails closed instead.
        path = store.paths.store_path()
        store.load()
        victim = path.with_name("victim.json")
        victim.write_text("PRECIOUS", encoding="utf-8")
        bak = path.with_name(path.name + ".bak")
        try:
            os.symlink(str(victim), str(bak))
        except (OSError, NotImplementedError, AttributeError):
            # no symlink privilege (common on Windows): fall back to a plain
            # existing file, which O_EXCL must refuse just the same
            bak.write_text("PRECIOUS", encoding="utf-8")
        path.write_text("{ corrupt", encoding="utf-8")
        # A later fix changed the outcome, not the guarantee. Failing closed was
        # what could offer; it also meant a second corruption bricked
        # the server. The backup now takes the next free name, so the load
        # RECOVERS -- while O_EXCL still forbids writing through the existing
        # path, which is the security property being asserted here.
        reloaded = store.load()
        self.assertIn("_reset_reason", reloaded)
        self.assertEqual(bak.read_text(encoding="utf-8"), "PRECIOUS")
        self.assertNotEqual(store.paths.Path(reloaded["_backup"]), bak)
        self.assertEqual(
            store.paths.Path(reloaded["_backup"]).read_text(encoding="utf-8"),
            "{ corrupt")
        # ...and the victim behind the symlink is untouched either way
        self.assertEqual(victim.read_text(encoding="utf-8"), "PRECIOUS")


class TestLockPathGuard(RobustnessTestBase):
    def test_lock_path_is_guarded_like_every_other_write(self):
        # the lock file was
        # opened without the personal-store guard, so a symlinked
        # store.json.lock had its target created OUTSIDE the tripwire.
        personal = store.paths.Path(
            os.path.expanduser(os.path.join("~", ".multidim", "probe.lock")))
        with self.assertRaises(RuntimeError):
            with store._file_lock(personal):
                pass
        self.assertFalse(personal.exists(), "nothing may be created there")

    def test_normal_lock_still_works(self):
        lock = store.paths.store_path().with_name("store.json.lock")
        with store._file_lock(lock):
            self.assertTrue(lock.exists())


class TestStoreJsonConstants(RobustnessTestBase):
    def test_nan_in_stored_trap_is_corruption_not_a_valid_store(self):
        # a store holding
        # severity: NaN loaded fine, the trap passed _valid_trap, and the
        # frame carried NaN onto the wire where no strict parser can read it.
        # Invalid JSON constants are corruption: backup + reset.
        path = store.paths.store_path()
        st = store.load()
        st["contexts"].append(
            {"name": "tctx", "description": "d", "keywords": [], "axes": [],
             "traps": [{"trap_id": "t1", "statement": "s", "mandatory_question": "q?",
                        "triggers": ["zz"], "severity": float("nan"), "active": True}]})
        # write it the way a hand edit would (Python's permissive dump)
        path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIsNone(store.find_context(reloaded, "tctx"))
        self.assertIn("_reset_reason", reloaded)

    def test_overflowing_literal_in_store_is_corruption(self):
        # 1e999 is valid
        # JSON syntax, so parse_constant never saw it; the store loaded and
        # a frame carried Infinity onto the wire. Same treatment as NaN.
        path = store.paths.store_path()
        store.load()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["contexts"].append(
            {"name": "tctx", "description": "d", "keywords": [], "axes": [],
             "traps": [{"trap_id": "t1", "statement": "s", "mandatory_question": "q?",
                        "triggers": ["zz"], "severity": 1e999, "active": True}]})
        # 1e999 written as a literal, exactly as a hand edit would
        path.write_text(json.dumps(raw).replace('"severity": Infinity',
                                                '"severity": 1e999'),
                        encoding="utf-8")
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIsNone(store.find_context(reloaded, "tctx"))

    def test_deeply_nested_store_is_corruption_not_a_crash(self):
        # a store nested
        # thousands of levels deep raised RecursionError inside json.loads --
        # uncaught on BOTH read paths, so load() crashed instead of backing
        # the file up. Same treatment as any other unreadable store.
        path = store.paths.store_path()
        store.load()
        path.write_bytes(b'{"version":1,"contexts":' + b"[" * 5000
                         + b"]" * 5000 + b"}")
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIn("generic", store.list_context_names(reloaded))
        self.assertIn("_reset_reason", reloaded)

    def test_save_refuses_to_write_nan(self):
        st = store.load()
        st["contexts"][0]["axes"].append(
            {"name": "A", "question": "q", "sublenses": [], "weight": float("inf")})
        with self.assertRaises(ValueError):
            store.save(st)


class TestJsonConstantsRefused(RobustnessTestBase):
    def test_nan_id_is_a_parse_error_and_serving_continues(self):
        # Python's decoder
        # accepts NaN/Infinity (not valid JSON); the value was echoed back in
        # 'id' and no strict client could read the response.
        import io as _io
        bad = '{"jsonrpc":"2.0","id":NaN,"method":"ping","params":{}}\n'
        good = '{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.StringIO(bad + good), stdout=out,
                     log_stream=_io.StringIO())
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["error"]["code"], -32700)
        self.assertEqual(json.loads(lines[1])["id"], 2)
        # every wire line is re-readable by a STRICT parser (no NaN, no
        # surrogate): this is the property the fix protects
        for l in lines:
            json.loads(l, parse_constant=server._reject_json_constant)

    def test_overflowing_float_literal_is_refused_and_loop_survives(self):
        # 1e999 is valid
        # JSON SYNTAX, so parse_constant never saw it; Python decoded it to
        # inf, which then hit allow_nan=False on the way out and killed the
        # loop with an uncaught ValueError -- the next ping got no answer.
        import io as _io
        bad = '{"jsonrpc":"2.0","id":1e999,"method":"ping","params":{}}\n'
        good = '{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.StringIO(bad + good), stdout=out,
                     log_stream=_io.StringIO())
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["error"]["code"], -32700)
        self.assertEqual(json.loads(lines[1])["id"], 2)

    def test_deeply_nested_payload_does_not_kill_the_loop(self):
        # thousands of
        # nested arrays raised RecursionError inside json.loads, which was
        # not caught -- the whole server died and the next ping went
        # unanswered. A malformed request must never be a server fault.
        import io as _io
        bomb = "[" * 5000 + "]" * 5000 + "\n"
        good = '{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.StringIO(bomb + good), stdout=out,
                     log_stream=_io.StringIO())
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        # WHICH error depends on the interpreter's stack, not on us: where the
        # decoder gives up it is a parse error (-32700), where 5000 levels fit
        # the payload parses into a list, which is not a request (-32600).
        # Pinning one code made the suite pass on Windows and fail on Linux.
        # What must hold everywhere is that this is answered as a CLIENT fault
        # and the loop survives it.
        self.assertIn(json.loads(lines[0])["error"]["code"], (-32700, -32600))
        self.assertEqual(json.loads(lines[1])["id"], 2)

    def test_infinity_in_params_is_refused(self):
        import io as _io
        bad = '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":Infinity}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.StringIO(bad), stdout=out, log_stream=_io.StringIO())
        first = json.loads([l for l in out.getvalue().splitlines() if l.strip()][0])
        self.assertEqual(first["error"]["code"], -32700)


class TestSurrogateIdOnWire(RobustnessTestBase):
    def test_lone_surrogate_id_does_not_kill_the_server(self):
        # JSON accepts a
        # lone surrogate in "id"; echoing it back through a strict UTF-8
        # writer raised UnicodeEncodeError and killed the loop. The wire now
        # uses ensure_ascii=True, so the surrogate round-trips as a JSON
        # escape and the NEXT request is still served.
        import io as _io
        bad_id = ('{"jsonrpc":"2.0","id":"' + "\\ud800" + '","method":"ping","params":{}}\n')
        good = '{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.StringIO(bad_id + good), stdout=out,
                     log_stream=_io.StringIO())
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        second = json.loads(lines[1])
        self.assertEqual(second["id"], 2)
        self.assertIn("result", second)
        # every wire line stays pure ASCII (encodable by any strict writer)
        for l in lines:
            l.encode("ascii")


class TestBooleanScoreForgery(RobustnessTestBase):
    def test_score_true_is_refused_even_with_recomputed_hash(self):
        # bool is an int
        # subclass, so score=true rode through every isinstance(int) check as
        # 1; with an adapted explanation and a recomputed hash the forged
        # frame validated. The type is now refused explicitly.
        from multidim_mcp import frames
        text, is_error = server.call_tool(store.load(), "multidim_analyze",
                                          {"subject": "review this code diff",
                                           "format": "v2"})
        self.assertFalse(is_error, text)
        frame = json.loads(text)
        self.assertGreaterEqual(frame["context"]["score"], 1)
        frame["context"]["score"] = True
        frame["context"]["explanation"] = "True context keyword(s) found in the subject"
        frame["frame_hash"] = frames.frame_hash_of(frame)
        text, is_error = server.call_tool(store.load(), "multidim_validate",
                                          {"frame": frame, "analysis": {}})
        self.assertTrue(is_error)
        self.assertIn("boolean", text)


class TestFrameSelfConsistency(RobustnessTestBase):
    def test_stripped_frame_with_original_hash_is_refused(self):
        # deleting
        # frame['axes'] while keeping the original frame_hash passed --
        # only the received HASH was compared to the rebuilt hash, never
        # the received BODY to its own hash.
        text, is_error = server.call_tool(store.load(), "multidim_analyze",
                                          {"subject": "neutral subject", "format": "v2"})
        self.assertFalse(is_error, text)
        frame = json.loads(text)
        del frame["axes"]
        text, is_error = server.call_tool(store.load(), "multidim_validate",
                                          {"frame": frame, "analysis": {"facts": []}})
        self.assertTrue(is_error)
        self.assertIn("tampered", text)

    def test_missing_or_edited_frame_id_is_refused(self):
        # frame_id is
        # excluded from the hash (to keep it self-verifiable), so deleting or
        # editing it slipped past the content check.
        text, _ = server.call_tool(store.load(), "multidim_analyze",
                                   {"subject": "neutral subject", "format": "v2"})
        base = json.loads(text)
        for mutate in (lambda f: f.pop("frame_id"),
                       lambda f: f.__setitem__("frame_id", "frame_forged")):
            with self.subTest(mutate=mutate):
                frame = json.loads(text)
                mutate(frame)
                out, is_error = server.call_tool(store.load(), "multidim_validate",
                                                 {"frame": frame, "analysis": {}})
                self.assertTrue(is_error)
                self.assertIn("frame_id", out)
        # sanity: the untouched frame still carries a derived id
        self.assertEqual(base["frame_id"], "frame_" + base["frame_hash"][:24])

    def test_untouched_frame_still_validates(self):
        text, _ = server.call_tool(store.load(), "multidim_analyze",
                                   {"subject": "neutral subject", "format": "v2"})
        frame = json.loads(text)
        text, is_error = server.call_tool(store.load(), "multidim_validate",
                                          {"frame": frame, "analysis": {}})
        self.assertFalse(is_error, text)  # verdict rendered (REJECTs inside), no tamper error
        self.assertIn("verdict", text)


class TestTrapActiveStrictness(RobustnessTestBase):
    def test_non_boolean_active_is_refused_not_coerced(self):
        # active: "false"
        # (a string) was silently coerced to True -- the lesson the caller
        # explicitly tried to disable stayed active. Refused loudly now.
        trap = {"statement": "lesson", "mandatory_question": "q?",
                "triggers": ["zz"], "active": "false"}
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "traps": [trap]})
        self.assertFalse(is_error, text)  # the call reports, per-trap refusal
        self.assertIn("Traps refused", text)
        self.assertIn("boolean", text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(ctx["traps"], [])

    def test_partial_resend_keeps_stored_optional_fields(self):
        # re-sending a trap
        # WITHOUT its optional fields reset severity to 'medium' and, worse,
        # re-activated a lesson the caller had explicitly disabled.
        full = {"trap_id": "t1", "statement": "lesson", "mandatory_question": "q?",
                "triggers": ["zz"], "severity": "high", "active": False}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [full]})
        partial = {"trap_id": "t1", "statement": "lesson",
                   "mandatory_question": "q?", "triggers": ["zz"]}
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "traps": [partial]})
        self.assertFalse(is_error, text)
        trap = store.find_context(store.load(), "c")["traps"][0]
        self.assertEqual(trap["severity"], "high")
        self.assertIs(trap["active"], False)

    def test_identical_partial_resend_is_a_true_noop(self):
        # with optional
        # fields now omitted, a strict dict comparison saw a difference on
        # every partial re-send and reported a phantom update.
        from multidim_mcp import frames
        ctx = {"name": "c", "traps": []}
        trap = [{"trap_id": "t1", "statement": "s", "mandatory_question": "q?",
                 "triggers": ["zz"]}]
        self.assertEqual(frames.upsert_traps(ctx, trap), (1, 0, []))
        self.assertEqual(frames.upsert_traps(ctx, trap), (0, 0, []))

    def test_frame_hash_survives_a_lone_surrogate_subject(self):
        # same round: a subject carrying a lone surrogate (valid JSON input,
        # not encodable in UTF-8) raised UnicodeEncodeError inside the hash.
        from multidim_mcp import frames
        st = store.load()
        ctx = store.find_context(st, "generic")
        frame = frames.build_frame(st, ctx, "subject " + "\ud800" + " casse", 0, "core")
        self.assertEqual(len(frame["frame_hash"]), 64)
        self.assertEqual(frames.frame_hash_of(frame), frame["frame_hash"])

    def test_explicit_resend_still_updates(self):
        full = {"trap_id": "t1", "statement": "lesson", "mandatory_question": "q?",
                "triggers": ["zz"], "severity": "high", "active": False}
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [full]})
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "traps": [dict(full, severity="low",
                                                         active=True)]})
        trap = store.find_context(store.load(), "c")["traps"][0]
        self.assertEqual(trap["severity"], "low")
        self.assertIs(trap["active"], True)

    def test_out_of_enum_severity_is_refused_not_downgraded(self):
        # 'critical' was
        # silently rewritten to 'medium' -- a lesson the caller marked as
        # serious was DOWNGRADED while the call reported success.
        trap = {"statement": "lesson", "mandatory_question": "q?",
                "triggers": ["zz"], "severity": "critical"}
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "traps": [trap]})
        self.assertFalse(is_error, text)   # per-trap refusal is reported
        self.assertIn("Traps refused", text)
        self.assertIn("severity", text)
        self.assertEqual(store.find_context(store.load(), "c")["traps"], [])

    def test_valid_severities_are_kept(self):
        for level in ("low", "medium", "high"):
            with self.subTest(level=level):
                trap = {"trap_id": "t_" + level, "statement": "lesson " + level,
                        "mandatory_question": "q?", "triggers": ["zz"],
                        "severity": level}
                text, is_error = server.call_tool(store.load(), "multidim_learn",
                                                  {"context": "c", "traps": [trap]})
                self.assertFalse(is_error, text)
        stored = {t["trap_id"]: t["severity"]
                  for t in store.find_context(store.load(), "c")["traps"]}
        self.assertEqual(stored, {"t_low": "low", "t_medium": "medium",
                                  "t_high": "high"})

    def test_absent_active_still_defaults_to_true(self):
        trap = {"statement": "lesson", "mandatory_question": "q?", "triggers": ["zz"]}
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "traps": [trap]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertIs(ctx["traps"][0]["active"], True)


class TestStrictStdinDecoding(RobustnessTestBase):
    def test_invalid_utf8_line_answers_parse_error_and_serving_continues(self):
        # errors="replace"
        # silently ALTERED a request -- an invalid byte inside a valid JSON
        # string became U+FFFD, the line parsed, and the server answered
        # -32601 for a method the client never sent. Strict per-line decoding:
        # invalid UTF-8 = -32700, and the next request still works.
        import io as _io
        bad = b'{"jsonrpc":"2.0","id":1,"method":"a\xffb","params":{}}\n'
        good = b'{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}\n'
        out = _io.StringIO()
        server.serve(stdin=_io.BytesIO(bad + good), stdout=out,
                     log_stream=_io.StringIO())
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["error"]["code"], -32700)
        self.assertIn("utf-8", first["error"]["message"])
        second = json.loads(lines[1])
        self.assertEqual(second["id"], 2)
        self.assertIn("result", second)


class TestWhitespaceKeyword(RobustnessTestBase):
    def test_whitespace_keyword_is_refused_at_learn(self):
        # keywords=[' ']
        # passed learn, then the multi-word substring branch matched EVERY
        # multi-word subject -- the context hijacked detection with score 1.
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "hijack", "keywords": [" "]})
        self.assertTrue(is_error)
        self.assertIn("keyword", text.lower())

    def test_hand_edited_whitespace_keyword_never_matches(self):
        # defence in depth at detection: a whitespace keyword already stored
        # (hand-edited store) is ignored, never a universal substring match
        self._write_store(lambda st: st["contexts"].append(
            {"name": "hijack", "description": "d", "keywords": [" "],
             "axes": [], "traps": []}))
        c, score = server.detect_context(store.load(), "alpha beta")
        self.assertEqual(c.get("name"), "generic")
        self.assertEqual(score, 0)

    def test_duplicate_keywords_are_stored_once(self):
        # ['cafe','cafe'] at
        # creation stored both, and detection counted 2 hits -- the duplicate
        # artificially beat a legitimate single-keyword context.
        cafe_accented = "caf" + chr(0xE9)  # ASCII-pure source (review pipe)
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "dupctx",
                                           "keywords": ["cafe", "cafe", cafe_accented]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "dupctx")
        self.assertEqual(ctx["keywords"], ["cafe"])
        c, score = server.detect_context(store.load(), "cafe")
        self.assertEqual((c.get("name"), score), ("dupctx", 1))

    def test_hand_edited_duplicate_keywords_count_once_in_detection(self):
        # defence in depth: duplicates already on disk never inflate the score
        cafe_accented = "caf" + chr(0xE9)
        self._write_store(lambda st: st["contexts"].append(
            {"name": "dupctx", "description": "d",
             "keywords": ["cafe", "cafe", cafe_accented], "axes": [], "traps": []}))
        c, score = server.detect_context(store.load(), "cafe decision choice")
        # decision (2 legitimate hits) must beat dupctx (1 folded hit)
        self.assertEqual(c.get("name"), "decision")

    def test_accent_variant_not_appended_on_enrichment(self):
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "keywords": ["cafe"]})
        server.call_tool(store.load(), "multidim_learn",
                         {"context": "c", "keywords": ["caf" + chr(0xE9)]})
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(ctx["keywords"], ["cafe"])

    def test_keywords_are_stored_stripped(self):
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "c", "keywords": [" Zorg "]})
        self.assertFalse(is_error, text)
        ctx = store.find_context(store.load(), "c")
        self.assertEqual(ctx["keywords"], ["zorg"])


class TestAlternativePairwiseDuplication(RobustnessTestBase):
    def _analysis(self, alt2_statement):
        return {
            "facts": [{"fact_id": "F1", "statement": "le module lit un fichier"}],
            "hypotheses": [{"hypothesis_id": "H1", "statement": "la lecture echoue",
                            "falsification_test": "lancer avec un fichier absent"}],
            "alternatives": [
                {"alternative_id": "A1", "statement": "le disque est plein"},
                {"alternative_id": "A2", "statement": alt2_statement}],
            "hidden_dependencies": [{"dependency_id": "D1", "dependency": "fs"}],
            "second_order_risks": [{"risk_id": "R1", "risk": "perte",
                                    "first_order_effect": "ecriture partielle",
                                    "second_order_effect": "corruption silencieuse"}],
            "contradictions": [{"contradiction_id": "C1", "contradiction": "atomique mais lu hors verrou"}],
            "open_questions": [{"question_id": "Q1", "question": "comportement windows"}],
            "decision_criteria": [{"criterion_id": "K1", "criterion": "zero perte"}],
            "cross_talk": {"tensions": "vitesse contre surete",
                           "blind_spots": "concurrence multi-process"},
            "premortem": {"story": "un an plus tard une bascule ecrase le store"},
            "synthesis": {"statement": "verrouiller la lecture", "references": ["F1", "H1"]},
        }

    def _frame(self, depth):
        from multidim_mcp import frames
        st = store.load()
        ctx = store.find_context(st, "generic")
        return frames.build_frame(st, ctx, "neutral subject", 0, depth)

    def test_duplicate_alternatives_rejected_deep_and_full(self):
        # A2 copying A1 under
        # a fresh identifier passed (only the vs-primary pair was checked) and
        # even satisfied min_alternatives. Both depths must reject.
        from multidim_mcp import validate
        for depth in ("deep", "full"):
            with self.subTest(depth=depth):
                verdict = validate.validate_analysis(
                    self._frame(depth), self._analysis("le disque est plein"))
                alts = [r for r in verdict["section_results"]
                        if r["section"] == "alternatives"][0]
                self.assertEqual(alts["verdict"], "REJECT")
                self.assertIn("ALTERNATIVE_DUPLICATES_ALTERNATIVE", alts["error_codes"])
                self.assertIn("NOT_ENOUGH_ALTERNATIVES", alts["error_codes"])

    def test_oversized_alternatives_list_is_refused_fast(self):
        # the pairwise
        # duplicate check is O(n^2) -- 800 alternatives took ~5 s. An
        # analysis lists distinct options, not a dump: refuse past the bound.
        import time as _time
        from multidim_mcp import validate
        big = {"alternatives": [{"alternative_id": "A%d" % i,
                                 "statement": "option %d" % i} for i in range(800)]}
        started = _time.time()
        verdict = validate.validate_analysis(self._frame("deep"), big)
        elapsed = _time.time() - started
        alts = [r for r in verdict["section_results"]
                if r["section"] == "alternatives"][0]
        self.assertIn("TOO_MANY_ALTERNATIVES", alts["error_codes"])
        self.assertEqual(alts["verdict"], "REJECT")
        self.assertLess(elapsed, 1.0, "refusal must short-circuit the O(n^2) pass")

    def test_list_at_the_bound_is_still_checked(self):
        from multidim_mcp import validate
        at_bound = {"alternatives": [{"alternative_id": "A%d" % i,
                                      "statement": "option %d" % i}
                                     for i in range(validate.MAX_ALTERNATIVES)]}
        verdict = validate.validate_analysis(self._frame("deep"), at_bound)
        alts = [r for r in verdict["section_results"]
                if r["section"] == "alternatives"][0]
        self.assertNotIn("TOO_MANY_ALTERNATIVES", alts["error_codes"])

    def test_distinct_alternatives_still_accepted(self):
        from multidim_mcp import validate
        verdict = validate.validate_analysis(
            self._frame("deep"), self._analysis("le reseau a coupe l'ecriture"))
        alts = [r for r in verdict["section_results"]
                if r["section"] == "alternatives"][0]
        self.assertNotIn("ALTERNATIVE_DUPLICATES_ALTERNATIVE", alts["error_codes"])
        self.assertNotIn("NOT_ENOUGH_ALTERNATIVES", alts["error_codes"])


class TestD6ReplaceRetry(RobustnessTestBase):
    def test_save_survives_transient_permission_error(self):
        # a transient Windows sharing violation (reader holding
        # the file for milliseconds) must be absorbed by the bounded retry.
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise PermissionError(13, "sharing violation")
            return real_replace(src, dst)

        st = store.load()
        with mock.patch.object(store, "REPLACE_BACKOFF_S", 0.001), \
                mock.patch.object(store.os, "replace", side_effect=flaky):
            store.save(st)
        self.assertGreaterEqual(calls["n"], 4)
        on_disk = json.loads(store.paths.store_path().read_text(encoding="utf-8"))
        self.assertIn("contexts", on_disk)

    def test_save_fails_closed_after_persistent_permission_error(self):
        # a PERSISTENT hold still fails with the original error after the
        # last attempt, and the temp file never lingers next to the store
        st = store.load()
        with mock.patch.object(store, "REPLACE_BACKOFF_S", 0.001), \
                mock.patch.object(store.os, "replace",
                                  side_effect=PermissionError(13, "denied")):
            with self.assertRaises(PermissionError):
                store.save(st)
        leftovers = [p.name for p in store.paths.store_path().parent.iterdir()
                     if p.name.startswith("store-")]
        self.assertEqual(leftovers, [])

    @unittest.skipUnless(sys.platform.startswith("win"),
                         "Windows file-sharing semantics")
    def test_save_succeeds_while_reader_briefly_holds_the_file(self):
        # real integration: a reader holds the store open ~0.15 s (far below
        # the ~0.9 s retry budget); save() must win once the handle closes.
        # This exact scenario failed with PermissionError before the fix.
        st = store.load()
        path = store.paths.store_path()
        started = threading.Event()

        def hold():
            with open(path, "r", encoding="utf-8") as fh:
                fh.read(5)
                started.set()
                time.sleep(0.15)

        t = threading.Thread(target=hold)
        t.start()
        self.assertTrue(started.wait(timeout=2.0))
        try:
            store.save(st)  # must retry through the reader's hold
        finally:
            t.join()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("contexts", on_disk)


class TestCallerMarkers(RobustnessTestBase):
    def test_markers_survive_learn_in_memory(self):
        # the reset markers are the caller's
        # diagnostics; the post-learn in-place refresh must not discard them
        # (the disk copy alone stays filtered).
        path = store.paths.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        st = store.load()
        self.assertIn("_reset_reason", st)
        text, is_error = server.call_tool(st, "multidim_learn",
                                          {"context": "c1", "keywords": ["k"]})
        self.assertFalse(is_error, text)
        self.assertIn("_reset_reason", st)
        self.assertIn("_backup", st)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("_reset_reason", on_disk)

    def test_markers_survive_handle_message_reload(self):
        # the per-call reload in
        # handle_message wiped the markers exactly like the learn path did --
        # the same contract applies to every in-place refresh.
        path = store.paths.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        st = store.load()
        self.assertIn("_reset_reason", st)
        resp = server.handle_message(st, {"jsonrpc": "2.0", "id": 1,
                                          "method": "tools/call",
                                          "params": {"name": "multidim_contexts",
                                                     "arguments": {}}})
        self.assertNotIn("error", resp)
        self.assertIn("_reset_reason", st)
        self.assertIn("_backup", st)

    def test_fresh_markers_from_reload_win_over_old_ones(self):
        # if the per-call reload
        # itself resets a corrupt store, ITS markers are newer than the ones
        # carried in memory -- the old ones must not overwrite them.
        st = store.load()
        st["_reset_reason"] = "OLD_MARKER"
        path = store.paths.store_path()
        path.write_text("{ corrupt again", encoding="utf-8")
        resp = server.handle_message(st, {"jsonrpc": "2.0", "id": 1,
                                          "method": "tools/call",
                                          "params": {"name": "multidim_contexts",
                                                     "arguments": {}}})
        self.assertNotIn("error", resp)
        self.assertIn("_reset_reason", st)
        self.assertNotEqual(st["_reset_reason"], "OLD_MARKER")

    def test_learn_message_reports_axis_counters(self):
        # like traps, the message distinguishes an addition
        # from an in-place update.
        text, _ = server.call_tool(store.load(), "multidim_learn",
                                   {"context": "c", "axes": [{"name": "A", "question": "q1"}]})
        self.assertIn("created with 1 axes.", text)
        text, _ = server.call_tool(store.load(), "multidim_learn",
                                   {"context": "c", "axes": [{"name": "A", "question": "q2"},
                                                             {"name": "B", "question": "qb"}]})
        self.assertIn("Axes: 1 added, 1 updated.", text)

    def test_case_variant_generic_is_never_keyword_matched(self):
        # a hand-edited 'Generic' must be treated as the
        # fallback family by DETECTION too, not keyword-matched while learn
        # treats it as generic (one predicate, one semantics).
        # this used to APPEND
        # a second 'Generic' beside the seeded 'generic'. Two contexts sharing
        # a normalised name is corruption, so load() reset the store and the
        # assertions passed on the seed -- the regression was never exercised
        # (the test still passed with the generic predicate stubbed to False).
        # RENAME the seeded one instead: the store stays valid.
        def rename_seeded_generic(st):
            for c in st["contexts"]:
                if c["name"] == "generic":
                    c["name"] = "Generic"
                    c["keywords"] = ["zorglubxyz"]

        self._write_store(rename_seeded_generic)
        loaded = store.load()
        self.assertNotIn("_reset_reason", loaded)  # the store must stay VALID
        self.assertIn("Generic", [c["name"] for c in loaded["contexts"]])
        # the keyword is on the fallback context, so it must NOT be matched
        c, score = server.detect_context(loaded, "subject zorglubxyz")
        self.assertEqual(c.get("name"), "Generic")
        self.assertEqual(score, 0)

    def test_case_variant_generic_learn_guard_applies(self):
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "GENERIC",
                                           "keywords": ["mortx"]})
        self.assertTrue(is_error)
        self.assertIn("generic", text.lower())


class TestPresentButUnusableTrapId(RobustnessTestBase):
    """a trap_id that was
    present but not usable text was silently replaced by a derived one. The
    caller went on referencing the id it sent while the store held another, so
    its trap answer could never match. Deriving is for a MISSING key only."""

    def test_non_text_id_is_refused_not_rewritten(self):
        from multidim_mcp import frames
        base = {"statement": "a lesson", "mandatory_question": "q?",
                "triggers": ["t"]}
        for bad in (123, None, [], {}, "", "   "):
            trap, err = frames.sanitize_trap("ctx", dict(base, trap_id=bad))
            self.assertIsNone(trap, repr(bad))
            self.assertIn("trap_id", err)

    def test_missing_id_still_derives(self):
        from multidim_mcp import frames
        trap, err = frames.sanitize_trap(
            "ctx", {"statement": "a lesson", "mandatory_question": "q?",
                    "triggers": ["t"]})
        self.assertIsNone(err)
        self.assertTrue(trap["trap_id"].startswith("trap_ctx_"))

    def test_learn_reports_the_refusal(self):
        text, is_error = server.call_tool(
            store.load(), "multidim_learn",
            {"context": "c", "traps": [{"trap_id": 123, "statement": "a lesson",
                                        "mandatory_question": "q?",
                                        "triggers": ["t"]}]})
        self.assertIn("trap_id", text)
        # nothing was stored under a fabricated id
        ctx = store.find_context(store.load(), "c")
        self.assertEqual([t for t in (ctx or {}).get("traps", [])
                          if t.get("statement") == "a lesson"], [])


class TestDerivedIdSurvivesUnencodableText(RobustnessTestBase):
    """the derived trap_id
    hashed a UTF-8 encoding of the context name. A lone surrogate is valid in a
    Python str but has no UTF-8 encoding, so learn died on UnicodeEncodeError
    instead of producing an id."""

    def test_lone_surrogate_context_still_derives_an_id(self):
        from multidim_mcp import frames
        trap, err = frames.sanitize_trap(
            "\ud800", {"statement": "a lesson", "mandatory_question": "q?",
                       "triggers": ["t"]})
        self.assertIsNone(err)
        # the id must be usable everywhere it travels: store and wire
        trap["trap_id"].encode("utf-8")
        json.dumps(trap)

    def test_non_latin_names_keep_a_readable_prefix(self):
        from multidim_mcp import frames
        trap, err = frames.sanitize_trap(
            "аудит",  # 'audit' in Cyrillic
            {"statement": "a lesson", "mandatory_question": "q?",
             "triggers": ["t"]})
        self.assertIsNone(err)
        self.assertIn("аудит", trap["trap_id"])

    def test_separator_in_a_name_cannot_forge_another_id(self):
        # the pair is serialised, so a '|' inside the name no longer collides
        # with the old 'name|statement' concatenation
        from multidim_mcp import frames
        a, _ = frames.sanitize_trap("ctx|x", {"statement": "y",
                                              "mandatory_question": "q?",
                                              "triggers": ["t"]})
        b, _ = frames.sanitize_trap("ctx", {"statement": "x|y",
                                            "mandatory_question": "q?",
                                            "triggers": ["t"]})
        self.assertNotEqual(a["trap_id"], b["trap_id"])


class TestMatchingFoldsMathLikeTheDedupKey(RobustnessTestBase):
    """detection folded with
    ``fold`` alone while the dedup key also folded math, so a trigger typed
    '!=' never fired on a subject written with the Unicode sign -- although
    the two are ONE lesson everywhere else. The lesson stayed silent exactly
    when it was needed."""

    def _ctx(self):
        return {"name": "membership", "description": "d", "keywords": ["&&"],
                "axes": [],
                "traps": [{"trap_id": "t1", "statement": "a lesson",
                           "mandatory_question": "q?", "triggers": ["!="],
                           "severity": "medium", "active": True}]}

    def test_unicode_subject_fires_an_ascii_trigger(self):
        from multidim_mcp import frames
        ctx = self._ctx()
        st = store.load()
        st["contexts"].append(ctx)
        ascii_subject = "a && b et x != 3"
        unicode_subject = "a %s b et x %s 3" % (chr(0x2227), chr(0x2260))
        for subject in (ascii_subject, unicode_subject):
            found, score = server.detect_context(st, subject)
            self.assertEqual(found.get("name"), "membership", subject)
            self.assertEqual(score, 1, subject)
            self.assertEqual([t["trap_id"] for t in frames.select_traps(ctx, subject)],
                             ["t1"], subject)

    def test_folding_math_does_not_make_opposites_match(self):
        # the fix must not turn every symbol into the same thing: a trigger on
        # AND must stay silent on a subject that says OR
        from multidim_mcp import frames
        ctx = {"name": "m2", "description": "d", "keywords": [chr(0x2227)],
               "axes": [],
               "traps": [{"trap_id": "t9", "statement": "l",
                          "mandatory_question": "q?", "triggers": [chr(0x2227)],
                          "severity": "medium", "active": True}]}
        st = store.load()
        st["contexts"].append(ctx)
        subject = "x %s allowed" % chr(0x2228)
        self.assertEqual(frames.select_traps(ctx, subject), [])
        found, score = server.detect_context(st, subject)
        self.assertEqual(score, 0)
        self.assertEqual(found.get("name"), "generic")


class TestDuplicateTrapIdIsCorruption(RobustnessTestBase):
    """two traps could share
    one trap_id. Disabling that id reached only the twin the index held, and
    select_traps kept injecting the other -- a lesson the user switched off
    came back."""

    def _ctx_with_ids(self, first_id, second_id):
        def trap(tid, stmt):
            return {"trap_id": tid, "statement": stmt, "mandatory_question": "q?",
                    "triggers": ["zz"], "severity": "medium", "active": True}
        return {"name": "dupctx", "description": "d", "keywords": [], "axes": [],
                "traps": [trap(first_id, "lesson one"), trap(second_id, "lesson two")]}

    def test_duplicate_ids_are_rejected_at_load(self):
        path = self._write_store(
            lambda st: st["contexts"].append(self._ctx_with_ids("dup", "dup")))
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIsNone(store.find_context(reloaded, "dupctx"))
        self.assertIn("_reset_reason", reloaded)

    def test_one_lesson_under_two_ids_is_corruption(self):
        # unique ids were
        # checked, the LESSON was not. Two ids carrying one idea passed the
        # load, and select_traps injected the same mandatory question twice
        # into every frame. upsert_traps cannot produce that, so it is a hand
        # edit -- treated like a duplicate id.
        def trap(tid, stmt):
            return {"trap_id": tid, "statement": stmt, "mandatory_question": "q?",
                    "triggers": ["zz"], "severity": "medium", "active": True}
        path = self._write_store(lambda st: st["contexts"].append(
            {"name": "dupctx", "description": "d", "keywords": [], "axes": [],
             # same idea, different spelling and different ids
             "traps": [trap("t1", "the same lesson"),
                       trap("t2", "the  SAME   lesson !")]}))
        reloaded = store.load()
        self.assertTrue(path.with_name(path.name + ".bak").exists())
        self.assertIsNone(store.find_context(reloaded, "dupctx"))
        self.assertIn("_reset_reason", reloaded)

    def test_distinct_lessons_load_normally(self):
        def trap(tid, stmt):
            return {"trap_id": tid, "statement": stmt, "mandatory_question": "q?",
                    "triggers": ["zz"], "severity": "medium", "active": True}
        self._write_store(lambda st: st["contexts"].append(
            {"name": "okctx", "description": "d", "keywords": [], "axes": [],
             "traps": [trap("t1", "first lesson"), trap("t2", "second lesson")]}))
        ctx = store.find_context(store.load(), "okctx")
        self.assertIsNotNone(ctx)
        self.assertEqual([t["trap_id"] for t in ctx["traps"]], ["t1", "t2"])

    def test_distinct_ids_load_normally(self):
        self._write_store(
            lambda st: st["contexts"].append(self._ctx_with_ids("t1", "t2")))
        ctx = store.find_context(store.load(), "dupctx")
        self.assertIsNotNone(ctx)
        self.assertEqual([t["trap_id"] for t in ctx["traps"]], ["t1", "t2"])


class TestBackupIsCompleteBeforeTheReset(RobustnessTestBase):
    """os.write may write
    fewer bytes than asked. The result was ignored, so a TRUNCATED backup was
    taken for a good one and the unreadable original was then overwritten --
    losing the only complete copy, which is exactly what this branch exists to
    prevent."""

    def test_a_short_write_refuses_instead_of_wiping_the_original(self):
        path = store.paths.store_path()
        store.load()
        corrupt = "{ corrupt but precious"
        path.write_text(corrupt, encoding="utf-8")
        real_write = os.write

        def half_write(fd, data):
            return real_write(fd, data[:len(data) // 2]) and 0

        with mock.patch.object(os, "write", half_write):
            with self.assertRaises(RuntimeError) as caught:
                store.load()
        self.assertIn("refusing to overwrite", str(caught.exception))
        # the unreadable original is still there, untouched
        self.assertEqual(path.read_text(encoding="utf-8"), corrupt)

    def test_a_second_corruption_is_still_recoverable(self):
        # the backup had one
        # fixed name, so a SECOND corruption could not create it (O_EXCL) and
        # load() raised for good -- the recovery branch made recovery
        # impossible, and the server stayed dead until someone deleted the
        # file by hand.
        path = store.paths.store_path()
        store.load()
        names = []
        for i in range(1, 4):
            path.write_text("{ corrupt %d" % i, encoding="utf-8")
            reloaded = store.load()
            self.assertIn("_reset_reason", reloaded)
            names.append(reloaded["_backup"])
        # three distinct backups, no evidence overwritten
        self.assertEqual(len(set(names)), 3)
        for i, name in enumerate(names, start=1):
            self.assertEqual(store.paths.Path(name).read_text(encoding="utf-8"),
                             "{ corrupt %d" % i)

    def test_backups_are_refused_once_they_pile_up(self):
        # rather than overwrite the oldest evidence, refuse and keep the
        # unreadable original untouched
        path = store.paths.store_path()
        store.load()
        corrupt = "{ corrupt and precious"
        path.write_text(corrupt, encoding="utf-8")
        with mock.patch.object(store, "_open_new_backup",
                               side_effect=OSError("too many backups")):
            with self.assertRaises(RuntimeError):
                store.load()
        self.assertEqual(path.read_text(encoding="utf-8"), corrupt)

    def test_a_complete_backup_still_allows_the_reset(self):
        path = store.paths.store_path()
        store.load()
        path.write_text("{ corrupt", encoding="utf-8")
        st = store.load()
        self.assertIn("_reset_reason", st)
        bak = path.with_name(path.name + ".bak")
        self.assertEqual(bak.read_text(encoding="utf-8"), "{ corrupt")


class TestStoreFilesAreNotWorldReadable(RobustnessTestBase):
    """the corrupt-store
    backup was created by os.open with no mode, so it landed at 0o777 & ~umask
    -- typically 0o755 -- while the store it copies is 0o600. The evidence dump
    was more exposed than the evidence."""

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_backup_and_lock_are_owner_only(self):
        import stat
        path = store.paths.store_path()
        store.load()  # seed a valid store first
        prev_umask = os.umask(0o022)  # the permissive-but-common default
        try:
            path.write_text("{ corrupt", encoding="utf-8")
            store.load()
        finally:
            os.umask(prev_umask)
        bak = path.with_name(path.name + ".bak")
        self.assertTrue(bak.exists())
        self.assertEqual(stat.S_IMODE(bak.stat().st_mode), 0o600)
        lock = path.with_name(path.name + ".lock")
        self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)
        # the store itself was already owner-only; it must stay that way
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class TestLegacyCollapseIsLossless(RobustnessTestBase):
    """the legacy-duplicate
    collapse kept the first twin and dropped the second, so a twin holding a
    DIFFERENT non-empty value lost it silently -- and the collapse ran even on
    a learn that carried no axes at all."""

    def _legacy_store(self, axes):
        self._write_store(lambda st: st["contexts"].append(
            {"name": "legacy", "description": "d", "keywords": [],
             "traps": [], "axes": axes}))

    def test_conflicting_twin_is_kept_not_destroyed(self):
        self._legacy_store([{"name": "A", "question": "ancienne", "sublenses": []},
                            {"name": "A", "question": "nouvelle", "sublenses": []}])
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy",
                                           "axes": [{"name": "A", "question": "q3"}]})
        self.assertFalse(is_error, text)
        axes = store.find_context(store.load(), "legacy")["axes"]
        # the survivor takes the update; the twin's own wording still exists
        self.assertEqual([a["question"] for a in axes], ["q3", "nouvelle"])

    def test_conflicting_sublenses_survive_too(self):
        self._legacy_store([{"name": "A", "question": "", "sublenses": ["s1"]},
                            {"name": "A", "question": "", "sublenses": ["s2"]}])
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy",
                                           "axes": [{"name": "B"}]})
        self.assertFalse(is_error, text)
        axes = store.find_context(store.load(), "legacy")["axes"]
        self.assertEqual([a.get("sublenses") for a in axes if a["name"] == "A"],
                         [["s1"], ["s2"]])

    def test_learn_without_axes_leaves_the_grid_untouched(self):
        before = [{"name": "A", "question": "ancienne", "sublenses": []},
                  {"name": "A", "question": "nouvelle", "sublenses": []}]
        self._legacy_store([dict(a) for a in before])
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy",
                                           "keywords": ["mortx"]})
        self.assertFalse(is_error, text)
        self.assertEqual(store.find_context(store.load(), "legacy")["axes"], before)

    def test_complementary_twin_is_still_collapsed(self):
        # the earlier behaviour is unchanged where absorbing loses nothing
        self._legacy_store([{"name": "A", "question": "q1", "sublenses": []},
                            {"name": "A", "question": "", "sublenses": ["s1"]}])
        text, is_error = server.call_tool(store.load(), "multidim_learn",
                                          {"context": "legacy",
                                           "axes": [{"name": "A", "question": "q2"}]})
        self.assertFalse(is_error, text)
        axes = store.find_context(store.load(), "legacy")["axes"]
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0]["question"], "q2")
        self.assertEqual(axes[0]["sublenses"], ["s1"])


class TestDataDirIsAlwaysAbsolute(RobustnessTestBase):
    """a relative data dir
    followed the process's cwd. The server is spawned by a client with a cwd it
    does not choose, so two launches read two different stores."""

    def test_relative_override_is_refused(self):
        os.environ["MULTIDIM_MCP_HOME"] = os.path.join("relative-home", "data")
        with self.assertRaises(RuntimeError) as caught:
            store.paths.data_dir()
        self.assertIn("MULTIDIM_MCP_HOME", str(caught.exception))
        # and the refusal reaches every caller, not just data_dir()
        with self.assertRaises(RuntimeError):
            store.paths.store_path()

    def test_absolute_override_still_wins(self):
        # the isolated temp home set up by the base class
        self.assertEqual(store.paths.data_dir(),
                         store.paths.Path(self._tmp.name))
        self.assertTrue(store.paths.data_dir().is_absolute())

    def test_relative_system_base_is_ignored_not_followed(self):
        # a malformed system base is not a deliberate choice: fall back to the
        # home-anchored default (the rule the XDG spec states) rather than
        # anchoring the store on the cwd.
        os.environ.pop("MULTIDIM_MCP_HOME", None)
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "relative-appdata",
                                          "APPDATA": "relative-appdata",
                                          "XDG_DATA_HOME": "relative-xdg"}):
            resolved = store.paths.data_dir()
        self.assertTrue(resolved.is_absolute(), resolved)
        self.assertNotIn("relative-appdata", str(resolved))
        self.assertNotIn("relative-xdg", str(resolved))


if __name__ == "__main__":
    unittest.main()
