"""Runnable demo: the full analyse -> fill -> validate -> fix -> accept cycle.

Run it with `python demo.py`. It uses a throwaway store in a temp directory, so
it never touches your real one, and it prints the transcript reproduced in
DEMO.md. Standard library only, like the rest of the project.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# run from a clean checkout, no install needed (same convention as run_tests.py)
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

SUBJECT ="Should we migrate the billing service from MySQL to PostgreSQL?"

# A first pass an agent produces when nothing forces it to do better: one
# alternative that restates the hypothesis, no falsification test, a
# second-order effect that repeats the first, a filler cross-talk, a pre-mortem
# that copies the risk, and a synthesis grounded in nothing.
SLOPPY = {
    "facts": [
        {"fact_id": "F1", "statement": "The billing service runs on MySQL 8.0 with about 400 GB of data."},
    ],
    "hypotheses": [
        {"hypothesis_id": "H1", "primary": True,
         "statement": "PostgreSQL will improve query performance for billing reports."},
    ],
    "alternatives": [
        {"alternative_id": "A1",
         "statement": "PostgreSQL will improve query performance for billing reports."},
    ],
    "hidden_dependencies": [
        {"dependency_id": "D1", "dependency": "The ORM layer and its MySQL-specific dialect settings."},
    ],
    "second_order_risks": [
        {"risk_id": "R1", "risk": "Migration downtime",
         "first_order_effect": "Billing is unavailable during the cutover.",
         "second_order_effect": "Billing is unavailable during the cutover."},
    ],
    "contradictions": [
        {"contradiction_id": "C1", "contradiction": "We want zero downtime but also a single hard cutover."},
    ],
    "open_questions": [
        {"question_id": "Q1", "question": "Who owns the rollback decision during the cutover window?"},
    ],
    "decision_criteria": [
        {"criterion_id": "K1", "criterion": "P95 latency of the monthly invoice report."},
    ],
    "cross_talk": {
        "tensions": "It depends on the context. A balanced approach is needed.",
        "blind_spots": "This deserves further thought. To be monitored closely.",
    },
    "premortem": {"story": "Migration downtime. Billing is unavailable during the cutover."},
    "synthesis": {"statement": "Overall, migrating seems like a reasonable idea.", "references": []},
}

# The rejected sections redone -- and only those.
FIXED = json.loads(json.dumps(SLOPPY))
FIXED["hypotheses"][0]["falsification_test"] = (
    "Replay the 20 slowest production report queries on a PostgreSQL replica loaded with a "
    "full dump; the hypothesis is false if median P95 does not drop by at least 30%.")
FIXED["alternatives"] = [
    {"alternative_id": "A1",
     "statement": "Keep MySQL and add a read replica dedicated to reporting queries."},
    {"alternative_id": "A2",
     "statement": "Keep MySQL for transactions and stream billing events into a column store "
                  "used only for reports."},
]
FIXED["second_order_risks"][0]["second_order_effect"] = (
    "Invoices issued late push customer payments past the quarter close, distorting "
    "recognised revenue.")
FIXED["cross_talk"] = {
    "tensions": "D1 collides with K1: the ORM dialect settings that make the cutover cheap are "
                "the ones that block the query rewrites K1 measures, so a fast migration and a "
                "fast report are not purchasable together.",
    "blind_spots": "Nothing in F1 or K1 covers write throughput at month end, when billing "
                   "writes peak; the whole analysis is built on read-side evidence.",
}
FIXED["premortem"] = {
    "story": "Six months on, the cutover itself went fine and nobody rolled back. The failure "
             "came from D1: a dialect-specific setting silently rounded tax amounts "
             "differently, so invoices were correct in aggregate but wrong per line. Finance "
             "found it during an audit, not in monitoring, because K1 only watched latency.",
}
FIXED["synthesis"] = {
    "statement": "Migrate only after H1 survives its falsification test on a replica, because "
                 "A1 delivers most of the reporting gain for a fraction of the risk carried by D1.",
    "references": ["H1", "A1", "D1", "K1", "R1"],
}


def _print_verdict(title, report):
    print("\n" + title)
    print("  overall verdict: " + report["verdict"])
    for section in report["section_results"]:
        if section["verdict"] == "ACCEPT":
            continue
        print("  {0:<7} {1:<19} {2}".format(
            section["verdict"], section["section"], ", ".join(section["error_codes"])))
        print("          " + section["details"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="multidim-demo-") as home:
        os.environ["MULTIDIM_MCP_HOME"] = home
        from multidim_mcp import server, store as store_mod

        store = store_mod.load()

        # 1. the frame
        raw, is_error = server.call_tool(
            store, "multidim_analyze", {"subject": SUBJECT, "format": "v2"})
        if is_error:
            print(raw, file=sys.stderr)
            return 1
        frame = json.loads(raw)
        print("1. FRAME")
        print("  context      : {0} (score {1})".format(
            frame["context"]["name"], frame["context"]["score"]))
        print("  frame_hash   : " + frame["frame_hash"])
        print("  sections     : " + ", ".join(frame["required_sections"]))
        print("  rounds       : {0}".format(frame["max_validation_rounds"]))

        # 2. sloppy pass
        raw, _ = server.call_tool(
            store, "multidim_validate", {"frame": frame, "analysis": SLOPPY})
        _print_verdict("2. FIRST PASS", json.loads(raw))

        # 3. rejected sections redone
        raw, _ = server.call_tool(
            store, "multidim_validate", {"frame": frame, "analysis": FIXED})
        report = json.loads(raw)
        _print_verdict("3. AFTER FIXING ONLY THE REJECTED SECTIONS", report)
        if report["verdict"] == "ACCEPT":
            print("  every section accepted")

        # 4. an agent quietly deleting the hardest rule from its own frame
        tampered = json.loads(json.dumps(frame))
        tampered["validation_rules"] = [
            rule for rule in tampered["validation_rules"]
            if rule["rule"] != "FALSIFICATION_REQUIRED"]
        raw, is_error = server.call_tool(
            store, "multidim_validate", {"frame": tampered, "analysis": FIXED})
        print("\n4. TAMPERED FRAME")
        print("  is_error: {0}".format(is_error))
        print("  " + raw)

        assert report["verdict"] == "ACCEPT", "the fixed analysis must be accepted"
        assert is_error, "a tampered frame must be refused"
        print("\nDemo OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
