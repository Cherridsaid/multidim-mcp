"""Standalone Multidim MCP server (stdio JSON-RPC).

Multidim routes a subject to a set of analysis lenses (a *context*) and returns
a hierarchical grid (axes -> sub-lenses) for the caller to fill in. The thinking
stays with the caller; Multidim provides structure, not cognition.

This server is **autonomous**: it has its own entry point
(``python -m multidim_mcp``), its own storage
(:mod:`multidim_mcp.store`) and its own MCP contract (below). Callers
never import these internals -- they speak to this server
over stdio by any MCP client.

MCP contract (JSON-RPC 2.0 over stdio, one message per line):

* ``initialize``      -> ``{protocolVersion, capabilities:{tools:{}}, serverInfo}``
* ``ping``            -> ``{}``
* ``tools/list``      -> ``{tools: [...]}`` (4 tools)
* ``tools/call``      -> ``{content:[{type:"text", text}], isError?}``
* notifications (no ``id``) are accepted and never answered.

Tools: ``multidim_analyze`` (subject[, context][, depth][, format]),
``multidim_contexts`` (none), ``multidim_validate`` (frame, analysis),
``multidim_learn`` (context[, description][, keywords][, axes][, traps]).

stdio rule: ONLY JSON-RPC goes to stdout; logs go to stderr.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Dict, List, Optional, Tuple

from . import frames as frames_mod
from . import store as store_mod
from . import validate as validate_mod
# folding lives in frames.py (single source); re-exported here for callers/tests
from .frames import ANALYSIS_SCHEMA_VERSION, DEPTH_PARAMS, fold, tokens_of

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "multidim"


def _package_version() -> str:
    """The version announced by ``initialize``, read from the INSTALLED
    package metadata -- a hardcoded literal drifted from pyproject and made
    the server claim a version it was not. Falls back to the packaging
    default only when running from a non-installed checkout."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:      # Python 3.7 fallback, unreachable on 3.9+
        return "0.0.0"
    try:
        return version("multidim-mcp")
    except PackageNotFoundError:
        return "0.0.0+unpackaged"


SERVER_VERSION = _package_version()


def detect_context(store: Dict, subject: str) -> Tuple[Dict, int]:
    """Route ``subject`` to the best-scoring context (word-boundary keyword hits).

    Matching follows the single shared rule of :func:`frames.keyword_matches`:
    a purely alphanumeric keyword matches a whole token (``postal`` never
    matches inside ``post``), while any other keyword -- multi-word, hyphenated
    or punctuated like ``c++`` -- matches as a bounded substring. ``generic``
    is never keyword-matched; it is the score-0 fallback.
    """
    folded = frames_mod.fold_for_matching(subject)
    toks = tokens_of(folded)
    best: Optional[Dict] = None
    best_score = 0
    for c in store.get("contexts", []):
        if is_generic(c.get("name")):
            continue
        score = 0
        seen_kf = set()
        for k in c.get("keywords", []):
            # strip AFTER folding: a whitespace-only keyword (hand-edited
            # store) would otherwise take the multi-word substring branch and
            # match EVERY multi-word subject, hijacking detection with score 1
            kf = frames_mod.fold_for_matching(k).strip()
            if not kf:
                continue
            # count each FOLDED keyword once: duplicates (or accent variants
            # of one word) in a hand-edited store must not inflate the score
            # and beat a legitimate context
            if kf in seen_kf:
                continue
            seen_kf.add(kf)
            # one shared rule with trap triggers (frames.keyword_matches):
            # punctuated keys like 'c++' match as bounded substrings
            if frames_mod.keyword_matches(kf, folded, toks):
                score += 1
        if score > best_score:
            best_score = score
            best = c
    if best is not None and best_score > 0:
        return best, best_score
    generic = store_mod.find_context(store, "generic")
    if generic is None:
        raise RuntimeError("the 'generic' context must exist")
    return generic, 0


def is_generic(name) -> bool:
    """Single predicate for the 'generic' fallback family.

    Detection (skip), the learn keyword guard and any future caller must agree
    on what counts as 'generic': any case or surrounding-space variant. Three
    call sites with three different comparisons (exact, lower, lower+strip)
    would let a hand-edited 'Generic' context be keyword-matched by detection
    while learn treats it as the fallback."""
    return isinstance(name, str) and name.strip().lower() == "generic"


def _reject_json_constant(name):
    """``parse_constant`` hook: NaN / Infinity / -Infinity are not JSON.

    Python's decoder accepts them by default; a request carrying one (say as
    its ``id``) would then be echoed back into a response that no strict JSON
    parser can read. Raising here turns it into a clean -32700 parse error.
    """
    raise ValueError("invalid JSON constant: %s" % name)


def _reject_non_finite_float(text):
    """``parse_float`` hook: a numeric LITERAL that overflows to infinity.

    ``1e999`` is well-formed JSON syntax, so ``parse_constant`` never sees it,
    yet Python decodes it to ``inf`` -- which then hits ``allow_nan=False`` on
    the way out and killed the serve loop with an uncaught ValueError. Refuse
    it at the door instead: -32700, and the next request is still served.
    """
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("non-finite JSON number: %s" % text)
    return value


class _NothingLearned(Exception):
    """Raised inside the learn mutator when the call would persist NOTHING
    (e.g. 'generic' keywords dropped and every trap refused): aborts the
    mutate() before its save, so the tool answers an error on an unchanged
    store instead of a false success."""


def _absorbable(survivor: Dict, twin: Dict) -> bool:
    """True when ``twin`` can be folded into ``survivor`` WITHOUT losing anything.

    That holds only when, field by field, the twin carries nothing, or the
    survivor carries nothing, or both carry the same thing. Two different
    non-empty values are a genuine conflict: only the user can say which one
    they meant, so the twin stays in the grid instead of being dropped.
    """
    for field in ("question", "sublenses"):
        mine, theirs = survivor.get(field), twin.get(field)
        if mine and theirs and mine != theirs:
            return False
    return True


def merge_axes(existing_axes: List[Dict], new_axes: List[Dict]) -> Tuple[int, int]:
    """Merge axes by NAME into ``existing_axes`` (in place).

    A re-sent axis (same name) UPDATES the existing one instead of appending a
    sibling duplicate, so a repeated identical learn is idempotent for axes
    exactly as ``upsert_traps`` already makes it for traps. Only the fields the
    caller EXPLICITLY provided are refreshed: re-sending ``{name, question}``
    must never erase the stored ``sublenses`` (the parser keeps omitted fields
    omitted for this reason). Defaults are filled at CREATION only, so a stored
    axis always carries the full shape.

    PRE-EXISTING same-name duplicates (legacy stores written before the dedup
    existed) are collapsed at this write door only (never at load): the FIRST
    occurrence keeps its position and survives; a later twin only fills the
    fields the survivor lacks, then disappears. Without this, a re-learn would
    update the duplicate ``by_name`` happens to index and leave its twin in the
    grid forever.

    That collapse is LOSSLESS or it does not happen (round 42): a twin holding
    a non-empty value where the survivor holds a DIFFERENT non-empty one is
    kept where it is. Dropping it would silently destroy something the user
    wrote, and a visible duplicate they can arbitrate is the lesser evil.
    Learning nothing about axes touches no axis at all -- the grid is only
    rewritten by a call that actually carries axes. Returns ``(added, updated)``.
    """
    if not new_axes:
        return 0, 0
    by_name: Dict = {}
    i = 0
    while i < len(existing_axes):
        a = existing_axes[i]
        if not isinstance(a, dict):
            i += 1
            continue
        n = a.get("name")
        # only NON-EMPTY names are identities: anonymous axes (hand-edited
        # store, name "" or missing) are preserved untouched -- indexing them
        # under one shared key would silently destroy all but the first
        if not (isinstance(n, str) and n.strip()):
            i += 1
            continue
        # index STRIPPED (round 8): a historical ' A ' and a new 'A' are one
        # identity; the survivor's stored name is normalised too
        n = n.strip()
        a["name"] = n
        first = by_name.get(n)
        if first is None:
            by_name[n] = a
            i += 1
        elif _absorbable(first, a):
            # prudent merge: recover content the survivor lacks, lose nothing
            if not first.get("question") and a.get("question"):
                first["question"] = a["question"]
            if not first.get("sublenses") and a.get("sublenses"):
                first["sublenses"] = list(a["sublenses"])
            del existing_axes[i]
        else:
            # conflicting twin: keep it rather than destroy what it holds
            i += 1
    added = updated = 0
    for ax in new_axes:
        current = by_name.get(ax["name"])
        if current is None:
            full = {"name": ax["name"],
                    "question": ax.get("question", ""),
                    "sublenses": list(ax.get("sublenses", []))}
            existing_axes.append(full)
            by_name[full["name"]] = full
            added += 1
        else:
            provided = {k: v for k, v in ax.items() if k != "name"}
            if any(current.get(k) != v for k, v in provided.items()):
                current.update(provided)
                updated += 1
    return added, updated


def build_grid(context: Dict, subject: str, score: int, depth: str) -> str:
    out: List[str] = []
    out.append("CONTEXT: {}  (detection score {})\n".format(context.get("name", ""), score))
    out.append("CONTEXT ROLE: {}\n".format(context.get("description", "")))
    out.append("SUBJECT: {}\n\n".format(subject))
    out.append("MULTIDIMENSIONAL GRID. Work EACH axis separately, then synthesize.\n\n")

    for i, a in enumerate(context.get("axes", [])):
        out.append("[{}] {} : {}\n".format(i + 1, a.get("name", ""), a.get("question", "")))
        if depth != "core":
            for sub in a.get("sublenses", []):
                out.append("      - {}\n".format(sub))

    if depth != "core":
        out.append(
            "\nSUB-LENSES are starting points. Adapt them to the subject and add your own.\n"
            "Any adapted sub-lens is disposable by default.\n"
        )

    out.append("\nCROSS-TALK (mandatory, do not skip):\n")
    if depth == "core":
        out.append("- Briefly relate the axes and name the main blind spot.\n")
    else:
        out.append(
            "- Relate each axis to the others: tensions, reinforcements, dependencies.\n"
            "- Explicitly expose the blind spots that remain after the axis-by-axis pass.\n"
        )

    if depth == "full":
        out.append(
            "\nTARGETED RECURSION (a single pass):\n"
            "- Take the most loaded tension or blind spot from the synthesis.\n"
            "- Re-run it alone in a dedicated sub-grid. One pass, not a loop.\n"
        )

    out.append(
        "\nAT THE END:\n"
        "- Finish with a SYNTHESIS that connects the axes, not a list.\n"
        "- Persist the synthesis wherever your workflow keeps analysis output.\n"
        "- Call multidim_learn ONLY if a sub-lens proved reusable beyond this subject.\n"
    )
    return "".join(out)


def list_contexts_text(store: Dict) -> str:
    out = ["store version {} | {} contexts available:\n\n".format(
        store.get("version", 0), len(store.get("contexts", [])))]
    for c in store.get("contexts", []):
        out.append("# {} ({} axes)\n  {}\n".format(
            c.get("name", ""), len(c.get("axes", [])), c.get("description", "")))
        if c.get("keywords"):
            out.append("  keywords: {}\n".format(", ".join(c["keywords"])))
        for a in c.get("axes", []):
            out.append("    - {} : {}\n".format(a.get("name", ""), a.get("question", "")))
            for s in a.get("sublenses", []):
                out.append("        . {}\n".format(s))
        out.append("\n")
    return "".join(out)


def call_tool(store: Dict, name: str, args: Dict) -> Tuple[str, bool]:
    """Return ``(text, is_error)`` for a tools/call."""
    if name == "multidim_analyze":
        subject = args.get("subject")
        if not isinstance(subject, str) or not subject:
            return ("parameter 'subject' is required (non-empty string).", True)
        # default ONLY when the key is absent or null: `or "deep"` also
        # swallowed False, 0 and "" -- a caller sending depth=false silently
        # got a deep grid instead of an error
        depth = args.get("depth")
        if depth is None:
            depth = "deep"
        if not isinstance(depth, str) or depth not in ("core", "deep", "full"):
            return ("parameter 'depth' must be core, deep or full.", True)
        fmt = args.get("format")
        if fmt is None:
            fmt = "text"
        if fmt not in ("text", "v2"):
            return ("parameter 'format' invalid: '{}'. Allowed values: 'text' "
                    "(v1 grid, default) or 'v2' (JSON frame).".format(fmt), True)
        forced = args.get("context")
        # only ABSENT/null means "auto-detect": false, 0, [] or {} are schema
        # violations, and treating them as absence silently ran a detection
        # the caller never asked for
        if forced is not None and not (isinstance(forced, str) and forced.strip()):
            return ("parameter 'context' must be a non-empty string (or omitted "
                    "for auto-detection).", True)
        if isinstance(forced, str) and forced:
            c = store_mod.find_context(store, forced)
            if c is None:
                return ("unknown context '{}'. List via multidim_contexts or create it "
                        "via multidim_learn.".format(forced), True)
            if fmt == "v2":
                # -1 is the v2 sentinel "forced by the caller": validate skips
                # the detection replay for such frames
                frame = frames_mod.build_frame(store, c, subject, -1, depth)
                return json.dumps(frame, ensure_ascii=False, indent=2), False
            # legacy text grid keeps its historical forced score of 0
            return build_grid(c, subject, 0, depth), False
        c, score = detect_context(store, subject)
        if fmt == "v2":
            frame = frames_mod.build_frame(store, c, subject, score, depth)
            return json.dumps(frame, ensure_ascii=False, indent=2), False
        return build_grid(c, subject, score, depth), False

    if name == "multidim_contexts":
        return list_contexts_text(store), False

    if name == "multidim_validate":
        # FAIL-CLOSED: any unusable input is an explicit ERROR, never a default
        # ACCEPT. The store is NOT handed to the pure core: multidim_validate
        # structurally cannot modify it.
        frame = args.get("frame")
        analysis = args.get("analysis")
        if not isinstance(frame, dict):
            return ("parameter 'frame' is required: the JSON frame returned by "
                    "multidim_analyze format v2.", True)
        if not isinstance(analysis, dict):
            return ("parameter 'analysis' is required: the filled analysis, a JSON "
                    "object section by section.", True)
        try:
            schema_version = int(frame.get("analysis_schema_version", 0))
        except (TypeError, ValueError):
            schema_version = -1  # non-numeric = invalid, never a crash
        if schema_version != ANALYSIS_SCHEMA_VERSION:
            return ("invalid frame: analysis_schema_version {} expected, got {}."
                    .format(ANALYSIS_SCHEMA_VERSION, frame.get("analysis_schema_version")), True)
        if not isinstance(frame.get("required_sections"), list) or not frame["required_sections"]:
            return ("invalid frame: required_sections absent or empty.", True)
        if not isinstance(frame.get("validation_rules"), list):
            return ("invalid frame: validation_rules absent.", True)
        # INTEGRITY: the frame is REBUILT server-side from the store (read
        # only) and compared to the received frame. A self-certified hash can
        # be publicly recomputed on a stripped frame; rebuilding makes the
        # stripping impossible. The pure core validate_analysis still never
        # receives the store: it receives the rebuilt frame, the source of truth.
        subject = frame.get("subject")
        ctx_info = frame.get("context") if isinstance(frame.get("context"), dict) else {}
        ctx_name = ctx_info.get("name")
        score = ctx_info.get("score")
        depth = frame.get("depth")
        if not isinstance(subject, str) or not subject:
            return ("invalid frame: 'subject' field absent.", True)
        # isinstance BEFORE membership: depth=[] would raise unhashable
        # TypeError on the 'in'
        if not isinstance(depth, str) or depth not in DEPTH_PARAMS:
            return ("invalid frame: 'depth' must be core, deep or full.", True)
        # bool is an int subclass: score=true would ride through every int
        # check as 1 (round 10) -- refuse the type explicitly
        if (not isinstance(ctx_name, str) or not ctx_name
                or not isinstance(score, int) or isinstance(score, bool)):
            return ("invalid frame: 'context' (name, score) absent or malformed "
                    "(score must be an integer, not a boolean).", True)
        ctx = store_mod.find_context(store, ctx_name)
        if ctx is None:
            return ("invalid frame: context '{}' unknown to the current store."
                    .format(ctx_name), True)
        if score != -1:
            # non-forced frame: detection is REPLAYED against the current
            # store. A later learn can re-route the subject to another context
            # without changing the original context's content -- the hash
            # alone does not see that.
            detected, detected_score = detect_context(store, subject)
            if detected.get("name") != ctx_name or detected_score != score:
                return ("stale frame: the subject now routes to context '{}' "
                        "(score {}), the frame was issued for '{}' (score {}). "
                        "Regenerate the frame via multidim_analyze format v2."
                        .format(detected.get("name"), detected_score, ctx_name, score), True)
            # rebuild from the VERIFIED replayed score, never the caller's
            # copy: the server's own detection is the source of truth
            score = detected_score
        # the received frame must be SELF-consistent first: comparing only the
        # received HASH to the rebuilt hash lets a stripped or edited BODY that
        # kept its original hash slip through as if untouched (round 9)
        if frames_mod.frame_hash_of(frame) != frame.get("frame_hash"):
            return ("tampered frame: the frame's content does not match its own "
                    "frame_hash (the frame was edited after issuance). Regenerate "
                    "it via multidim_analyze format v2 and reuse it as-is.", True)
        # frame_id is EXCLUDED from the hash (so the hash stays self-verifiable),
        # which leaves it unchecked by the test above: verify it derives from
        # the hash, otherwise a deleted or edited id passes as untouched
        if frame.get("frame_id") != "frame_" + str(frame.get("frame_hash"))[:24]:
            return ("tampered frame: 'frame_id' does not derive from frame_hash. "
                    "Regenerate the frame via multidim_analyze format v2 and "
                    "reuse it as-is.", True)
        rebuilt = frames_mod.build_frame(store, ctx, subject, score, depth)
        if frame.get("frame_hash") != rebuilt["frame_hash"]:
            return ("tampered or stale frame: frame_hash does not match the frame "
                    "rebuilt from the current store. Regenerate the frame via "
                    "multidim_analyze format v2 and reuse it as-is.", True)
        verdict = validate_mod.validate_analysis(rebuilt, analysis)
        return json.dumps(verdict, ensure_ascii=False, indent=2), False

    if name == "multidim_learn":
        cname = args.get("context")
        if not isinstance(cname, str) or not cname.strip():
            return ("parameter 'context' is required.", True)
        # normalise BEFORE lookup and creation: otherwise 'code_review ' with a
        # trailing space misses the existing context and creates a duplicate
        cname = cname.strip()
        # refuse a sentence-name ONLY at creation time: an already existing
        # context (even historical) stays enrichable without blocking.
        if store_mod.find_context(store, cname) is None:
            name_err = store_mod.validate_context_name(cname)
            if name_err is not None:
                return ("context name refused: {}.".format(name_err), True)
        # Strict input validation BEFORE any mutation: whatever we persist must
        # itself pass store._valid_context, so a learn call can never write data
        # that would make the next load() reset the whole store.
        desc = args.get("description", "")
        if not isinstance(desc, str):
            return ("parameter 'description' must be a string.", True)
        kw = args.get("keywords", [])
        if not isinstance(kw, list) or not all(isinstance(k, str) for k in kw):
            return ("parameter 'keywords' must be an array of strings.", True)
        # strip BEFORE the emptiness check: a whitespace-only keyword is not
        # an identity, and unstripped it would either hijack detection (the
        # substring branch matches every multi-word subject) or die silently
        stripped_kw = [k.strip().lower() for k in kw]
        if any(not k for k in stripped_kw):
            return ("keywords must be non-empty strings (whitespace-only "
                    "keyword refused).", True)
        # dedupe by FOLDED key (order preserved): ['cafe','cafe'] -- or an
        # accent variant of an existing keyword -- must never be stored twice,
        # since detection counts hits and a duplicate would inflate the score
        seen_fold = set()
        new_keywords = []
        for k in stripped_kw:
            fk = frames_mod.fold(k)
            if fk not in seen_fold:
                seen_fold.add(fk)
                new_keywords.append(k)
        axes_in = args.get("axes", [])
        if not isinstance(axes_in, list):
            return ("parameter 'axes' must be an array of objects.", True)
        new_axes = []
        for v in axes_in:
            if not isinstance(v, dict):
                return ("each axis must be an object.", True)
            n = v.get("name")
            # strip BEFORE the emptiness check: a whitespace-only name ('   ')
            # is not an identity -- unstripped it would dodge the dedup (which
            # ignores blank names) and pile up duplicates at every learn
            if not isinstance(n, str) or not n.strip():
                return ("each axis needs a non-empty string 'name'.", True)
            n = n.strip()
            # OMITTED optional fields stay omitted (merge_axes then leaves the
            # stored value untouched): normalising an absent 'sublenses' to []
            # here would make a partial re-send ERASE the stored sub-lenses.
            axis = {"name": n}
            if "question" in v:
                q = v.get("question")
                if not isinstance(q, str):
                    return ("axis 'question' must be a string.", True)
                axis["question"] = q
            if "sublenses" in v:
                subs_in = v.get("sublenses")
                if not isinstance(subs_in, list) or not all(isinstance(s, str) for s in subs_in):
                    return ("axis 'sublenses' must be an array of strings.", True)
                axis["sublenses"] = list(subs_in)
            new_axes.append(axis)

        raw_traps = args.get("traps", [])
        if not isinstance(raw_traps, list):
            return ("parameter 'traps' must be an array of objects.", True)

        # 'generic' is the score-0 fallback: detect_context NEVER keyword-
        # matches it, so keywords stored there are silently dead data. A
        # keywords-only learn is refused loudly; a mixed learn (axes/traps/
        # description are all legitimate on generic) succeeds but says what
        # was dropped instead of reporting a false full success.
        generic_keywords_ignored = False
        if is_generic(cname) and new_keywords:
            # only a VALID trap makes the call "mixed" (round 9): an invalid
            # one (e.g. {}) is refused later by upsert_traps, so counting it
            # here would report success on a call that learns nothing at all
            has_valid_trap = any(
                frames_mod.sanitize_trap(cname, t)[0] is not None
                for t in raw_traps)
            if not new_axes and not has_valid_trap and not desc:
                return ("keywords are never matched for 'generic' (it is the "
                        "score-0 fallback, excluded from keyword detection), so "
                        "there is nothing to learn: target a specific context, "
                        "or add axes/traps instead.", True)
            generic_keywords_ignored = True
            new_keywords = []

        # SERIALIZED read-modify-write: store_mod.mutate takes a cross-process
        # lock, RELOADS the store from disk (so a concurrent writer's changes
        # are merged, not clobbered), applies this mutation, persists, releases.
        # The live in-memory store is refreshed ONLY on success; any failure
        # (lock timeout or save OSError) leaves memory and disk as they were.
        def _apply(fresh):
            existing = store_mod.find_context(fresh, cname)
            if existing is not None:
                if desc:
                    existing["description"] = desc
                # defence in depth: migrate_additive guarantees these keys on
                # anything load() returns, but _apply must stay safe even on a
                # store handed in by another caller
                existing.setdefault("keywords", [])
                existing.setdefault("axes", [])
                existing.setdefault("traps", [])
                # membership by FOLDED key, not exact string: an accent
                # variant of a stored keyword is the same detection hit and
                # must not be appended as a score-inflating duplicate
                existing_folds = {frames_mod.fold(str(x)).strip()
                                  for x in existing["keywords"]}
                for k in new_keywords:
                    fk = frames_mod.fold(k)
                    if fk not in existing_folds:
                        existing_folds.add(fk)
                        existing["keywords"].append(k)
                a_add, a_upd = merge_axes(existing["axes"], new_axes)
                t_add, t_upd, t_err = frames_mod.upsert_traps(existing, raw_traps)
                # round 10: the keywords-only decision can only fall AFTER
                # upsert_traps, under the lock -- a trap valid in shape can
                # still be refused (id/statement collision), and if every trap
                # was REFUSED while the keywords were dropped, this call
                # learned nothing and must not report success. Raising aborts
                # the mutate() before its save, leaving the store untouched.
                # round 11: refused means t_err -- an identical re-send is a
                # legitimate idempotent NO-OP (0 added, 0 updated, 0 errors)
                # and stays a success, so only error when every trap errored.
                if (generic_keywords_ignored and not desc and not new_axes
                        and raw_traps and t_add == 0 and t_upd == 0
                        and len(t_err) == len(raw_traps)):
                    raise _NothingLearned(
                        "nothing learned on 'generic': keywords are never "
                        "matched for it, and every trap was refused ({})."
                        .format(" ; ".join(t_err)))
                msg = "context '{}' enriched. {} axes total.".format(
                    existing["name"], len(existing["axes"]))
                if new_axes:
                    # same contract as traps: the caller can tell an addition
                    # from an in-place update without diffing the store
                    msg += " Axes: {} added, {} updated.".format(a_add, a_upd)
                if raw_traps:
                    msg += " Traps: {} added, {} updated, {} total.".format(
                        t_add, t_upd, len(existing["traps"]))
                if t_err:
                    msg += " Traps refused: {}.".format(" ; ".join(t_err))
                return msg
            c = {
                "name": cname,
                "description": desc or "Custom context.",
                "keywords": new_keywords,
                "axes": [],
                "traps": [],
            }
            # merge (not assign) so two same-named axes within one call
            # collapse too, keeping creation and enrichment consistent
            merge_axes(c["axes"], new_axes)
            t_add, _t_upd, t_err = frames_mod.upsert_traps(c, raw_traps)
            fresh["contexts"].append(c)
            # count what was STORED, not what was sent: same-name axes within
            # one call collapse in merge_axes, the message must not lie
            msg = "context '{}' created with {} axes.".format(cname, len(c["axes"]))
            if raw_traps:
                msg += " Traps: {} added.".format(t_add)
            if t_err:
                msg += " Traps refused: {}.".format(" ; ".join(t_err))
            return msg

        try:
            fresh, msg = store_mod.mutate(_apply)
        except _NothingLearned as exc:
            # aborted before mutate()'s save: disk and memory are unchanged
            return (str(exc), True)
        except (OSError, TimeoutError) as exc:
            # lock could not be taken, or disk write failed: memory and disk
            # stay in their prior consistent state
            return ("failed to persist the learned context (store unchanged): {}"
                    .format(exc), True)
        # committed to disk -> refresh the live store in place so the serve
        # loop's reference sees the merged result. The in-memory reset markers
        # are the CALLER's diagnostics (contract: "for the caller only"): they
        # must survive the refresh -- the DISK copy alone stays filtered.
        # old diagnostics first, fresh state second: if the reload itself just
        # produced NEWER markers (a reset during this call), they must win
        markers = {k: store[k] for k in store_mod.PRIVATE_MARKERS if k in store}
        store.clear()
        store.update(markers)
        store.update(fresh)
        if generic_keywords_ignored:
            msg += (" Keywords ignored: 'generic' is never keyword-matched "
                    "(score-0 fallback).")
        return msg, False

    return ("unknown tool: {}".format(name), True)


def tools_schema() -> List[Dict]:
    return [
        {
            "name": "multidim_analyze",
            "description": ("Build a hierarchical multidimensional grid (axes -> sub-lenses) "
                            "adapted to the subject's context. Returns the grid to fill, not "
                            "the analysis."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "The subject to analyse."},
                    "context": {"type": "string", "description": "Optional. Force a context; otherwise auto-detected."},
                    "depth": {"type": "string", "enum": ["core", "deep", "full"],
                              "description": "Optional, default deep. core=axes+short cross-talk; deep=+sub-lenses; full=+one targeted recursion."},
                    "format": {"type": "string", "enum": ["text", "v2"],
                               "description": ("Optional, default text (v1 grid). v2 = deterministic "
                                               "JSON frame of the v2 contract: frame_hash, mandatory "
                                               "questions, learned traps, required_sections, "
                                               "validation_rules, max_validation_rounds, optional "
                                               "host_hints. The calling LLM fills this frame section "
                                               "by section.")},
                },
                "required": ["subject"],
            },
        },
        {
            "name": "multidim_contexts",
            "description": "List every known context with its axes and sub-lenses.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "multidim_validate",
            "description": ("Deterministic, STATELESS check of a filled analysis against its v2 "
                            "frame (multidim_analyze format v2). Returns an ACCEPT / WARNING / "
                            "REJECT verdict per section with actionable error codes. Never "
                            "modifies the store, calls no LLM, never judges the truth of the "
                            "content: structure, internal consistency and checkable requirements "
                            "only. The calling LLM redoes ONLY the rejected sections, within the "
                            "frame's max_validation_rounds (tracked by the caller)."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "frame": {"type": "object",
                              "description": "The exact JSON frame returned by multidim_analyze format v2."},
                    "analysis": {"type": "object",
                                 "description": "The analysis filled by the calling LLM, section by section."},
                },
                "required": ["frame", "analysis"],
            },
        },
        {
            "name": "multidim_learn",
            "description": ("Create or enrich a context. Use only to promote a lens reusable "
                            "beyond the current subject: deliberate promotion, never automatic. "
                            "Persists to the dedicated store (the only write door)."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "description": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "axes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "question": {"type": "string"},
                                "sublenses": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name"],
                        },
                    },
                    "traps": {
                        "type": "array",
                        "description": ("Optional. Learned traps (paid lessons): each becomes a "
                                        "mandatory question injected into future v2 frames when a "
                                        "trigger matches the subject. Deduplicated by trap_id or "
                                        "statement; active:false disables a lesson without "
                                        "deleting it."),
                        "items": {
                            "type": "object",
                            "properties": {
                                "trap_id": {"type": "string", "description": "Optional, derived from the statement otherwise."},
                                "statement": {"type": "string", "description": "The trap statement (the lesson)."},
                                "mandatory_question": {"type": "string", "description": "The question to ask forever."},
                                "triggers": {"type": "array", "items": {"type": "string"},
                                             "description": "Words or phrases of the subject that trigger the injection."},
                                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                                "active": {"type": "boolean", "description": "false = lesson disabled, never injected."},
                            },
                            "required": ["statement", "mandatory_question", "triggers"],
                        },
                    },
                },
                "required": ["context"],
            },
        },
    ]


def log(msg: str, stream=None) -> None:
    print("[multidim] " + msg, file=stream or sys.stderr, flush=True)


def handle_message(store: Dict, msg: Dict) -> Optional[Dict]:
    """Map one JSON-RPC request to a response dict (or None for a notification)."""
    raw_params = msg.get("params")

    # A NOTIFICATION is a message with NO "id" member. An explicit "id": null is
    # a request and still expects a response (with id null), so distinguish
    # absence from a null value rather than conflating the two. Notifications
    # never get a response, even when otherwise malformed.
    if "id" not in msg:
        return None
    msg_id = msg.get("id")

    # JSON-RPC 2.0 (sect. 4): an id is a String, a Number or Null -- never a
    # boolean, array or object. Echoing a structured id back would make the
    # response uncorrelatable for a strict client, so refuse it with a NULL
    # id (the only safe echo for an invalid one).
    if msg_id is not None and (isinstance(msg_id, bool)
                               or not isinstance(msg_id, (str, int, float))):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600,
                          "message": "invalid request: 'id' must be a string, "
                                     "a number or null"}}

    # Validate the JSON-RPC 2.0 envelope of a REQUEST: the version tag must be
    # exactly "2.0" and "method" must be a non-empty string. Anything else is an
    # Invalid Request (-32600), not a silently accepted call.
    if msg.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32600, "message": "invalid request: jsonrpc must be '2.0'"}}
    method = msg.get("method")
    if not isinstance(method, str) or not method:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32600, "message": "invalid request: 'method' must be a non-empty string"}}

    # ``params``, when present, must be an object; a bare string/array is invalid.
    if raw_params is None:
        params = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32602, "message": "invalid params: must be an object"}}

    if method == "initialize":
        # Real negotiation: echo the client's version only if we support it,
        # otherwise answer with OUR supported version. Echoing an unsupported
        # value back would falsely claim agreement on a version we do not speak.
        requested = params.get("protocolVersion")
        proto = requested if requested == PROTOCOL_VERSION else PROTOCOL_VERSION
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": proto,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools_schema()}}
    if method == "tools/call":
        # same rule as the tool parameters: only a MISSING name defaults;
        # a non-string one is not silently turned into the empty string
        tname = params.get("name")
        if tname is None:
            tname = ""
        elif not isinstance(tname, str):
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602,
                              "message": "invalid params: 'name' must be a string"}}
        targs = params.get("arguments")
        if targs is None:
            targs = {}
        elif not isinstance(targs, dict):
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602, "message": "invalid params: 'arguments' must be an object"}}
        # Reload the store from disk before EACH tool call so a second server
        # sharing the same store never operates on a stale snapshot (learn from
        # another process is seen immediately). The lock-free fast path keeps
        # this cheap for an unchanged store; learn re-reads again under its lock.
        try:
            live = store_mod.load()
        except Exception as exc:  # noqa: BLE001 -- reload failure must not crash the loop
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text",
                                            "text": "internal error: cannot load store: %s" % exc}],
                               "isError": True}}
        # keep the caller's reference in sync with disk too (in-place refresh).
        # Same marker contract as the learn path: the in-memory reset markers
        # are the caller's diagnostics and survive every refresh -- only the
        # disk copy is filtered.
        # old diagnostics first, fresh state second: a reset that happened
        # during THIS reload carries newer markers and must win over the old
        markers = {k: store[k] for k in store_mod.PRIVATE_MARKERS if k in store}
        store.clear()
        store.update(markers)
        store.update(live)
        # Defence in depth: any unexpected tool error becomes an isError result,
        # never an unhandled exception that would kill the serve loop.
        try:
            text, is_error = call_tool(store, tname, targs)
        except Exception as exc:  # noqa: BLE001 -- a tool bug must not crash the server
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": "internal error: %s" % exc}],
                               "isError": True}}
        result = {"content": [{"type": "text", "text": text}]}
        if is_error:
            result["isError"] = True
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": "unsupported method: {}".format(method)}}


def serve(stdin=None, stdout=None, log_stream=None) -> int:
    """Run the stdio JSON-RPC loop. Reads lines from stdin, writes to stdout.

    ``log_stream`` defaults to ``sys.stderr`` (the MCP rule: only JSON-RPC on
    stdout, logs on stderr). Tests pass an in-memory stream so no diagnostic
    text leaks to the real stderr.
    """
    if stdin is None:
        # BINARY stdin, decoded STRICTLY line by line below. errors="replace"
        # would keep the loop alive but silently ALTER the request: an invalid
        # byte inside a valid JSON string (say, a method name) becomes U+FFFD,
        # the line parses, and the server answers -32601 -- or executes
        # content the client never sent. A line that is not valid UTF-8 is a
        # PARSE error: -32700, then keep serving.
        stdin = sys.stdin.buffer
    if stdout is None:
        # Reconfigure the EXISTING sys.stdout rather than wrapping its buffer:
        # a new TextIOWrapper takes ownership and leaves sys.stdout detached,
        # so any later print() in the host process raised ValueError. The wire
        # is ASCII-escaped anyway; newline="\n" keeps one line per message on
        # Windows too.
        try:
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
            stdout = sys.stdout
        except (AttributeError, ValueError):
            # exotic stdout (already wrapped, or a stub without reconfigure)
            stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      newline="\n")
    store = store_mod.load()
    log("started. {} contexts, store: {}".format(len(store.get("contexts", [])),
                                                  store_mod.paths.store_path()), log_stream)
    for line in stdin:
        if isinstance(line, (bytes, bytearray)):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError as exc:
                stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700,
                                                   "message": "parse error: invalid utf-8"}},
                                        ensure_ascii=True, allow_nan=False) + "\n")
                stdout.flush()
                log("unicode error: {}".format(exc), log_stream)
                continue
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            # NaN/Infinity are a Python extension, NOT valid JSON: accepting
            # them let a non-serialisable id ride into the response, which a
            # strict client then rejects. Refuse at the door with -32700.
            msg = json.loads(trimmed, parse_constant=_reject_json_constant,
                             parse_float=_reject_non_finite_float)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            # RecursionError: deeply nested arrays blow the C recursion limit
            # inside the decoder. That is still a malformed REQUEST, not a
            # server fault -- answer -32700 and keep serving the next line.
            # JSON-RPC parse error: answer -32700 with id null, keep serving.
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error: invalid json"}},
                                    ensure_ascii=True, allow_nan=False) + "\n")
            stdout.flush()
            log("parse error: {}".format(exc), log_stream)
            continue
        if not isinstance(msg, dict):
            # Valid JSON but not a JSON-RPC object (e.g. a bare array). Answer
            # -32600 and keep serving, never crash the loop.
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32600,
                                               "message": "invalid request: not a JSON object"}},
                                    ensure_ascii=True, allow_nan=False) + "\n")
            stdout.flush()
            continue
        resp = handle_message(store, msg)
        if resp is not None:
            try:
                payload = json.dumps(resp, ensure_ascii=True, allow_nan=False)
            except (ValueError, TypeError) as exc:
                # defence in depth: whatever slipped into the response, a
                # serialisation failure must not kill the loop -- answer an
                # internal error (id preserved when it is itself printable)
                log("unserialisable response: {}".format(exc), log_stream)
                try:
                    safe_id = json.dumps(resp.get("id"), allow_nan=False)
                except (ValueError, TypeError):
                    safe_id = "null"
                payload = ('{"jsonrpc": "2.0", "id": %s, "error": {"code": -32603, '
                           '"message": "internal error: response not serialisable"}}'
                           % safe_id)
            stdout.write(payload + "\n")
            stdout.flush()
    log("stopped", log_stream)
    return 0
