# One full cycle, end to end

Everything below is the verbatim output of `python demo.py` — a script in this
repository that runs against a throwaway store, so you can reproduce it in one
command from a clean checkout:

```bash
python demo.py
```

The subject is deliberately ordinary:

> Should we migrate the billing service from MySQL to PostgreSQL?

## 1. The frame

`multidim_analyze` with `format: "v2"` returns a deterministic frame — not an
analysis. It carries the required sections, the schema of each one, the
validation rules that will be applied, and a `frame_hash` over its own content.

```
context      : generic (score 0)
frame_hash   : 4e889f8bbfae895fe9512aeed37d411360402f924920aae4b34b2ee262062a7e
sections     : facts, hypotheses, alternatives, hidden_dependencies,
               second_order_risks, contradictions, open_questions,
               decision_criteria, cross_talk, premortem, synthesis
rounds       : 2
```

The same subject always produces the same frame, and the same hash.

## 2. First pass — what an unconstrained answer looks like

The agent fills the frame. Nothing is wrong on the surface: every section is
present, every sentence is plausible. `multidim_validate` disagrees.

```
overall verdict: REJECT

REJECT  alternatives        NOT_ENOUGH_ALTERNATIVES, ALTERNATIVE_DUPLICATES_PRIMARY
        1 distinct alternative(s), minimum 2 at this depth ; alternative A1
        duplicates the primary hypothesis (identical idea)
REJECT  cross_talk          GENERIC_DENSITY_HIGH
        4 hollow sentence(s) out of 4: section mostly filler
REJECT  hypotheses          HYPOTHESIS_NOT_FALSIFIABLE
        hypothesis H1 without an observable falsification_test (scope primary)
WARNING premortem           PREMORTEM_SIMILAR_TO_RISKS
        pre-mortem close to risk R1 (similarity 0.89)
REJECT  second_order_risks  SECOND_ORDER_REPEATS_FIRST
        risk R1: the second-order effect repeats the first order (identical idea)
REJECT  synthesis           SYNTHESIS_WITHOUT_REFERENCES
        the synthesis references no identifier from upstream sections
```

Five rejections and one warning, on an analysis that reads fine. Each one names
a specific failure the reader would otherwise have to catch by hand:

| What the agent wrote | What the checker saw |
|---|---|
| Alternative A1: *"PostgreSQL will improve query performance for billing reports."* — the hypothesis, restated | an alternative that is not an alternative |
| Hypothesis H1 asserted, no test attached | a claim that cannot be proven wrong |
| First-order: *"Billing is unavailable during the cutover."* → second-order: the same sentence | a second-order analysis that never left the first order |
| Cross-talk: *"It depends on the context. A balanced approach is needed."* | four filler sentences out of four |
| Synthesis: *"Overall, migrating seems like a reasonable idea."* | a conclusion grounded in none of the work above |

## 3. Only the rejected sections are redone

The contract is explicit: accepted sections are kept, rejected ones are redone,
within `max_validation_rounds`. One round was enough here.

```
overall verdict: ACCEPT
every section accepted
```

What changed, concretely:

**Hypothesis** — a falsification test was attached:

> Replay the 20 slowest production report queries on a PostgreSQL replica loaded
> with a full dump; the hypothesis is false if median P95 does not drop by at
> least 30%.

**Second-order effect** — it now leaves the first order:

> Invoices issued late push customer payments past the quarter close, distorting
> recognised revenue.

**Synthesis** — it now stands on identifiers from the sections above:

> Migrate only after H1 survives its falsification test on a replica, because A1
> delivers most of the reporting gain for a fraction of the risk carried by D1.
> `references: ["H1", "A1", "D1", "K1", "R1"]`

The second version is not more eloquent than the first. It is checkable.

## 4. The frame cannot be quietly relaxed

An agent that finds a rule inconvenient could simply delete it from the frame it
was handed. Here it removes `FALSIFICATION_REQUIRED` and submits the same
analysis:

```
is_error: True
tampered frame: the frame's content does not match its own frame_hash (the frame
was edited after issuance). Regenerate it via multidim_analyze format v2 and
reuse it as-is.
```

This is the difference between a scaffold and a prompt. A prompt asking for
rigour can be ignored silently, and you find out later. A frame that carries its
own hash cannot be edited without the check refusing it.

## What this does not do

Being clear about the boundary matters more than the demo:

- **It does not judge whether the content is true.** `multidim_validate` checks
  structure, internal consistency and checkable requirements. A confident,
  well-formed, factually wrong analysis is accepted. Verification of facts stays
  with you.
- **It does not make the model smarter.** Same model, same knowledge, same
  mistakes available. What changes is that the reasoning is complete, connected
  and auditable — every time, not on a good day.
- **The filler blacklist ships with eight phrases.** It is a floor, not a style
  checker; extend it through your store for the phrasing your own agents lean on.
- **`generic` is the fallback grid.** The subject above matched no context
  keyword, so it fell back to the general-purpose lens — visible in the frame as
  `score 0`, never silently.
