<p align="center">
  <img src="https://raw.githubusercontent.com/Cherridsaid/multidim-mcp/main/docs/hero.png" alt="multidim-mcp: one subject, split through a prism into eight analysis lenses" width="100%">
</p>

# multidim-mcp

<!-- mcp-name: io.github.Cherridsaid/multidim-mcp -->

**Structured thinking grids for AI agents — a standalone MCP server, pure standard library.**

Multidim routes a subject to a set of analysis lenses (a *context*) and returns a
hierarchical grid — axes, sub-lenses, mandatory questions — for the calling LLM to
fill in. **The thinking stays with the caller**: the server provides structure,
never cognition. It calls no LLM, makes no network requests, and the same input
always produces the same frame.

- **Zero dependencies** — Python 3.9+, standard library only.
- **Deterministic v2 contract** — every frame carries a self-verifiable `frame_hash`;
  a filled analysis is checked section by section with actionable error codes.
- **Learned traps** — lessons you record once become mandatory questions injected
  into every future frame whose subject matches.
- **Hardened store** — atomic writes, native cross-process locking, additive
  migrations, backed-up resets, and a guard that refuses to ever touch a foreign
  `~/.multidim` store.

## See the difference

An agent analyses *"Should we migrate the billing service from MySQL to
PostgreSQL?"*. Every section is filled, every sentence reads fine. Here is what
`multidim_validate` returns on that first pass:

```
overall verdict: REJECT

REJECT  alternatives        NOT_ENOUGH_ALTERNATIVES, ALTERNATIVE_DUPLICATES_PRIMARY
REJECT  hypotheses          HYPOTHESIS_NOT_FALSIFIABLE
REJECT  second_order_risks  SECOND_ORDER_REPEATS_FIRST
REJECT  cross_talk          GENERIC_DENSITY_HIGH
REJECT  synthesis           SYNTHESIS_WITHOUT_REFERENCES
WARNING premortem           PREMORTEM_SIMILAR_TO_RISKS
```

The only alternative restated the hypothesis, the hypothesis carried no test that
could prove it wrong, the second-order effect repeated the first one word for
word, and the conclusion referenced none of the work above. None of that is
visible when you read the answer; all of it is reported here, by name.

Redo the rejected sections and the same checker returns `ACCEPT`. Edit the frame
to delete the rule you find inconvenient, and it refuses the whole submission —
the frame carries a hash of its own content.

Full transcript, including what the fixed sections look like and what this
deliberately does *not* check: **[DEMO.md](DEMO.md)**. Reproduce it in one
command: `python demo.py`.

## Quickstart

```bash
pip install multidim-mcp        # from PyPI
pip install .                   # or from a source checkout
```

Register the server with any MCP client (stdio transport):

```json
{
  "mcpServers": {
    "multidim": {
      "command": "multidim-mcp"
    }
  }
}
```

Or run it directly: `python -m multidim_mcp`, or without installing: `uvx multidim-mcp`.

The server is listed in the official MCP Registry as
[`io.github.Cherridsaid/multidim-mcp`](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.Cherridsaid/multidim-mcp).

## Tools

| Tool | Role |
|---|---|
| `multidim_analyze` | Build the grid for a subject (`depth`: `core` / `deep` / `full`; `format`: text or deterministic `v2` JSON frame) |
| `multidim_contexts` | List every known context with its axes and sub-lenses |
| `multidim_validate` | Deterministic, stateless check of a filled analysis against its v2 frame — `ACCEPT` / `WARNING` / `REJECT` per section |
| `multidim_learn` | Create or enrich a context (keywords, axes, traps) — the only write door |

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/Cherridsaid/multidim-mcp/main/docs/workflow.png" alt="multidim_analyze produces a deterministic v2 frame; your LLM fills it; multidim_validate stamps ACCEPT / WARNING / REJECT and only rejected sections are redone" width="90%">
</p>

1. `multidim_analyze` detects the best context for your subject (word-boundary
   keyword matching, accent-folded) and returns a **v2 frame**: required sections,
   section schemas, validation rules, mandatory questions — including every
   **learned trap** whose triggers match the subject.
2. Your LLM fills the frame, section by section.
3. `multidim_validate` rebuilds the frame from the store, refuses a tampered or
   stale one (`frame_hash`), then checks the analysis: structural completeness,
   falsification tests on hypotheses, alternatives that genuinely differ from the
   primary, second-order effects distinct from first-order, a pre-mortem that does
   not copy the risk list, a synthesis that references real identifiers, and a
   filler-phrase density cap. Only rejected sections are redone, within the
   frame's `max_validation_rounds`.

The four seed contexts are neutral and deterministic: `generic` (8 general
lenses), `code_review`, `technical_writing`, `decision`.

## Storage

The store lives on a dedicated per-user data path (`MULTIDIM_MCP_HOME` overrides
it) and is created on first run from the neutral seeds. Writes are atomic and
serialized across processes with the OS's native file locking; a corrupt store is
backed up before any reset, never silently discarded. A tripwire refuses every
read or write that would resolve into a foreign personal `~/.multidim` store.

Maintainers publishing forks can extend the neutrality guard with their own
private markers via `MULTIDIM_MCP_EXTRA_FORBIDDEN` (comma-separated), without
hardcoding them into public source.

## Transparency

- **Not an AI system.** multidim-mcp contains no model and performs no inference:
  it is deterministic, rule-based software. Under the EU AI Act (Reg. 2024/1689)
  it is not an AI system in the sense of Art. 3(1), and as free and open-source
  software it falls under the Art. 2(12) exemption. It collects no data and makes
  no network calls.
- **Illustrations** in this README were generated with GPT and keep their C2PA
  provenance metadata intact.

## Development

```bash
python run_tests.py      # full suite, stdlib only
python smoke_install.py  # packaging smoke test (wheel + venv + entry point)
python demo.py           # the analyse -> validate -> fix -> accept cycle of DEMO.md
```

CI runs both on Ubuntu and Windows across Python 3.9 / 3.11 / 3.13.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
