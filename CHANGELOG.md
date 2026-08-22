# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-22

First public release: a standalone Multidim MCP server extracted from a larger
private codebase, with no dependency on it. Published to PyPI as `multidim-mcp`
and listed in the official MCP Registry as `io.github.Cherridsaid/multidim-mcp`.

### Added

- MCP stdio server exposing `multidim_analyze`, `multidim_contexts`,
  `multidim_validate` and `multidim_learn`.
- Deterministic v2 frame contract carrying its own integrity hash, rebuilt
  server-side at validation so a tampered or stale frame is refused.
- Stateless validator with no access to the store.
- Dedicated, portable data directory that never resolves into `~/.multidim`.
- Learned traps: a lesson becomes a mandatory question in later frames.
- Packaging with a real install check (`smoke_install.py`), and CI on Linux and
  Windows across Python 3.9, 3.11 and 3.13.
- `demo.py` and `DEMO.md`: a reproducible end-to-end run of the
  analyse -> fill -> validate -> fix -> accept cycle against a throwaway store.
- `server.json` describing the server for the official MCP Registry, plus the
  matching `mcp-name` ownership marker in the README.
- `publish.yml` workflow: on a `v*` tag, runs the test suite and the packaging
  smoke test, publishes the package to PyPI (trusted publishing, no stored
  token) and then registers the release in the MCP Registry (GitHub OIDC).

### Fixed

The extraction was reviewed adversarially before release. The dominant defect
class was **silent data loss** in the trap deduplication key: a lesson could be
overwritten by its own opposite, because the tokenizer discarded whatever
inverted the meaning. Now preserved — arithmetic and logical operators, unary
and repeated signs, negation before a group or a relation, and every Unicode
symbol. Negated relations are detected by decomposition rather than listed,
since Unicode normalisation strips the very stroke that negates them.

Also fixed before release:

- the file lock and the corrupt-store backup could write outside the
  personal-store guard, at world-readable permissions, truncated, or not at all
  after a second corruption;
- a relative data directory made two launches of the server read two different
  stores;
- the pre-mortem comparison was unbounded and quadratic, so a modest payload
  stalled the server for tens of seconds;
- a `trap_id` that was present but unusable was silently replaced by a derived
  one, so the caller's own references could never match;
- duplicate trap identifiers, and one lesson stored under two identifiers,
  passed the load and injected the same question twice.

[Unreleased]: https://github.com/Cherridsaid/multidim-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Cherridsaid/multidim-mcp/releases/tag/v0.1.0
