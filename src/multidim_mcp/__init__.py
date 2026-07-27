"""multidim-mcp: a standalone Multidim MCP server.

Multidim routes a subject to a set of analysis lenses (a *context*) and
returns a hierarchical grid (axes -> sub-lenses) for the calling LLM to fill
in. The thinking stays with the caller; Multidim provides structure, not
cognition. The server is autonomous: its own stdio entry point
(``python -m multidim_mcp``), its own storage on a dedicated path, its own
deterministic v2 contract and its own tests.

Public surface:

* :func:`multidim_mcp.server.serve` -- run the stdio MCP server;
* :func:`multidim_mcp.paths.store_path` -- the dedicated store path;
* :mod:`multidim_mcp.store` / :mod:`multidim_mcp.base_contexts`.
"""

from __future__ import annotations

from . import base_contexts, frames, paths, server, store, validate

__all__ = ["server", "store", "paths", "base_contexts", "frames", "validate"]
