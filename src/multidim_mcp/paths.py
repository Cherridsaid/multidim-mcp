"""Cross-platform data directory for the bundled Multidim MCP server.

``multidim-mcp`` is a **zero-dependency** package, so this deliberately does not
use ``platformdirs``: it resolves a dedicated, portable data directory with the
standard library alone, following the same platform conventions.

Resolution order:

1. ``$MULTIDIM_MCP_HOME`` if set (explicit override, expanded, must be absolute);
2. Windows  -> ``%LOCALAPPDATA%\\multidim-mcp`` (then ``%APPDATA%``);
3. macOS    -> ``~/Library/Application Support/multidim-mcp``;
4. other    -> ``$XDG_DATA_HOME/multidim-mcp`` or ``~/.local/share/multidim-mcp``.

The resolved directory is always ABSOLUTE. A relative override or a relative
system base would anchor the store on the process's current directory, and the
server is spawned by a client with a cwd it does not choose: two launches would
then read two different stores.

The Multidim store lives at ``<data_dir>/multidim/store.json``.

This module NEVER points at ``~/.multidim`` -- that path belongs to a separate,
personal installation and must never be read, written, migrated or overwritten
by the bundled OSS server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

APP_NAME = "multidim-mcp"
_ENV_OVERRIDE = "MULTIDIM_MCP_HOME"

# A hard tripwire: the OSS server must never resolve to the personal store.
_FORBIDDEN_PERSONAL = os.path.join("~", ".multidim")


def _home() -> Path:
    # ``Path.home()`` can raise on an environment with no HOME; fall back to cwd
    # rather than crash, so the server still starts (in a local ``.multidim-mcp``).
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return Path(os.getcwd())


def _absolute_base(var: str) -> Optional[Path]:
    """Return ``$var`` as a base directory, or ``None`` if unusable.

    A base that is unset, blank or RELATIVE is treated as a malformed
    environment and ignored in favour of the home-anchored default -- the rule
    the XDG spec states for ``$XDG_DATA_HOME``, applied to every platform.
    """
    raw = os.environ.get(var)
    if not raw or not raw.strip():
        return None
    base = Path(os.path.expanduser(raw.strip()))
    return base if base.is_absolute() else None


def data_dir() -> Path:
    """Return the dedicated, portable data directory (absolute, not created here).

    Raises ``RuntimeError`` if the explicit override is relative: unlike a
    system base, an override is a deliberate choice, so silently storing
    somewhere else would be worse than refusing.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override and override.strip():
        path = Path(os.path.expanduser(override.strip()))
        if not path.is_absolute():
            raise RuntimeError(
                "%s must be an absolute path (got %r): a relative one follows "
                "the process's current directory, so two launches of the "
                "server would read two different stores"
                % (_ENV_OVERRIDE, override)
            )
        return path

    if sys.platform.startswith("win"):
        for var in ("LOCALAPPDATA", "APPDATA"):
            base = _absolute_base(var)
            if base is not None:
                return base / APP_NAME
        return _home() / "AppData" / "Local" / APP_NAME

    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP_NAME

    xdg = _absolute_base("XDG_DATA_HOME")
    if xdg is not None:
        return xdg / APP_NAME
    return _home() / ".local" / "share" / APP_NAME


def multidim_dir() -> Path:
    return data_dir() / "multidim"


def assert_not_personal(path) -> None:
    """Raise if ``path`` is the personal ``~/.multidim`` dir or anywhere beneath.

    Called before every store read AND write, so even an explicit path argument
    (bypassing :func:`store_path`) can never touch the personal store.
    ``relative_to`` succeeds (returns ``.`` or a subpath) exactly when ``path`` is
    equal to or under ``personal``; ``resolve()`` also defeats a symlink pointing
    into the personal store.
    """
    personal = Path(os.path.expanduser(_FORBIDDEN_PERSONAL))
    try:
        personal = personal.resolve()
    except OSError:
        pass
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    try:
        resolved.relative_to(personal)
        inside_personal = True
    except ValueError:
        inside_personal = False
    if inside_personal:
        raise RuntimeError(
            "refusing: this store must never resolve inside ~/.multidim, "
            "which belongs to a separate personal installation"
        )


def store_path() -> Path:
    """Absolute path of the OSS Multidim store (dedicated, never ~/.multidim)."""
    path = multidim_dir() / "store.json"
    assert_not_personal(path)  # defence in depth on the default path too
    return path
