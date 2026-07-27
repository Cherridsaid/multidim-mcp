# Security policy

## Threat model

`multidim-mcp` is a **local** process. It speaks JSON-RPC over stdin/stdout to
the client that spawned it, opens no socket, makes no network call and has no
dependencies. The asset it protects is therefore local data: the store file it
owns, and the files around it.

The guarantees it tries to hold:

- every read and write stays inside its own data directory, and never resolves
  into `~/.multidim`, which belongs to a separate personal installation;
- a store it cannot read is backed up before anything is reset — never
  discarded silently;
- input from the client is refused when malformed, rather than coerced into
  something that looks valid.

Out of scope: whatever the calling client does with the grid it receives, and
the trust you place in that client.

## Reporting a vulnerability

Please report privately, not through a public issue: open a
[security advisory](https://github.com/Cherridsaid/multidim-mcp/security/advisories/new)
on the repository.

Include what you ran, what happened and what you expected. A reproducer that
fails on the current `main` is worth more than a description.

Expect a first answer within a week. This is a small project maintained by one
person; that is the honest figure, not a service commitment.

## Supported versions

The latest released version only.
