# Contributing

Thanks for taking the time. This project is small, dependency-free and meant to
stay that way.

## Running the tests

```bash
python run_tests.py          # the whole suite
python smoke_install.py      # build a wheel, install it in a throwaway venv
```

`run_tests.py` must be green on **Linux and Windows** before a pull request.
Several past defects only showed on one of the two: a file name taken by a
directory raises a different error per platform, and the JSON decoder gives up
at a different nesting depth. A green run on your machine alone is not proof.

## House rules

**No dependencies.** The server is spawned by arbitrary MCP clients; a single
third-party import is a supply-chain risk for every one of them. Everything
here is standard library, including the data-directory resolution.

**Comments carry invariants, not history.** Explain *why* a check exists and
what breaks without it. Do not record when it was added, which review found it
or which iteration it came from — that belongs in the changelog and in git.

**Refuse, do not rewrite.** When input is malformed, return an error the caller
can act on. Silently substituting a sane value is how a lesson gets replaced by
its opposite without anyone noticing.

**Every fix ships with a test that fails without it.** If the test passes on the
unfixed code, it proves nothing — this has happened here and was caught late.

## Scope

Multidim provides *structure*, not cognition: it returns a grid for a caller to
fill in. Proposals that move reasoning into the server itself are out of scope.
