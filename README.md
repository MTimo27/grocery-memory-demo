# grocery-memory-demo

This is a small Python experiment about customer memory in a grocery-shopping agent.

I built it to make one research question concrete: if an agent remembers something about
a household, how should it decide whether to use that memory, ignore it, or ask the
customer first?

The project uses one synthetic household with 16 weeks of orders. The history contains a
few deliberately planted patterns: a lactose-free preference, a peanut allergy mentioned
once in an order note, a keto phase that later stops, and a one-off party order. Claude
extracts readable claims from that history. Everything after extraction—reliability,
expiry, policy decisions, and the final allergen check—is ordinary Python.

This prototype is designed for inspection and discussion. Its data is synthetic, its
scoring constants are hand-tuned, and its live agent is intentionally small, so the
results do not establish production performance.

## What is being compared

The demo runs the same shopping agent three ways:

- **A — purchase counts:** the agent only sees how often products appeared in previous
  orders.
- **B — memory:** it also sees the extracted memory claims and their reliability scores.
- **C — memory + policy:** it sees the same memory, plus a deterministic `use`, `ignore`,
  or `ask` verdict for every claim.

The distinction matters because storing a claim and deciding to act on it are different
problems. A stale preference may still be useful, but perhaps only as a question. An
allergy should not depend on the model making the right judgement at all, so the basket is
checked in code after the agent finishes.

## Running it

Python 3.12 is required.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

Put an Anthropic API key in `.env`, then run the full demo:

```bash
.venv/bin/python demo.py
```

You can also run one scenario at a time:

```bash
.venv/bin/python demo.py weekly
.venv/bin/python demo.py keto
.venv/bin/python demo.py allergy
```

The demo makes live API calls. The tests and replay evaluation do not:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## The memory file

[out/memory.json](out/memory.json) is a generated artifact, but it is kept in Git on
purpose. It is readable, diffable, and lets the demo run against a known memory snapshot.
`demo.py` reads this file; it never rebuilds it automatically.

To extract fresh initial memory from training weeks 1-12 only:

```bash
.venv/bin/python build_memory.py
```

This overwrites `out/memory.json`; its diff is part of the experiment. The current file
contains 14 claims built only from weeks 1-12. It contains no evidence from the test weeks
13-16. The model returns claims and evidence IDs, and the Python code checks them and
calculates their dates and reliability scores.

## Current replay result

The offline replay trains on weeks 1–12 and evaluates three approaches over four held-out
weeks:

```text
arm                       correct  stale  missed  asked  unsafe
---------------------------------------------------------------
A  purchase counts             29      7       3      0       4
B  memory                      32     20       0      0       0
C  memory + policy             24      2       6      7       0
```

These results come from one synthetic household. Raw memory often acts on outdated
preferences. The policy reduces those mistakes, but it asks more questions and sometimes
misses useful patterns. A separate safety check catches the peanut violation.
