# agent-reliability-bench

[![CI](https://github.com/kishorhange111/agent-reliability-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/agent-reliability-bench/actions/workflows/ci.yml)

A stateful tool-use benchmark for customer-facing LLM agents, built around the questions a
forward-deployed engineer gets asked before an agent goes live:

1. **Does it do the right thing to the system of record?** — graded on the *final database
   state*, not on the transcript.
2. **Does it follow operating policy?** — authentication before data access, no cross-customer
   access, explicit confirmation before every write.
3. **Does it do it *every* time?** — reliability via **pass^k** (all *k* independent trials
   succeed), not just average accuracy.
4. **Does it survive a flaky backend?** — injected transient 503s on tool calls.

Pure-Python core (no dependencies); optional Claude adapter.

## Design

```
ScriptedUser ──text──▶ Agent ──tool call──▶ RetailEnv (DB + policy monitor + fault injector)
     ▲                   │                         │
     └──────text─────────┘◀────── JSON result ─────┘
                                   grade(): final state == ground truth
                                            ∧ required info told to user
                                            ∧ zero policy violations
```

| Component | What it does |
|---|---|
| [`env.py`](src/arbench/env.py) | 8 JSON-schema tools over an in-memory retail DB. A **policy monitor** records `auth_required`, `cross_user_access` and `unconfirmed_mutation` violations; business rules (shipped orders, 30-day return window) are refused as tool errors. Fault injection returns side-effect-free 503s |
| [`tasks.py`](src/arbench/tasks.py) | 31 seeded tasks in 8 categories — cancel, return in/out of window, address change, status, multi-intent, and two privacy probes ("what's in my friend's order?"). Ground-truth final states come from replaying reference actions, so every task is provably solvable |
| [`user_sim.py`](src/arbench/user_sim.py) | Deterministic user that only reveals what it is asked for and confirms only when asked. Rule-based on purpose: LLM-simulated users add variance that gets attributed to the agent |
| [`runner.py`](src/arbench/runner.py) | Episode loop, grading with a failure taxonomy, unbiased **pass^k = C(c,k)/C(n,k)** |
| [`agents.py`](src/arbench/agents.py) | `OracleAgent` (answer-key executor — harness validation) and `RuleAgent` (slot-filling, policy-aware, never sees the answer) |
| [`llm_agent.py`](src/arbench/llm_agent.py) | Claude adapter (Messages API, manual tool loop): batches all `tool_result` blocks of a turn into one user message, appends assistant content unchanged, handles `refusal`; server-side refusal fallback enabled. Backend protocol makes other providers a ~20-line class |

## Results (reference agents, 31 tasks × 8 trials per cell)

| agent | fault rate | pass^1 | pass^2 | pass^4 | pass^8 | violation rate |
|---|---|---|---|---|---|---|
| oracle (retries) | 0.00 – 0.30 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| oracle, no retries | 0.10 | 0.911 | 0.838 | 0.724 | 0.581 | 0.028 |
| oracle, no retries | 0.30 | 0.677 | 0.514 | 0.388 | 0.355 | 0.137 |
| rule (3 retries) | 0.00 | 0.935 | 0.935 | 0.935 | 0.935 | 0.065 |
| rule (3 retries) | 0.30 | 0.911 | 0.887 | 0.839 | 0.742 | 0.065 |
| rule, no retries | 0.10 | 0.774 | 0.646 | 0.471 | 0.323 | 0.048 |
| rule, no retries | 0.30 | 0.492 | 0.310 | 0.187 | 0.129 | 0.032 |

Full table: [`results/results.md`](results/results.md). Reproduce:
`arbench --agents oracle oracle_noretry rule rule_noretry --faults 0 0.1 0.2 0.3 --trials 8`.

**Takeaways**

- **Average accuracy hides unreliability.** At a 10% fault rate the no-retry rule agent's
  pass^1 is 0.77, but the chance it handles the *same* request correctly 8 times in a row is
  0.32. That gap is what customers experience as "it works sometimes".
- **Faults turn into policy violations, not just failures.** Without retries, a failed
  authentication call is followed by unauthenticated data access: the oracle's violation
  rate rises from 0 to 13.7% at 30% faults. Error handling is a *safety* property.
- **Bounded retries recover most of the loss** (rule agent pass^8 0.13 → 0.74 at 30% faults)
  but not all: three retries still get exhausted on multi-call tasks.
- **The privacy probes catch a naive agent.** The rule agent fails both cross-user tasks by
  calling `get_order` on someone else's order. The tool refuses, but the *attempt* is a
  violation — this is the class of bug a transcript-level "did it answer?" eval misses.

### Running Claude

```bash
pip install -e ".[anthropic]"
arbench --agents claude --faults 0 0.2 --trials 4 --out results/claude
```

The adapter defaults to `claude-opus-5-5` (`--model` to change) and enables the server-side
refusal fallback. No LLM numbers are reported here yet: they will be added only from real runs.

## Limitations

- One domain (retail) and a scripted user: it measures tool-use discipline and reliability,
  not open-ended conversation quality.
- The `RuleAgent` is a deliberately strong non-LLM baseline for *this* domain; it is not a
  claim about LLM capability.

## References

Yao et al. 2024, *τ-bench* (pass^k, state-based grading) · Anthropic, *Building effective agents* ·
Chen et al. 2021 (unbiased pass@k estimator)
