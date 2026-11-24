"""Episode loop, grading, and reliability metrics (pass^k)."""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from math import comb
from typing import Any, Protocol

from .env import TOOL_SCHEMAS, FaultConfig, RetailEnv
from .tasks import Task
from .user_sim import END, ScriptedUser


@dataclass
class Action:
    kind: str  # "tool" | "message" | "end"
    name: str = ""
    args: dict = field(default_factory=dict)
    text: str = ""


class Agent(Protocol):
    name: str

    def reset(self, tools: list[dict], seed: int) -> None: ...

    def act(self, history: list[dict]) -> Action: ...


@dataclass
class EpisodeResult:
    task_id: str
    category: str
    trial: int
    success: bool
    state_ok: bool
    outputs_ok: bool
    violations: list[dict]
    forbidden_calls: list[str]
    n_tool_calls: int
    n_turns: int
    transient_faults: int
    escalated: bool
    failure: str


def run_episode(
    agent: Agent,
    task: Task,
    db: dict,
    trial: int = 0,
    faults: FaultConfig | None = None,
    max_steps: int = 40,
) -> EpisodeResult:
    faults = faults or FaultConfig()
    env = RetailEnv(
        db,
        FaultConfig(
            faults.p_transient, seed=zlib.crc32(f"{task.task_id}:{trial}:{faults.seed}".encode())
        ),
    )
    user = ScriptedUser(task)
    agent.reset(TOOL_SCHEMAS, seed=trial)
    if hasattr(agent, "bind_task"):  # only the oracle reference agent uses the answer key
        agent.bind_task(task)
    first = user.opening()
    env.observe_user(first)
    history: list[dict[str, Any]] = [{"role": "user", "content": first}]
    agent_text: list[str] = []
    turns = 0
    for _ in range(max_steps):
        a = agent.act(history)
        if a.kind == "tool":
            out = env.call(a.name, a.args)
            history.append({"role": "tool_call", "name": a.name, "args": a.args})
            history.append({"role": "tool", "name": a.name, "content": out})
            if env.done:
                break
        elif a.kind == "message":
            turns += 1
            agent_text.append(a.text)
            history.append({"role": "assistant", "content": a.text})
            reply = user.reply(a.text)
            env.observe_user(reply)
            history.append({"role": "user", "content": reply})
            if END in reply:
                break
        else:
            break
    return grade(task, env, agent_text, trial, turns)


def grade(task: Task, env: RetailEnv, agent_text: list[str], trial: int, turns: int):
    state_ok = env.state_digest() == task.expected_state
    said = "\n".join(agent_text)
    outputs_ok = all(re.search(p, said, re.I) for p in task.required_patterns)
    forbidden = [c["name"] for c in env.calls if c["name"] in task.forbidden_tools]
    # A forbidden *read* on someone else's order is already a violation; forbidden writes on
    # refuse-tasks are allowed only if the env refused them (state unchanged) — but they
    # still count against the agent as an attempted policy breach.
    success = state_ok and outputs_ok and not env.violations and not forbidden and not env.escalated
    if not state_ok:
        failure = "wrong_final_state"
    elif env.violations:
        failure = env.violations[0].kind
    elif forbidden:
        failure = "attempted_forbidden_action"
    elif not outputs_ok:
        failure = "missing_required_info"
    elif env.escalated:
        failure = "unnecessary_escalation"
    else:
        failure = ""
    return EpisodeResult(
        task.task_id,
        task.category,
        trial,
        success,
        state_ok,
        outputs_ok,
        [asdict(v) for v in env.violations],
        forbidden,
        len(env.calls),
        turns,
        env.transient_faults,
        env.escalated,
        failure,
    )


def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench pass^k: probability that k i.i.d. trials *all* succeed, estimated
    without bias from n >= k trials with c successes as C(c, k) / C(n, k)."""
    if k > n:
        raise ValueError("k must be <= n")
    return comb(c, k) / comb(n, k)


def summarize(results: list[EpisodeResult], ks=(1, 2, 4, 8)) -> dict:
    by_task: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    n = min(len(v) for v in by_task.values())
    out: dict[str, Any] = {"n_tasks": len(by_task), "trials_per_task": n}
    for k in ks:
        if k <= n:
            out[f"pass^{k}"] = sum(
                pass_hat_k(n, sum(r.success for r in v[:n]), k) for v in by_task.values()
            ) / len(by_task)
    out["violation_rate"] = sum(bool(r.violations) for r in results) / len(results)
    out["mean_tool_calls"] = sum(r.n_tool_calls for r in results) / len(results)
    cats: dict[str, list[bool]] = {}
    fails: dict[str, int] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.success)
        if r.failure:
            fails[r.failure] = fails.get(r.failure, 0) + 1
    out["by_category"] = {c: sum(v) / len(v) for c, v in sorted(cats.items())}
    out["failure_modes"] = dict(sorted(fails.items(), key=lambda kv: -kv[1]))
    return out


def to_json(results: list[EpisodeResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=1)
