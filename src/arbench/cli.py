"""Run agents across fault rates and trials; write JSON results and a markdown table.

arbench --agents oracle oracle_noretry rule --faults 0 0.1 0.2 0.3 --trials 8
arbench --agents claude --faults 0 0.2 --trials 4      # needs ANTHROPIC credentials
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import OracleAgent, RuleAgent
from .env import FaultConfig
from .runner import run_episode, summarize
from .tasks import build_db, build_tasks


def make_agent(name: str, model: str):
    if name == "oracle":
        return OracleAgent()
    if name == "oracle_noretry":
        a = OracleAgent(retry=False)
        a.name = "oracle_noretry"
        return a
    if name == "rule":
        return RuleAgent()
    if name == "rule_noretry":
        a = RuleAgent(max_retries=0)
        a.name = "rule_noretry"
        return a
    if name == "claude":
        from .llm_agent import AnthropicBackend, LLMAgent

        return LLMAgent(AnthropicBackend(model=model), name=f"claude:{model}")
    raise SystemExit(f"unknown agent {name!r}")


def table(rows: list[dict], ks: list[int]) -> str:
    head = "| agent | fault rate | " + " | ".join(f"pass^{k}" for k in ks)
    head += " | violation rate | tool calls/ep |"
    out = [head, "|" + "---|" * (len(ks) + 4)]
    for r in rows:
        cells = " | ".join(f"{r[f'pass^{k}']:.3f}" for k in ks)
        out.append(
            f"| {r['agent']} | {r['fault_rate']:.2f} | {cells} | {r['violation_rate']:.3f} | "
            f"{r['mean_tool_calls']:.1f} |"
        )
    return "\n".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="arbench")
    p.add_argument("--agents", nargs="+", default=["oracle", "oracle_noretry", "rule"])
    p.add_argument("--faults", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args(argv)
    ks = [k for k in (1, 2, 4, 8) if k <= a.trials]

    db = build_db()
    tasks = build_tasks(db)
    rows, raw = [], {}
    for name in a.agents:
        agent = make_agent(name, a.model)
        for f in a.faults:
            res = [
                run_episode(agent, t, db, trial=i, faults=FaultConfig(f))
                for t in tasks
                for i in range(a.trials)
            ]
            s = summarize(res, ks=ks)
            rows.append({"agent": agent.name, "fault_rate": f, **s})
            raw[f"{agent.name}@{f}"] = [r.__dict__ for r in res]
            print(f"{agent.name} faults={f}: pass^1={s['pass^1']:.3f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=1))
    (a.out / "episodes.json").write_text(json.dumps(raw))
    md = table(rows, ks)
    (a.out / "results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
