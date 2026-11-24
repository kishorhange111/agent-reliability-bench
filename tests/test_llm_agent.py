from types import SimpleNamespace as NS

from arbench.llm_agent import LLMAgent, to_anthropic_tools
from arbench.runner import run_episode
from arbench.tasks import build_db, build_tasks


def text(t):
    return NS(type="text", text=t)


def tool(i, name, inp):
    return NS(type="tool_use", id=i, name=name, input=inp)


class ScriptedBackend:
    """Replays a fixed list of responses and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent = []

    def create(self, system, messages, tools):
        self.sent.append([dict(m) for m in messages])
        content, stop = self.turns.pop(0)
        return NS(content=content, stop_reason=stop)


def test_tool_schema_conversion():
    t = to_anthropic_tools([{"name": "x", "description": "d", "parameters": {"type": "object"}}])
    assert t == [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]


def test_llm_agent_solves_cancel_task_and_batches_tool_results():
    db = build_db()
    task = next(t for t in build_tasks(db) if t.category == "cancel_pending")
    oid, reason = task.facts["order_id"], task.facts["reason"]
    auth = (
        ("find_user_by_email", {"email": task.facts["email"]})
        if task.auth == "email"
        else ("find_user_by_name_zip", {"name": task.facts["name"], "zip": task.facts["zip"]})
    )
    backend = ScriptedBackend(
        [
            ([text("Let me verify you.")], "end_turn"),  # asks nothing specific
            # two parallel calls in one turn -> results must come back in ONE user message
            ([tool("a", *auth), tool("b", "get_order", {"order_id": oid})], "tool_use"),
            ([text(f"Cancel {oid} as '{reason}'. Shall I proceed?")], "end_turn"),
            ([tool("c", "cancel_order", {"order_id": oid, "reason": reason})], "tool_use"),
            ([text("Cancelled. Refund: see below. Anything else?")], "end_turn"),
        ]
    )
    agent = LLMAgent(backend)
    res = run_episode(agent, task, db)
    assert res.state_ok and not res.violations
    batched = backend.sent[2][-1]  # user message carrying tool results
    assert [b["tool_use_id"] for b in batched["content"]] == ["a", "b"]


def test_refusal_ends_episode():
    db = build_db()
    task = build_tasks(db)[0]
    agent = LLMAgent(ScriptedBackend([([], "refusal")]))
    res = run_episode(agent, task, db)
    assert not res.success and res.n_tool_calls == 0
