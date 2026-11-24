"""Reference agents.

* ``OracleAgent`` — follows each task's ground-truth plan with correct auth and
  confirmation. Validates the harness (must score 100% without faults) and,
  with ``retry=False``, isolates the cost of not handling transient errors.
* ``RuleAgent`` — a realistic non-LLM baseline: regex intent parsing, slot
  filling through questions, policy-aware execution, bounded retries. It sees
  only the conversation and tool outputs, never the task's answer.
"""

from __future__ import annotations

import json
import random
import re

from .runner import Action
from .tasks import Task

ORDER_RE = re.compile(r"#W\d+")


def _tool_results(history, name=None):
    for h in history:
        if h["role"] == "tool" and (name is None or h["name"] == name):
            yield json.loads(h["content"])


def _last(history, role):
    for h in reversed(history):
        if h["role"] == role:
            return h
    return None


class OracleAgent:
    name = "oracle"

    def __init__(self, retry: bool = True, max_retries: int = 5):
        self.retry = retry
        self.max_retries = max_retries

    def reset(self, tools, seed):
        self.plan: list[Action] | None = None
        self.pending: Action | None = None
        self.retries = 0

    def bind_task(self, task: Task) -> None:
        self.task = task

    def act(self, history) -> Action:
        if self.plan is None:
            t = self.task
            auth = (
                Action("tool", "find_user_by_email", {"email": t.facts["email"]})
                if t.auth == "email"
                else Action(
                    "tool",
                    "find_user_by_name_zip",
                    {"name": t.facts["name"], "zip": t.facts["zip"]},
                )
            )
            plan = [auth]
            for name, args in t.actions:
                plan.append(Action("message", text=f"I will run {name}. Shall I proceed?"))
                plan.append(Action("tool", name, args))
            self.plan = plan
        last = history[-1]
        if self.pending is not None and last["role"] == "tool":
            out = json.loads(last["content"])
            if "error" in out and "503" in out["error"] and self.retry:
                if self.retries < self.max_retries:
                    self.retries += 1
                    return self.pending
            self.pending = None
        if self.plan:
            a = self.plan.pop(0)
            if a.kind == "tool":
                self.pending = a
                self.retries = 0
            return a
        return Action("message", text=self._summary(history) + " Is there anything else?")

    def _summary(self, history) -> str:
        t = self.task
        bits = []
        for out in _tool_results(history):
            if "refund" in out:
                bits.append(f"Refund of ${out['refund']:.2f} has been issued.")
        if t.category in ("cancel_shipped_refuse", "return_out_of_window_refuse"):
            bits.append("Sorry, I cannot do that: the order is not eligible under our policy.")
        if t.category == "cross_user_refuse":
            bits.append("Sorry, I can't share details of an order on another person's account.")
        if t.category == "status_inquiry":
            bits.append("Your order has shipped.")
        return " ".join(bits) or "Done."


class RuleAgent:
    """Slot-filling, policy-aware agent built from rules — no access to the answer key."""

    name = "rule"

    def __init__(self, max_retries: int = 3, seed_noise: float = 0.0):
        self.max_retries = max_retries
        self.seed_noise = seed_noise

    def reset(self, tools, seed):
        self.rng = random.Random(seed)
        self.user_id = None
        self.intents: list[str] = []
        self.queue: list[Action] = []
        self.done_intents: set[str] = set()
        self.retries = 0
        self.last_tool: Action | None = None
        self.notes: list[str] = []
        self.awaiting: str | None = None
        self.slots: dict = {}
        self.order_info: dict = {}

    # ------------------------------------------------------------ parsing
    def _parse_intents(self, text: str) -> list[str]:
        t = text.lower()
        intents = []
        if "cancel" in t:
            intents.append("cancel")
        if "return" in t:
            intents.append("return")
        if "address" in t or "moved" in t:
            intents.append("address")
        if "status" in t or "what's in" in t or "tell me what" in t:
            intents.append("info")
        return intents or ["info"]

    def _user_texts(self, history):
        return [h["content"] for h in history if h["role"] == "user"]

    def _extract_slots(self, history):
        """Fill slots message by message (first value wins), never from concatenated text."""
        for text in self._user_texts(history):
            if m := ORDER_RE.search(text):
                self.slots.setdefault("order_id", m.group(0))
            if m := re.search(r"[\w.]+@[\w.]+\.\w+", text):
                self.slots.setdefault("email", m.group(0).rstrip("."))
            if m := re.search(r"I'm ([A-Z][a-z]+ [A-Z][a-z]+), zip (\d+)", text) or re.search(
                r"name is ([A-Z][a-z]+ [A-Z][a-z]+) and my zip code is (\d+)", text
            ):
                self.slots.setdefault("name", m.group(1))
                self.slots.setdefault("zip", m.group(2))
            if m := re.search(r"new address is (.+?)\.(?:\s|$)", text):
                self.slots.setdefault("address", m.group(1))
            if "gift card" in text.lower():
                self.slots.setdefault("refund_method", "gift_card")
            elif "original payment" in text.lower():
                self.slots.setdefault("refund_method", "original")
            if m := re.search(r"reason is: (.+?)\.", text):
                self.slots.setdefault("reason", m.group(1))
            if m := re.search(r"return the (.+?) from order", text):
                self.slots.setdefault("item_name", m.group(1))

    # ------------------------------------------------------------- policy
    def act(self, history) -> Action:
        last = history[-1]
        if not self.intents:
            self.intents = self._parse_intents(history[0]["content"])
        self._extract_slots(history)

        if last["role"] == "tool":
            out = json.loads(last["content"])
            if "error" in out and "503" in out["error"] and self.retries < self.max_retries:
                self.retries += 1
                return self.last_tool
            self.retries = 0
            self._absorb(last["name"], out)

        # 1) authenticate
        if self.user_id is None:
            if "email" in self.slots and not self.slots.get("email_failed"):
                return self._tool("find_user_by_email", {"email": self.slots["email"]})
            if "name" in self.slots:
                return self._tool(
                    "find_user_by_name_zip", {"name": self.slots["name"], "zip": self.slots["zip"]}
                )
            return Action("message", text="To verify your identity, what is your email address?")

        # 2) work through intents
        for intent in self.intents:
            if intent in self.done_intents:
                continue
            step = getattr(self, f"_do_{intent}")(history)
            if step is not None:
                return step
            self.done_intents.add(intent)

        msg = " ".join(self.notes) if self.notes else "All done."
        self.notes = []
        return Action("message", text=f"{msg} Is there anything else I can help with?")

    def _tool(self, name, args) -> Action:
        a = Action("tool", name, args)
        self.last_tool = a
        return a

    def _absorb(self, name, out):
        if name.startswith("find_user"):
            if "user_id" in out:
                self.user_id = out["user_id"]
            elif name == "find_user_by_email":
                self.slots["email_failed"] = True
        elif name == "get_order":
            self.order_info[self.slots.get("order_id")] = out
        elif name in ("cancel_order", "return_items", "update_address"):
            self.awaiting = None
            if "error" in out:
                self.notes.append(f"Sorry, I cannot complete that: {out['error']}.")
            elif "refund" in out:
                self.notes.append(f"Done — a refund of ${out['refund']:.2f} is on its way.")
            else:
                self.notes.append("Your address has been updated.")
            self.done_intents.add(
                {"cancel_order": "cancel", "return_items": "return", "update_address": "address"}[
                    name
                ]
            )

    def _need_order(self, history):
        oid = self.slots.get("order_id")
        if oid is None:
            return Action("message", text="Which order is this about? Please share the order id.")
        if oid not in self.order_info:
            return self._tool("get_order", {"order_id": oid})
        return None

    def _confirmed(self, history, key) -> bool:
        if self.awaiting != key:
            return False
        last_user = [h for h in history if h["role"] == "user"][-1]["content"]
        return bool(re.search(r"\b(yes|go ahead|proceed)\b", last_user, re.I))

    def _ask_confirm(self, key, text) -> Action:
        self.awaiting = key
        return Action("message", text=f"{text} Shall I proceed?")

    def _do_cancel(self, history):
        if a := self._need_order(history):
            return a
        oid = self.slots["order_id"]
        info = self.order_info[oid]
        if "error" in info:
            self.notes.append(f"Sorry, I cannot access {oid}: {info['error']}.")
            return None
        if info["status"] != "pending":
            self.notes.append(f"Sorry, I cannot cancel {oid}: it is already {info['status']}.")
            return None
        if "reason" not in self.slots:
            return Action("message", text="What is the reason for cancelling?")
        reason = self.slots["reason"].lower()
        reason = (
            reason if reason in ("no longer needed", "ordered by mistake") else "no longer needed"
        )
        if not self._confirmed(history, "cancel"):
            return self._ask_confirm("cancel", f"I'll cancel order {oid} ({reason}).")
        return self._tool("cancel_order", {"order_id": oid, "reason": reason})

    def _do_return(self, history):
        if a := self._need_order(history):
            return a
        oid = self.slots["order_id"]
        info = self.order_info[oid]
        if "error" in info:
            self.notes.append(f"Sorry, I cannot access {oid}: {info['error']}.")
            return None
        if info["status"] != "delivered" or (
            info["delivered_day"] is not None and 100 - info["delivered_day"] > 30
        ):
            self.notes.append(f"Sorry, order {oid} is not eligible for return under our policy.")
            return None
        name = self.slots.get("item_name", "")
        ids = [i["item_id"] for i in info["items"] if i["name"].lower() == name.lower()]
        if not ids:
            return Action("message", text="Which item would you like to return?")
        if "refund_method" not in self.slots:
            return Action(
                "message", text="Should the refund go to your original payment or a gift card?"
            )
        if not self._confirmed(history, "return"):
            return self._ask_confirm("return", f"I'll return {name} from {oid}.")
        return self._tool(
            "return_items",
            {"order_id": oid, "item_ids": ids, "refund_method": self.slots["refund_method"]},
        )

    def _do_address(self, history):
        if "address" not in self.slots:
            return Action("message", text="What is the new address?")
        if not self._confirmed(history, "address"):
            return self._ask_confirm(
                "address", f"I'll set your address to {self.slots['address']}."
            )
        return self._tool(
            "update_address", {"user_id": self.user_id, "address": self.slots["address"]}
        )

    def _do_info(self, history):
        oid = self.slots.get("order_id")
        if oid is None:
            return None
        if oid not in self.order_info:
            return self._tool("get_order", {"order_id": oid})
        info = self.order_info[oid]
        if "error" in info:
            self.notes.append(f"Sorry, I cannot share that: {info['error']}.")
        else:
            self.notes.append(f"Order {oid} status: {info['status']}.")
        return None
