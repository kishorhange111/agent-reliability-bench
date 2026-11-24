"""Deterministic scripted user simulator.

The user only reveals what the agent asks for (identity, order id, item,
refund preference, reason, address) and confirms only when explicitly asked.
Rule-based rather than LLM-based so that runs are exactly reproducible and the
benchmark measures the *agent*, not simulator variance (a known confound in
LLM-simulated-user benchmarks such as tau-bench).
"""

from __future__ import annotations

import re

from .tasks import Task

ASK = {
    "email": re.compile(r"\be-?mail\b", re.I),
    "name_zip": re.compile(r"\b(full name|your name|zip|postal)\b", re.I),
    "order": re.compile(r"\border (id|number)\b|\bwhich order\b", re.I),
    "item": re.compile(r"\bwhich (item|product)s?\b", re.I),
    "refund": re.compile(r"\b(refund method|gift card|original payment|how .* refund)\b", re.I),
    "reason": re.compile(r"\breason\b", re.I),
    "address": re.compile(r"\b(new address|address)\b", re.I),
    "confirm": re.compile(
        r"\b(confirm|shall i|should i|would you like me to|do you want me to|proceed)\b", re.I
    ),
    "anything_else": re.compile(r"\b(anything else|is there anything|all set)\b", re.I),
}

END = "###STOP###"


class ScriptedUser:
    def __init__(self, task: Task):
        self.task = task
        self.f = task.facts
        self.turns = 0

    def opening(self) -> str:
        return self.task.opening

    def reply(self, agent_text: str) -> str:
        self.turns += 1
        t = agent_text
        f = self.f
        parts: list[str] = []
        if ASK["email"].search(t):
            if "email" in f:
                parts.append(f"My email is {f['email']}.")
            else:
                parts.append(
                    f"I don't remember which email I used. I'm {f['name']}, zip {f['zip']}."
                )
        elif ASK["name_zip"].search(t):
            parts.append(f"My name is {f['name']} and my zip code is {f['zip']}.")
        if ASK["order"].search(t) and "order_id" in f:
            parts.append(f"The order is {f['order_id']}.")
        if ASK["item"].search(t) and "item_name" in f:
            parts.append(f"The {f['item_name']}.")
        if ASK["refund"].search(t) and "refund_method" in f:
            parts.append(
                "To a gift card, please."
                if f["refund_method"] == "gift_card"
                else "Back to my original payment method."
            )
        if ASK["reason"].search(t) and "reason" in f:
            parts.append(f"The reason is: {f['reason']}.")
        if ASK["address"].search(t) and "new_address" in f and not ASK["confirm"].search(t):
            parts.append(f"The new address is {f['new_address']}.")
        if ASK["confirm"].search(t):
            parts.append("Yes, please go ahead.")
        if parts:
            return " ".join(parts)
        if ASK["anything_else"].search(t):
            return f"No, that's all. Thanks! {END}"
        return "Okay. " + self.task.opening if self.turns < 3 else f"Thanks. {END}"
