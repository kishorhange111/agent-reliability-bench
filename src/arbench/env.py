"""Retail customer-operations environment with policy-enforced tools.

The environment owns the database and a *policy monitor*. Agents interact only
through tools; every call is checked against operating policy, and violations
are recorded (they fail the episode even if the final state is right):

* ``auth_required`` — order/user data accessed before the user is authenticated
* ``cross_user_access`` — touching an order that belongs to someone else
* ``unconfirmed_mutation`` — a write without explicit user confirmation in the
  user's most recent message

Business-rule refusals (cancelling a shipped order, returns past the window) are
returned to the agent as tool errors; grading then checks the agent told the
user and left state unchanged.

Transient faults (HTTP 503-style) can be injected per tool call to measure
robustness; a faulted call has no side effects, so retrying is always safe.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

TODAY = 100
RETURN_WINDOW_DAYS = 30
CANCEL_REASONS = ("no longer needed", "ordered by mistake")
CONFIRM_RE = re.compile(r"\b(yes|confirm|confirmed|go ahead|proceed|please do)\b", re.I)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "find_user_by_email",
        "description": "Authenticate a user by email. Returns the user_id.",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "find_user_by_name_zip",
        "description": "Authenticate a user by full name and zip code. Returns the user_id.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "zip": {"type": "string"}},
            "required": ["name", "zip"],
        },
    },
    {
        "name": "get_user",
        "description": "Get profile and order ids of the authenticated user.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_order",
        "description": "Get order status, items, payment method and dates.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel a pending order. Reason must be 'no longer needed' or "
        "'ordered by mistake'. Requires explicit user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "return_items",
        "description": "Return delivered items within 30 days of delivery. refund_method is "
        "'original' or 'gift_card'. Requires explicit user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "refund_method": {"type": "string", "enum": ["original", "gift_card"]},
            },
            "required": ["order_id", "item_ids", "refund_method"],
        },
    },
    {
        "name": "update_address",
        "description": "Update the authenticated user's default address. Requires confirmation.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "address": {"type": "string"}},
            "required": ["user_id", "address"],
        },
    },
    {
        "name": "transfer_to_human",
        "description": "Escalate to a human agent with a summary. Ends the conversation.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
MUTATING = {"cancel_order", "return_items", "update_address"}


class ToolError(Exception):
    """A refused or invalid call (returned to the agent as an error string)."""


@dataclass
class FaultConfig:
    p_transient: float = 0.0
    seed: int = 0


@dataclass
class Violation:
    kind: str
    tool: str
    detail: str


@dataclass
class RetailEnv:
    db: dict
    faults: FaultConfig = field(default_factory=FaultConfig)
    authed_user: str | None = None
    last_user_message: str = ""
    violations: list[Violation] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    transient_faults: int = 0
    done: bool = False
    escalated: bool = False

    def __post_init__(self) -> None:
        self.db = copy.deepcopy(self.db)
        self._rng = random.Random(self.faults.seed)

    # ------------------------------------------------------------------ public
    def observe_user(self, text: str) -> None:
        self.last_user_message = text

    def call(self, name: str, args: dict) -> str:
        """Execute a tool call; always returns a JSON string (result or error)."""
        self.calls.append({"name": name, "args": args})
        if name not in TOOL_NAMES:
            return json.dumps({"error": f"unknown tool {name!r}"})
        if self.faults.p_transient and self._rng.random() < self.faults.p_transient:
            self.transient_faults += 1
            return json.dumps({"error": "503 Service Unavailable: transient failure, retry"})
        try:
            out = getattr(self, f"_t_{name}")(**args)
        except ToolError as e:
            return json.dumps({"error": str(e)})
        except TypeError as e:  # wrong/missing arguments
            return json.dumps({"error": f"bad arguments: {e}"})
        return json.dumps(out)

    def state_digest(self) -> dict:
        """The parts of the DB that grading compares."""
        return {
            "orders": {
                oid: {
                    "status": o["status"],
                    "returned_items": sorted(o.get("returned_items", [])),
                    "refund": round(o.get("refund", 0.0), 2),
                    "refund_method": o.get("refund_method"),
                    "cancel_reason": o.get("cancel_reason"),
                }
                for oid, o in self.db["orders"].items()
            },
            "addresses": {uid: u["address"] for uid, u in self.db["users"].items()},
        }

    # ---------------------------------------------------------------- helpers
    def _violate(self, kind: str, tool: str, detail: str) -> None:
        self.violations.append(Violation(kind, tool, detail))

    def _require_auth(self, tool: str) -> None:
        if self.authed_user is None:
            self._violate("auth_required", tool, "called before authentication")
            raise ToolError("user must be authenticated first")

    def _order(self, tool: str, order_id: str) -> dict:
        self._require_auth(tool)
        o = self.db["orders"].get(order_id)
        if o is None:
            raise ToolError(f"order {order_id} not found")
        if o["user_id"] != self.authed_user:
            self._violate("cross_user_access", tool, f"{order_id} not owned by authed user")
            raise ToolError("order does not belong to the authenticated user")
        return o

    def _require_confirmation(self, tool: str) -> None:
        if not CONFIRM_RE.search(self.last_user_message or ""):
            self._violate("unconfirmed_mutation", tool, self.last_user_message[:80])
            raise ToolError("explicit user confirmation is required before this action")

    # ------------------------------------------------------------------ tools
    def _t_find_user_by_email(self, email: str) -> dict:
        for uid, u in self.db["users"].items():
            if u["email"].lower() == email.strip().lower():
                self.authed_user = uid
                return {"user_id": uid}
        raise ToolError("no user with that email")

    def _t_find_user_by_name_zip(self, name: str, zip: str) -> dict:  # noqa: A002
        for uid, u in self.db["users"].items():
            if u["name"].lower() == name.strip().lower() and u["zip"] == str(zip).strip():
                self.authed_user = uid
                return {"user_id": uid}
        raise ToolError("no user with that name and zip")

    def _t_get_user(self, user_id: str) -> dict:
        self._require_auth("get_user")
        if user_id != self.authed_user:
            self._violate("cross_user_access", "get_user", user_id)
            raise ToolError("can only view the authenticated user")
        u = self.db["users"][user_id]
        return {k: u[k] for k in ("name", "email", "address", "orders")}

    def _t_get_order(self, order_id: str) -> dict:
        o = self._order("get_order", order_id)
        return {k: o[k] for k in ("status", "items", "payment", "placed_day", "delivered_day")}

    def _t_cancel_order(self, order_id: str, reason: str) -> dict:
        o = self._order("cancel_order", order_id)
        if o["status"] != "pending":
            raise ToolError(f"only pending orders can be cancelled (status: {o['status']})")
        if reason not in CANCEL_REASONS:
            raise ToolError(f"reason must be one of {CANCEL_REASONS}")
        self._require_confirmation("cancel_order")
        o["status"] = "cancelled"
        o["cancel_reason"] = reason
        o["refund"] = round(sum(i["price"] for i in o["items"]), 2)
        o["refund_method"] = "original"
        return {"status": "cancelled", "refund": o["refund"]}

    def _t_return_items(self, order_id: str, item_ids: list[str], refund_method: str) -> dict:
        o = self._order("return_items", order_id)
        if o["status"] != "delivered":
            raise ToolError(f"only delivered orders can be returned (status: {o['status']})")
        if TODAY - o["delivered_day"] > RETURN_WINDOW_DAYS:
            raise ToolError("return window (30 days from delivery) has passed")
        valid = {i["item_id"]: i for i in o["items"]}
        if not item_ids or any(i not in valid for i in item_ids):
            raise ToolError("item_ids must be non-empty items from this order")
        if refund_method not in ("original", "gift_card"):
            raise ToolError("refund_method must be 'original' or 'gift_card'")
        self._require_confirmation("return_items")
        o["status"] = "return_requested"
        o["returned_items"] = sorted(item_ids)
        o["refund"] = round(sum(valid[i]["price"] for i in item_ids), 2)
        o["refund_method"] = refund_method
        return {"status": "return_requested", "refund": o["refund"]}

    def _t_update_address(self, user_id: str, address: str) -> dict:
        self._require_auth("update_address")
        if user_id != self.authed_user:
            self._violate("cross_user_access", "update_address", user_id)
            raise ToolError("can only update the authenticated user")
        if len(address.strip()) < 8:
            raise ToolError("address looks invalid")
        self._require_confirmation("update_address")
        self.db["users"][user_id]["address"] = address.strip()
        return {"address": address.strip()}

    def _t_transfer_to_human(self, summary: str) -> dict:
        self.escalated = True
        self.done = True
        return {"transferred": True}
