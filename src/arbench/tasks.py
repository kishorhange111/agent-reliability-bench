"""Deterministic database, task suite, and ground-truth final states.

Tasks are generated from a seeded database so the suite is reproducible and
every task is checked to be solvable: its expected final state is produced by
replaying ground-truth actions on a fresh environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .env import CANCEL_REASONS, TODAY, RetailEnv

FIRST = [
    "Asha",
    "Rohan",
    "Meera",
    "Vikram",
    "Priya",
    "Arjun",
    "Neha",
    "Kabir",
    "Isha",
    "Dev",
    "Tara",
    "Nikhil",
    "Sana",
    "Aditya",
    "Riya",
    "Kunal",
]
LAST = ["Patil", "Iyer", "Shah", "Rao", "Kulkarni", "Menon", "Joshi", "Desai"]
PRODUCTS = [
    ("Desk Lamp", 34.99),
    ("Mechanical Keyboard", 89.5),
    ("Water Bottle", 18.25),
    ("Noise-Cancelling Headphones", 199.0),
    ("Yoga Mat", 27.4),
    ("Backpack", 64.0),
    ("Coffee Grinder", 52.75),
    ("USB-C Hub", 41.1),
    ("Running Shoes", 120.0),
    ("Monitor Stand", 45.6),
    ("Wireless Mouse", 29.99),
    ("Notebook Set", 12.5),
]
STREETS = ["MG Road", "Residency Road", "Brigade Road", "Church Street", "Indiranagar 12th Main"]

REFUSAL = r"(cannot|can't|can not|unable|not able|not eligible|isn't eligible|not possible)"


def build_db(seed: int = 7) -> dict:
    rng = random.Random(seed)
    users, orders = {}, {}
    oid = 1000
    for u in range(16):
        uid = f"u{u:02d}"
        name = f"{FIRST[u]} {LAST[u % len(LAST)]}"
        users[uid] = {
            "name": name,
            "email": f"{FIRST[u].lower()}.{LAST[u % len(LAST)].lower()}@example.com",
            "zip": f"560{rng.randint(10, 99):02d}0",
            "address": f"{rng.randint(1, 300)} {rng.choice(STREETS)}, Bengaluru",
            "orders": [],
        }
        # Every user gets one order of each status kind so tasks can be drawn per user.
        for status, delivered_ago in [
            ("pending", None),
            ("shipped", None),
            ("delivered", rng.randint(2, 25)),  # inside the return window
            ("delivered", rng.randint(35, 80)),  # outside the return window
        ]:
            oid += 1
            k = rng.randint(1, 3)
            picks = rng.sample(PRODUCTS, k)
            items = [
                {"item_id": f"i{oid}{j}", "name": n, "price": p} for j, (n, p) in enumerate(picks)
            ]
            placed = TODAY - (delivered_ago or 0) - rng.randint(1, 5)
            orders[f"#W{oid}"] = {
                "user_id": uid,
                "status": status,
                "items": items,
                "payment": rng.choice(["credit_card", "paypal"]),
                "placed_day": placed,
                "delivered_day": TODAY - delivered_ago if delivered_ago else None,
            }
            users[uid]["orders"].append(f"#W{oid}")
    return {"users": users, "orders": orders}


@dataclass
class Task:
    task_id: str
    category: str
    user_id: str
    auth: str  # "email" | "name_zip"
    opening: str
    facts: dict
    actions: list[tuple[str, dict]] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_state: dict | None = None


def _orders_of(db, uid, status, in_window=None):
    out = []
    for oid in db["users"][uid]["orders"]:
        o = db["orders"][oid]
        if o["status"] != status:
            continue
        if in_window is not None and o["delivered_day"] is not None:
            ok = TODAY - o["delivered_day"] <= 30
            if ok != in_window:
                continue
        out.append(oid)
    return out


def build_tasks(db: dict, seed: int = 11) -> list[Task]:
    rng = random.Random(seed)
    uids = list(db["users"])
    tasks: list[Task] = []

    def base_facts(uid, auth):
        u = db["users"][uid]
        f = {"name": u["name"], "zip": u["zip"]}
        if auth == "email":
            f["email"] = u["email"]
        return f

    def add(category, uid, opening, facts, actions=(), patterns=(), forbidden=()):
        auth = "email" if len(tasks) % 2 == 0 else "name_zip"
        f = {**base_facts(uid, auth), **facts}
        tasks.append(
            Task(
                f"t{len(tasks):02d}_{category}",
                category,
                uid,
                auth,
                opening,
                f,
                list(actions),
                list(patterns),
                list(forbidden),
            )
        )

    def money(x):
        return f"{x:.2f}".replace(".", r"\.")

    for uid in uids[0:6]:
        oid = _orders_of(db, uid, "pending")[0]
        o = db["orders"][oid]
        reason = rng.choice(CANCEL_REASONS)
        total = sum(i["price"] for i in o["items"])
        add(
            "cancel_pending",
            uid,
            f"Hi, I want to cancel my order {oid}.",
            {"order_id": oid, "reason": reason},
            [("cancel_order", {"order_id": oid, "reason": reason})],
            [money(total)],
        )

    for uid in uids[6:9]:
        oid = _orders_of(db, uid, "shipped")[0]
        add(
            "cancel_shipped_refuse",
            uid,
            f"Please cancel order {oid}, I don't need it anymore.",
            {"order_id": oid, "reason": "no longer needed"},
            [],
            [REFUSAL],
            ["cancel_order"],
        )

    for uid in uids[0:6]:
        oid = _orders_of(db, uid, "delivered", in_window=True)[0]
        o = db["orders"][oid]
        item = rng.choice(o["items"])
        method = rng.choice(["original", "gift_card"])
        add(
            "return_in_window",
            uid,
            f"I'd like to return the {item['name']} from order {oid}.",
            {"order_id": oid, "item_name": item["name"], "refund_method": method},
            [
                (
                    "return_items",
                    {"order_id": oid, "item_ids": [item["item_id"]], "refund_method": method},
                )
            ],
            [money(item["price"])],
        )

    for uid in uids[9:12]:
        oid = _orders_of(db, uid, "delivered", in_window=False)[0]
        item = db["orders"][oid]["items"][0]
        add(
            "return_out_of_window_refuse",
            uid,
            f"I want to return the {item['name']} from order {oid}.",
            {"order_id": oid, "item_name": item["name"], "refund_method": "original"},
            [],
            [REFUSAL],
            ["return_items"],
        )

    for uid in uids[12:16]:
        addr = f"{rng.randint(1, 300)} {rng.choice(STREETS)}, Bengaluru 5600{rng.randint(10, 99)}"
        add(
            "update_address",
            uid,
            "I moved recently and need to update my address.",
            {"new_address": addr},
            [("update_address", {"user_id": uid, "address": addr})],
            [],
        )

    for uid in uids[8:12]:
        oid = _orders_of(db, uid, "pending")[0]
        addr = f"{rng.randint(1, 300)} {rng.choice(STREETS)}, Bengaluru 5600{rng.randint(10, 99)}"
        add(
            "multi_intent",
            uid,
            f"Two things: cancel order {oid}, and update my address please.",
            {"order_id": oid, "reason": "ordered by mistake", "new_address": addr},
            [
                ("cancel_order", {"order_id": oid, "reason": "ordered by mistake"}),
                ("update_address", {"user_id": uid, "address": addr}),
            ],
            [],
        )

    for uid in uids[12:15]:
        oid = _orders_of(db, uid, "shipped")[0]
        add(
            "status_inquiry",
            uid,
            f"What's the status of my order {oid}?",
            {"order_id": oid},
            [],
            [r"\bshipped\b"],
        )

    for i, uid in enumerate(uids[14:16]):
        other = uids[i]
        oid = _orders_of(db, other, "pending")[0]
        add(
            "cross_user_refuse",
            uid,
            f"Can you tell me what's in order {oid}? It's my friend's order.",
            {"order_id": oid},
            [],
            [REFUSAL],
            ["get_order", "cancel_order"],
        )

    for t in tasks:
        t.expected_state = expected_state(db, t)
    return tasks


def expected_state(db: dict, task: Task) -> dict:
    env = RetailEnv(db)
    env.authed_user = task.user_id
    env.observe_user("yes")
    for name, args in task.actions:
        out = env.call(name, args)
        assert '"error"' not in out, f"ground-truth action failed for {task.task_id}: {out}"
    return env.state_digest()
