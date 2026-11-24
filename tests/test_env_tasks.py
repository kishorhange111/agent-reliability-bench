import json

import pytest

from arbench.env import FaultConfig, RetailEnv
from arbench.tasks import build_db, build_tasks
from arbench.user_sim import END, ScriptedUser


@pytest.fixture(scope="module")
def db():
    return build_db()


def first_order(db, uid, status):
    return next(o for o in db["users"][uid]["orders"] if db["orders"][o]["status"] == status)


def test_db_is_deterministic():
    assert build_db(7) == build_db(7)
    assert build_db(7) != build_db(8)


def test_access_before_auth_is_a_violation(db):
    env = RetailEnv(db)
    out = json.loads(env.call("get_order", {"order_id": first_order(db, "u00", "pending")}))
    assert "error" in out and env.violations[0].kind == "auth_required"


def test_cross_user_access_is_a_violation(db):
    env = RetailEnv(db)
    env.call("find_user_by_email", {"email": db["users"]["u01"]["email"]})
    env.call("get_order", {"order_id": first_order(db, "u00", "pending")})
    assert env.violations[-1].kind == "cross_user_access"


def test_mutation_requires_confirmation(db):
    env = RetailEnv(db)
    env.call("find_user_by_email", {"email": db["users"]["u00"]["email"]})
    oid = first_order(db, "u00", "pending")
    env.observe_user("cancel it")
    out = json.loads(env.call("cancel_order", {"order_id": oid, "reason": "no longer needed"}))
    assert "error" in out and env.violations[-1].kind == "unconfirmed_mutation"
    assert env.db["orders"][oid]["status"] == "pending"
    env.observe_user("Yes, please go ahead.")
    out = json.loads(env.call("cancel_order", {"order_id": oid, "reason": "no longer needed"}))
    assert out["status"] == "cancelled"


def test_business_rules_refuse_without_violation(db):
    env = RetailEnv(db)
    env.call("find_user_by_email", {"email": db["users"]["u00"]["email"]})
    env.observe_user("yes")
    shipped = first_order(db, "u00", "shipped")
    out = json.loads(env.call("cancel_order", {"order_id": shipped, "reason": "no longer needed"}))
    assert "only pending" in out["error"] and not env.violations


def test_transient_faults_have_no_side_effects(db):
    env = RetailEnv(db, FaultConfig(p_transient=1.0))
    out = json.loads(env.call("find_user_by_email", {"email": db["users"]["u00"]["email"]}))
    assert "503" in out["error"] and env.authed_user is None and env.transient_faults == 1


def test_env_does_not_mutate_source_db(db):
    before = json.dumps(db, sort_keys=True)
    env = RetailEnv(db)
    env.call("find_user_by_email", {"email": db["users"]["u00"]["email"]})
    env.observe_user("yes")
    env.call(
        "cancel_order",
        {"order_id": first_order(db, "u00", "pending"), "reason": "no longer needed"},
    )
    assert json.dumps(db, sort_keys=True) == before


def test_task_suite_is_solvable_and_balanced(db):
    tasks = build_tasks(db)
    assert len(tasks) == 31
    assert len({t.task_id for t in tasks}) == len(tasks)
    cats = {t.category for t in tasks}
    assert {"multi_intent", "cross_user_refuse", "return_in_window"} <= cats
    assert all(t.expected_state is not None for t in tasks)
    assert {t.auth for t in tasks} == {"email", "name_zip"}


def test_user_sim_reveals_only_what_is_asked(db):
    t = next(t for t in build_tasks(db) if t.auth == "name_zip")
    u = ScriptedUser(t)
    reply = u.reply("What is your email?")
    assert t.facts["name"] in reply and "@" not in reply
    assert "Yes" in u.reply("Shall I proceed?")
    assert END in u.reply("Is there anything else?")
