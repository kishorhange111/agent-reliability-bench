import pytest

from arbench.agents import OracleAgent, RuleAgent
from arbench.env import FaultConfig
from arbench.runner import Action, pass_hat_k, run_episode, summarize
from arbench.tasks import build_db, build_tasks


@pytest.fixture(scope="module")
def suite():
    db = build_db()
    return db, build_tasks(db)


def test_pass_hat_k():
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(4, 3, 4) == 0.0
    assert pass_hat_k(4, 2, 1) == pytest.approx(0.5)
    assert pass_hat_k(4, 3, 2) == pytest.approx(3 / 6)
    with pytest.raises(ValueError):
        pass_hat_k(2, 1, 3)


def test_oracle_solves_every_task(suite):
    db, tasks = suite
    res = [run_episode(OracleAgent(), t, db) for t in tasks]
    assert all(r.success for r in res), [r.task_id for r in res if not r.success]


def test_oracle_with_retries_is_robust_to_faults(suite):
    db, tasks = suite
    res = [run_episode(OracleAgent(), t, db, faults=FaultConfig(0.3)) for t in tasks]
    assert sum(r.transient_faults for r in res) > 0
    assert all(r.success for r in res)


def test_no_retry_oracle_degrades_and_pass_hat_k_decays(suite):
    db, tasks = suite
    ag = OracleAgent(retry=False)
    res = [
        run_episode(ag, t, db, trial=i, faults=FaultConfig(0.2)) for t in tasks for i in range(4)
    ]
    s = summarize(res, ks=(1, 4))
    assert s["pass^4"] < s["pass^1"] < 1.0


def test_rule_agent_baseline_and_failure_taxonomy(suite):
    db, tasks = suite
    res = [run_episode(RuleAgent(), t, db) for t in tasks]
    s = summarize(res, ks=(1,))
    assert s["by_category"]["cancel_pending"] == 1.0
    assert s["by_category"]["cross_user_refuse"] == 0.0  # naive agent looks up a friend's order
    assert s["failure_modes"] == {"cross_user_access": 2}


class LazyAgent:
    name = "lazy"

    def reset(self, tools, seed):
        pass

    def act(self, history):
        return Action("end")


def test_doing_nothing_fails_every_task(suite):
    db, tasks = suite
    res = [run_episode(LazyAgent(), t, db) for t in tasks]
    assert not any(r.success for r in res)
