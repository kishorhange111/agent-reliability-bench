import json

import pytest

from arbench.cli import main


def test_cli_sweep(tmp_path):
    main(
        [
            "--agents",
            "oracle",
            "rule",
            "--faults",
            "0",
            "0.3",
            "--trials",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert [(r["agent"], r["fault_rate"]) for r in rows] == [
        ("oracle", 0.0),
        ("oracle", 0.3),
        ("rule", 0.0),
        ("rule", 0.3),
    ]
    assert rows[0]["pass^1"] == 1.0 and "pass^2" in rows[0]
    assert (tmp_path / "results.md").read_text().startswith("| agent |")


def test_unknown_agent():
    with pytest.raises(SystemExit):
        main(["--agents", "gpt9", "--trials", "1"])
