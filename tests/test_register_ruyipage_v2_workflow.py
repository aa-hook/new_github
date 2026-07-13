from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "register-ruyipage-v2.yml"


def load_workflow():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_dispatch_exposes_only_requested_inputs():
    workflow = load_workflow()

    assert list(workflow["on"]["workflow_dispatch"]["inputs"]) == [
        "yescaptcha_key",
        "count",
        "max_parallel",
    ]


def test_registration_step_retries_three_times_with_fixed_browser_arguments():
    workflow = load_workflow()
    steps = workflow["jobs"]["register"]["steps"]
    command = next(step["run"] for step in steps if step.get("name") == "Run RuyiPage v2 registration")

    assert "for attempt in 1 2 3" in command
    assert "--network-mode 1" in command
    assert "--click-style human" in command
    assert "--human-move-min-ms 800" in command
    assert "--human-move-max-ms 1400" in command
