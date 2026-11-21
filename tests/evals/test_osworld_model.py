"""evals/osworld.py --model (KAN-184 follow-up): the environment of a Claude or GPT run, with no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evals import osworld


def test_a_gemini_run_does_not_inherit_the_runners_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    env = osworld.operon_env("http://vm:5000", "/runs")
    assert env["OPERON_DESKTOP_BACKEND"] == "computer_use" and env["OPERON_MODEL_PROFILE"] == ""
    assert env["OPERON_ENVIRONMENT"] == "osworld" and env["OPERON_DESKTOP_PLATFORM"] == "linux"


@pytest.mark.parametrize("profile", ["claude-sonnet-5-5", "openai-computer-use"])
def test_a_model_run_selects_the_model_backend(profile) -> None:
    env = osworld.operon_env("http://vm:5000", "/runs", profile)
    assert env["OPERON_DESKTOP_BACKEND"] == "model" and env["OPERON_MODEL_PROFILE"] == profile
    assert env["OPERON_OSWORLD_URL"] == "http://vm:5000" and env["OPERON_TOOLS"] == "off"


def test_the_allow_list_names_real_registry_profiles() -> None:
    path = Path(osworld.__file__).parents[1] / "src/operon/agent/backends/model_adapter/models.yaml"
    assert set(osworld.MODEL_LABELS) <= set(yaml.safe_load(path.read_text("utf-8")))


def test_main_passes_the_model_and_records_it(tmp_path, monkeypatch) -> None:
    seen = {}

    class FakeEnv:
        vm_ip, server_port = "1.2.3.4", 5000

        def reset(self, task_config): ...

        def evaluate(self):
            return 1.0

    def fake_run(cmd, env, **kw):
        seen["env"] = env

        class Done:
            returncode, stdout, stderr = 0, "", ""

        return Done()

    task = {"domain": "chrome", "id": "abcd1234", "instruction": "x"}
    monkeypatch.setattr(osworld, "load_subset", lambda *a: [task])
    monkeypatch.setattr(osworld, "task_config", lambda repo, t: {})
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld.time, "sleep", lambda s: None)
    code = osworld.main(["--osworld-repo", str(tmp_path), "--model", "claude-sonnet-5-5", "--out", str(tmp_path / "out")],
                        env_factory=lambda repo: FakeEnv(), run=fake_run)
    assert code == 0
    assert seen["env"]["OPERON_MODEL_PROFILE"] == "claude-sonnet-5-5"
    report = json.loads(next((tmp_path / "out").glob("osworld-*.json")).read_text("utf-8"))
    assert report["model"] == "claude-sonnet-5-5" and report["label"] == "Claude Sonnet 5.5 · OSWorld"


def test_main_refuses_an_unknown_model(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        osworld.main(["--osworld-repo", str(tmp_path), "--model", "claude-opus-5-5"], env_factory=lambda r: None)
    assert "--model must be one of" in capsys.readouterr().err
