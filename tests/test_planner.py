from copy import deepcopy
import json
import os

import pytest
from pydantic import ValidationError

from app import planner
from app.catalogue import LAYER_CATALOGUE, OPERATION_CATALOGUE


@pytest.fixture
def draft():
    return {
        "title": "Slope planting screening",
        "summary": "A provisional recipe requiring local assessment.",
        "criteria": [
            {
                "layer": "slope", "operation": "keep_range",
                "lower": None, "upper": 45.0,
                "lower_inclusive": True, "upper_inclusive": False,
                "reason": "Illustrative screening limit, not a safety limit.",
                "basis": "provisional_assumption",
            },
            {
                "layer": "land_cover", "operation": "exclude_classes",
                "classes": ["built_up"], "reason": "Avoid incompatible land use.",
                "basis": "provisional_assumption",
            },
        ],
        "assumptions": ["Slope limit needs local validation."],
        "questions": ["What is the location and hemisphere?"],
        "limitations": ["Missing soil mechanics, drainage and measured light."],
    }


@pytest.fixture
def gemini(draft, monkeypatch):
    state = {"draft": draft, "requests": [], "finish_reason": "STOP", "status": 200}

    class Connection:
        def __init__(self, host, timeout):
            assert host == "generativelanguage.googleapis.com"
            assert timeout == 120
            self.status = state["status"]

        def request(self, method, path, body, headers):
            assert method == "POST"
            assert path == "/v1beta/models/gemini-3.1-flash-lite:generateContent"
            assert headers["x-goog-api-key"] == "test-key"
            state["requests"].append(json.loads(body))

        def getresponse(self):
            return self

        def read(self):
            return json.dumps({"candidates": [{
                "finishReason": state["finish_reason"],
                "content": {"parts": [{"text": state.get("content", json.dumps(state["draft"]))}]},
            }]}).encode()

        def close(self):
            state["closed"] = True

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setattr(planner, "HTTPSConnection", Connection)
    return state


def test_catalogue_matches_builders():
    from app.gee import LAYERS

    assert LAYER_CATALOGUE.keys() == LAYERS.keys()


def test_local_env_file_loads_and_does_not_override_process_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=local-key\nBRAVE_SEARCH_API_KEY=brave-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    planner.config.load_local_env(env_file)

    assert os.environ["GEMINI_API_KEY"] == "local-key"
    assert os.environ["BRAVE_SEARCH_API_KEY"] == "brave-key"
    monkeypatch.setenv("GEMINI_API_KEY", "process-key")
    planner.config.load_local_env(env_file)
    assert os.environ["GEMINI_API_KEY"] == "process-key"


def test_valid_plan_and_gemini_request(gemini):
    plan, model = planner.generate_plan("Cold slope, little light")
    assert model == "gemini-3.1-flash-lite"
    assert plan.criteria[0].upper == 45
    payload = gemini["requests"][0]
    context = json.loads(payload["contents"][0]["parts"][0]["text"])
    assert context["layers"].keys() == LAYER_CATALOGUE.keys()
    assert context["operations"] == OPERATION_CATALOGUE
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    items = payload["generationConfig"]["responseJsonSchema"]["properties"]["criteria"]["items"]
    assert len(items["anyOf"]) == 4
    assert "discriminator" not in items
    definitions = payload["generationConfig"]["responseJsonSchema"]["$defs"]
    assert definitions["KeepRange"]["properties"]["operation"]["enum"] == ["keep_range"]
    prompt = payload["systemInstruction"]["parts"][0]["text"]
    for text in ("hemisphere", "tolerance in (0, 180)", "keep_direction only with",
                 "NDWI is a spectral water", "score_linear is monotonic",
                 "full physical domain", "not a user_requirement"):
        assert text in prompt
    assert gemini["closed"]


def test_issue_conversation_keeps_prior_turns_and_returns_reply(gemini):
    gemini["content"] = "What location and habitat constraints should we consider?"
    reply = planner.reply_to_issue_conversation([
        {"role": "user", "content": "I want to explore woodland restoration."},
        {"role": "assistant", "content": "What is the main goal?"},
        {"role": "user", "content": "Improve connectivity between habitats."},
    ])
    assert reply == gemini["content"]
    payload = gemini["requests"][0]
    contents = payload["contents"]
    assert [turn["role"] for turn in contents] == ["user", "model", "user"]
    assert contents[0]["parts"][0]["text"] == "I want to explore woodland restoration."
    assert "earlier messages as context" in payload["systemInstruction"]["parts"][0]["text"]
    assert payload["generationConfig"]["maxOutputTokens"] == 1024


def test_issue_conversation_summary_uses_all_turns_and_summary_instruction(gemini):
    gemini["content"] = "Explore woodland restoration to improve habitat connectivity."
    summary = planner.reply_to_issue_conversation([
        {"role": "user", "content": "I want to explore woodland restoration."},
        {"role": "assistant", "content": "What is your goal?"},
        {"role": "user", "content": "Improve habitat connectivity."},
    ], summarize=True)
    assert summary == gemini["content"]
    payload = gemini["requests"][0]
    assert payload["contents"][-1]["role"] == "user"
    assert "Summarize our discussion" in payload["contents"][-1]["parts"][0]["text"]
    assert "standalone site brief" in payload["systemInstruction"]["parts"][0]["text"]


@pytest.mark.parametrize("updates", [
    {"layer": "invented"},
    {"operation": "execute_python"},
    {"layer": "aspect"},
    {"lower": None, "upper": None},
    {"lower": 45, "upper": 40},
    {"lower": 0, "upper": 90},
    {"upper": 100},
    {"upper": float("nan")},
    {"upper": float("inf")},
    {"upper": "45"},
    {"extra_code": "print('unsafe')"},
])
def test_reject_invalid_range(draft, updates):
    draft["criteria"][0].update(updates)
    with pytest.raises(ValidationError):
        planner.Plan.model_validate(draft)


@pytest.mark.parametrize("classes", [["invented"], [], ["built_up", "unknown"]])
def test_reject_invalid_classes(draft, classes):
    draft["criteria"][1]["classes"] = classes
    with pytest.raises(ValidationError):
        planner.Plan.model_validate(draft)


def test_direction_and_score_validation(draft):
    common = {"reason": "Hypothesis to review", "basis": "provisional_assumption"}
    direction = dict(common, layer="aspect", operation="keep_direction", centre=0, tolerance=20)
    score = dict(common, layer="ndvi", operation="score_linear",
                 lower=-0.5, upper=0.8, preference="increasing", weight=1.0)
    draft["criteria"] = [direction, score]
    assert len(planner.Plan.model_validate(draft).criteria) == 2
    for field, value in [("centre", 360), ("tolerance", 0), ("tolerance", 180)]:
        bad = deepcopy(draft)
        bad["criteria"][0][field] = value
        with pytest.raises(ValidationError):
            planner.Plan.model_validate(bad)
    for field, value in [("weight", 0), ("upper", -0.9), ("lower", -2)]:
        bad = deepcopy(draft)
        bad["criteria"][1][field] = value
        with pytest.raises(ValidationError):
            planner.Plan.model_validate(bad)


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set GEMINI_API_KEY in the environment before using Gemini"):
        planner.generate_plan("slope")


def test_issue_conversation_requires_key_before_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set GEMINI_API_KEY in the environment before using Gemini"):
        planner.reply_to_issue_conversation([
            {"role": "user", "content": "Explore woodland restoration."},
        ])


def test_explicit_key_is_used_when_environment_is_unset(gemini, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    planner.generate_plan("slope", api_key="test-key")
    assert len(gemini["requests"]) == 1


@pytest.mark.parametrize("timeout_stage", ["request", "getresponse", "read"])
def test_timeout_identifies_gemini_and_closes_connection(monkeypatch, timeout_stage):
    class Connection:
        closed = False

        def request(self, *args, **kwargs):
            if timeout_stage == "request":
                raise TimeoutError("timed out")

        def getresponse(self):
            if timeout_stage == "getresponse":
                raise TimeoutError("timed out")
            return self

        def read(self):
            raise TimeoutError("timed out")

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(planner, "HTTPSConnection", lambda *args, **kwargs: connection)
    with pytest.raises(ValueError, match="Gemini request timed out") as error:
        planner.gemini_json("gemini-3.1-flash-lite", {}, "test-key", 120)
    assert "--timeout 600" in str(error.value)
    assert "not from Earth Engine" in str(error.value)
    assert isinstance(error.value.__cause__, TimeoutError)
    assert connection.closed


@pytest.mark.parametrize("failure", ["invalid_plan", "truncated", "http"])
def test_server_failures_are_explicit(gemini, failure):
    if failure == "invalid_plan":
        gemini["draft"]["criteria"][0]["layer"] = "invented"
    elif failure == "truncated":
        gemini["finish_reason"] = "MAX_TOKENS"
    else:
        gemini["status"] = 500
    with pytest.raises(ValueError):
        planner.generate_plan("slope")


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_model_plan_is_reported_explicitly(gemini, content):
    gemini["content"] = content
    with pytest.raises(ValueError, match="Gemini returned an empty plan"):
        planner.generate_plan("slope")


def test_reconcile_provisional_threshold_with_model_assumptions(draft):
    draft["criteria"][0]["basis"] = "user_requirement"
    draft["assumptions"] = ["The 45 degree slope threshold is a provisional assumption."]
    plan = planner._reconcile_provisional_assumptions(planner.Plan.model_validate(draft))
    assert plan.criteria[0].basis == "provisional_assumption"
    assert any("treated as provisional" in item for item in plan.assumptions)


@pytest.mark.parametrize("answer", ["n", "", "yes", "y"])
def test_cli_approval_controls_persistence(gemini, tmp_path, monkeypatch, answer):
    output = tmp_path / "plan.json"
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert planner.main([
        "Cold slope", "--output", str(output),
    ]) == 0
    assert output.exists() == (answer in {"y", "yes"})
    if output.exists():
        record = json.loads(output.read_text())
        assert record["status"] == "approved_draft_for_review"
        assert record["brief"] == "Cold slope"
        assert record["plan"]["questions"]
        assert record["catalogues"]["layers"]["land_cover"]["classes"]["built_up"] == 50


def test_invalid_plan_never_requests_approval(gemini, tmp_path, monkeypatch):
    gemini["draft"]["criteria"][0]["layer"] = "invented"

    def unexpected_input(_):
        pytest.fail("An invalid plan must not reach approval")

    monkeypatch.setattr("builtins.input", unexpected_input)
    output = tmp_path / "plan.json"
    assert planner.main(["Slope", "--output", str(output)]) == 1
    assert not output.exists()


def test_catalogue_listing_is_offline(capsys):
    assert planner.main(["--list-catalogues"]) == 0
    assert json.loads(capsys.readouterr().out)["operations"] == OPERATION_CATALOGUE


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "plan.json"
    output.write_text("existing")
    assert planner.main(["Slope", "--output", str(output)]) == 1
    assert output.read_text() == "existing"


@pytest.mark.parametrize("brief,timeout", [("", 1), ("Slope", 0), ("Slope", float("nan"))])
def test_invalid_inputs_fail_before_network(brief, timeout):
    with pytest.raises(ValueError):
        planner.generate_plan(brief, timeout=timeout)
