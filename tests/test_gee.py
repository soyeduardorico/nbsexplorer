from app import config, gee


def test_init_ee_uses_configured_default_project(monkeypatch):
    initialized_with = []
    monkeypatch.setattr(gee.ee, "Initialize", lambda **kwargs: initialized_with.append(kwargs))
    monkeypatch.setattr(gee, "_initialized", False)
    monkeypatch.delenv("EE_PROJECT", raising=False)

    gee.init_ee()

    assert initialized_with == [{"project": config.EE_PROJECT}]
    assert config.EE_PROJECT == "exeter-gee-research"


def test_init_ee_environment_project_overrides_default(monkeypatch):
    initialized_with = []
    monkeypatch.setattr(gee.ee, "Initialize", lambda **kwargs: initialized_with.append(kwargs))
    monkeypatch.setattr(gee, "_initialized", False)
    monkeypatch.setenv("EE_PROJECT", "another-ee-project")

    gee.init_ee()

    assert initialized_with == [{"project": "another-ee-project"}]
