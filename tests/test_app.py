from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_and_demo_interactions(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "CHECKPOINT", tmp_path / "absent.pth")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 6
    assert any("DEMO" in warning.value for warning in app.warning)
    app.button(key="run_single").click().run()
    assert not app.exception and not app.error
    assert app.session_state.result["regions"] == []
    app.selectbox(key="sample_choice").select("B — one bright artificial region").run()
    app.button(key="run_single").click().run()
    assert not app.exception and len(app.session_state.result["regions"]) == 1
    app.selectbox(key="sample_choice").select("C — two artificial regions").run()
    app.button(key="run_single").click().run()
    assert not app.exception and len(app.session_state.result["regions"]) == 2
    app.radio(key="threshold_mode").set_value("Manual").run()
    app.slider(key="manual_threshold").set_value(1.0).run()
    app.button(key="run_single").click().run()
    assert not app.exception and app.session_state.result["regions"] == []
    app.radio(key="threshold_mode").set_value("Automatic").run()
    # Sidebar download toggle, then synthetic multi-modal toggle.
    next(c for c in app.checkbox if c.label == "Use synthetic multi-modal software test").check().run()
    next(c for c in app.checkbox if "pre-aligned slices" in c.label).check().run()
    app.button(key="run_multi").click().run()
    assert not app.exception and not app.error
    assert len(app.session_state.result["modalities"]) == 3
