"""Mixed control coverage, persistence, exports and the real Qt editor."""

from collections import Counter
from itertools import product

import pytest

from nam.capture.params import (
    KnobSpec,
    SwitchSpec,
    capture_recommendations,
    validate_knobs,
)
from nam.capture.planner import corner_capture_count, corner_settings, plan_captures
from nam.capture.project import (
    add_corner_captures,
    load_project,
    new_project,
    save_project,
)
from nam.capture.export import (
    build_concat_model_config,
    build_hyper_model_config,
    build_hyper_a2_model_config,
    build_al_model_config,
)


def controls():
    return [
        KnobSpec("Gain", 0, 10, step=0.25, is_gain=True),
        SwitchSpec("Mode", ("Clean", "Crunch", "Lead")),
        KnobSpec("Tone", 0, 10),
        SwitchSpec("Bright", ("Off", "On")),
    ]


@pytest.mark.parametrize("labels", [("Only",), ("", "On"), ("On", "on"), "On"])
def test_invalid_positions(labels):
    with pytest.raises(ValueError):
        SwitchSpec("Mode", labels)


def test_duplicate_control_names():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_knobs([KnobSpec("Mode", 0, 10), SwitchSpec("mode", ("A", "B"))])


def test_plan_balances_every_joint_state_and_keeps_validation_held_out():
    specs = controls()
    assert capture_recommendations(specs) == (180, 36, 6)
    train, val = plan_captures(specs, n_train=181, n_validation=37, seed=8)
    assert (train, val) == plan_captures(specs, n_train=181, n_validation=37, seed=8)
    for entries, expected in ((train, [30] * 5 + [31]), (val, [6] * 5 + [7])):
        states = Counter((e.params["Mode"], e.params["Bright"]) for e in entries)
        assert sorted(states.values()) == expected
        assert set(states) == set(product(("Clean", "Crunch", "Lead"), ("Off", "On")))
        assert all(list(e.params) == [c.name for c in specs] for e in entries)
    key = lambda e: tuple(e.params.values())
    assert not set(map(key, train)) & set(map(key, val))
    corners = corner_settings([c.to_param_spec() for c in specs], gain_index=0)
    assert len(corners) == corner_capture_count(specs) == 24
    assert {c["Mode"] for c in corners} == {"Clean", "Crunch", "Lead"}
    assert not {tuple(c.values()) for c in corners} & set(map(key, val))
    assert len({e.y_path for e in train + val}) == 218


def test_each_state_gets_the_full_knob_range():
    specs = [KnobSpec("Gain", 0, 10, step=0.1), SwitchSpec("Mode", ("A", "B", "C"))]
    train, _ = plan_captures(specs, n_train=45, n_validation=9)
    for state in ("A", "B", "C"):
        values = [e.params["Gain"] for e in train if e.params["Mode"] == state]
        assert min(values) < 1 and max(values) > 9


@pytest.mark.parametrize("train,val", [(5, 6), (12, 5)])
def test_insufficient_joint_state_coverage_is_rejected(train, val):
    with pytest.raises(ValueError, match="6 switch combinations"):
        plan_captures(controls(), n_train=train, n_validation=val)


def test_switches_only_capture_each_state_once():
    specs = [SwitchSpec("Mode", ("A", "B", "C")), SwitchSpec("Bright", ("Off", "On"))]
    assert capture_recommendations(specs) == (6, 0, 6)
    project = new_project(specs, n_train=6, n_validation=0)
    assert len(project.entries) == 6
    assert add_corner_captures(project) == ([], 6)
    with pytest.raises(ValueError, match="switches only"):
        plan_captures(specs, n_train=6, n_validation=6)


def test_switch_labels_defaults_and_calibration_survive_project_and_exports(tmp_path):
    specs = [KnobSpec("Gain", 0, 10), SwitchSpec("Mode", ("1", "Hi/Lo: boost", "C"), 1)]
    project = new_project(specs, n_train=9, n_validation=3)
    project.input_level_dbu = 18
    project.output_level_dbu = 12
    save_project(project, tmp_path)
    restored = load_project(tmp_path)
    assert restored.control_specs() == tuple(specs)
    assert restored.entries == project.entries
    assert any(e.params["Mode"] == "1" for e in restored.entries)
    assert all("/Lo" not in e.y_path and ":" not in e.y_path for e in restored.entries)
    for builder in (
        build_concat_model_config,
        build_hyper_model_config,
        build_hyper_a2_model_config,
        build_al_model_config,
    ):
        config = builder(restored)
        params = config["net"]["config"]["params"]
        assert params[1]["type"] == "switch"
        assert params[1]["enum_names"] == list(specs[1].enum_names)
        assert params[1]["default"] == 1


def test_gui_controls_switch_roundtrip_and_recommendations(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from nam.capture.gui import main

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(main, "_list_devices", lambda **kwargs: [])
    window = main.MainWindow()
    try:
        window.knob_table.item(0, 0).setText("Gain")
        spec = SwitchSpec("Mode", ("Off", "Normal", "Boost"))
        window._add_switch_row(spec)
        window._apply_n_train_recommendation()
        window._apply_n_validation_recommendation()
        window._refresh_corner_count()
        assert window.n_train_spin.value() == 45
        assert window.n_validation_spin.value() == 9
        assert "minimum: 3" in window.n_train_recommended_label.text()
        assert main.knob_rows_to_specs(window._knob_table_rows())[1] == spec
        assert main.format_params({"Mode": "Normal"}) == "Mode=Normal"
        project = new_project(
            main.knob_rows_to_specs(window._knob_table_rows()),
            n_train=45,
            n_validation=9,
        )
        save_project(project, tmp_path)
        window.project = load_project(tmp_path)
        window._load_knobs_into_ui()
        assert main.knob_rows_to_specs(window._knob_table_rows())[1] == spec
        assert window.n_train_spin.value() == 45
        # Deleting preceding rows must not break the edit button's row lookup.
        window.knob_table.removeRow(0)
        monkeypatch.setattr(
            window, "_edit_switch", lambda old: SwitchSpec(old.name, ("A", "B"))
        )
        window.knob_table.cellWidget(0, 7).click()
        assert main.knob_rows_to_specs(window._knob_table_rows())[0].enum_names == (
            "A",
            "B",
        )
    finally:
        window.close()


def test_switch_dialog_resizes_names_and_rejects_duplicate_positions(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QDialogButtonBox,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
    )
    from nam.capture.gui import main

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(main, "_list_devices", lambda **kwargs: [])
    window = main.MainWindow()
    warnings = []
    monkeypatch.setattr(
        main._QMessageBox, "warning", lambda *args: warnings.append(args)
    )
    observed = []

    def edit_positions():
        dialog = app.activeModalWidget()
        count = dialog.findChild(QSpinBox)
        labels = dialog.findChild(QTableWidget)
        count.setValue(3)
        observed.append([labels.item(i, 0).text() for i in range(3)])
        buttons = dialog.findChild(QDialogButtonBox)
        labels.setItem(2, 0, QTableWidgetItem("Off"))
        buttons.button(QDialogButtonBox.StandardButton.Ok).click()
        observed.append(dialog.isVisible())  # Invalid input must keep the dialog open.
        labels.item(2, 0).setText("Boost")
        buttons.button(QDialogButtonBox.StandardButton.Ok).click()

    try:
        QTimer.singleShot(0, edit_positions)
        result = window._edit_switch(SwitchSpec("Mode", ("Off", "Normal")))
        assert observed == [["Off", "Normal", "Position 3"], True]
        assert len(warnings) == 1
        assert result.enum_names == ("Off", "Normal", "Boost")
    finally:
        window.close()
