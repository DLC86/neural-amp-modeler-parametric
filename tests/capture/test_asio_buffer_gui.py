import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from nam.capture.gui import main as ui
from nam.capture.audio import DeviceInfo
from nam.capture.project import CaptureProject, load_project
from nam.capture.session import CaptureSession


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("frames", [0, 32, 64, 128, 256, 512, 1024, 2048])
def test_buffer_ui_persistence_and_recorder(app, tmp_path, monkeypatch, frames):
    device = DeviceInfo(
        index=0,
        name="Test device",
        host_api="ASIO",
        max_input_channels=8,
        max_output_channels=8,
        default_samplerate=48000,
    )
    monkeypatch.setattr(ui, "_list_devices", lambda **kwargs: [device])
    monkeypatch.setattr(ui, "_current_device_sample_rates", lambda: {})
    window = ui.MainWindow()
    try:
        window.project = CaptureProject(name="Test", knobs=[])
        window.project_dir = tmp_path
        window._load_audio_settings_into_ui()
        index = window.buffer_size_combo.findData(frames)
        assert index >= 0
        window.buffer_size_combo.setCurrentIndex(index)
        window._save_audio_settings()
        assert window.project.audio.blocksize == frames
        assert load_project(tmp_path).audio.blocksize == frames

        class Recorder:
            def playrec(self, playback, sample_rate, **kwargs):
                self.kwargs = kwargs
                return np.zeros_like(playback), np.zeros_like(playback)

        recorder = Recorder()
        session = CaptureSession(window.project, tmp_path, recorder=recorder)
        monkeypatch.setattr(
            session, "_resolve_devices", lambda: dict(input_device=0, output_device=0)
        )
        session._playrec(
            np.zeros(128, dtype=np.float32),
            48000,
            None,
            None,
            np.zeros(128, dtype=np.float32),
        )
        assert recorder.kwargs["blocksize"] == frames
        window.project = load_project(tmp_path)
        window._load_audio_settings_into_ui()
        assert window.buffer_size_combo.currentData() == frames
    finally:
        window.close()
