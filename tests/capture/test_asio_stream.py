import threading
from types import SimpleNamespace

import numpy as np
import pytest

from nam.capture._asio_stream import AsioRecorder, _BlockJob, check_timing, transfer
from nam.capture.audio import AudioDeviceError, SounddeviceRecorder


@pytest.fixture
def signal():
    return np.random.default_rng(4).normal(0, 0.04, 4 * 48000).astype(np.float32)


def test_stable_loopback_is_accepted_despite_level_changes(signal):
    rate, delay = 48000, 450
    recorded = np.concatenate((np.zeros(delay), signal))[: len(signal)]
    reference = transfer(signal[:rate], recorded[delay : delay + rate], rate)
    recorded[2 * rate :] *= 0.7
    assert check_timing(signal, recorded, rate, delay, reference) >= 3


@pytest.mark.parametrize("jump", [7, 8, -7])
def test_mid_capture_timing_change_is_refused(signal, jump):
    rate, delay = 48000, 450
    original = np.concatenate((np.zeros(delay), signal))[: len(signal)]
    reference = transfer(signal[:rate], original[delay : delay + rate], rate)
    recorded = original.copy()
    start = 2 * rate
    indices = np.arange(start, len(signal)) + jump
    recorded[start:] = original[np.clip(indices, 0, len(signal) - 1)]
    with pytest.raises(AudioDeviceError, match="timing changed"):
        check_timing(signal, recorded, rate, delay, reference)


def test_missing_loopback_is_not_silently_accepted(signal):
    rate = 48000
    reference = transfer(signal[:rate], signal[:rate], rate)
    with pytest.raises(AudioDeviceError, match="No reliable timing"):
        check_timing(signal, np.zeros_like(signal), rate, 0, reference)


def test_changed_loopback_filter_is_refused(signal):
    rate = 48000
    reference = transfer(signal[:rate], signal[:rate], rate)
    with pytest.raises(AudioDeviceError, match="response changed"):
        check_timing(signal, -signal, rate, 0, reference)


def engine():
    obj = AsioRecorder.__new__(AsioRecorder)
    obj.stream = None
    obj.config = None
    obj.reference = None
    obj.last_status = None
    obj.actual_blocksize = None
    obj.job = None
    return obj


def settings():
    return dict(
        sample_rate=48000,
        input_device=0,
        output_device=0,
        input_channel=1,
        output_channel=7,
        loopback_input_channel=2,
        loopback_output_channel=8,
        blocksize=2048,
    )


def test_requested_native_buffer_is_verified_before_playback():
    obj = engine()
    obj.settings = settings()
    obj.job = _BlockJob(np.ones(100), np.ones(100), np.zeros(100), np.zeros(100))
    job = obj.job
    output = np.ones((64, 8))
    obj._callback(
        np.zeros((64, 8)),
        output,
        64,
        None,
        SimpleNamespace(input_overflow=False, output_underflow=False),
    )
    assert isinstance(job.error, AudioDeviceError)
    assert "driver uses 64" in str(job.error)
    assert job.done.is_set()
    assert not output.any()


def test_cancelled_job_is_acknowledged_without_playing_audio():
    obj = engine()
    obj.settings = settings()
    job = _BlockJob(np.ones(100), None, np.zeros(100), None, cancelled=True)
    obj.job = job
    output = np.ones((64, 8))
    obj._callback(
        np.zeros((64, 8)),
        output,
        64,
        None,
        SimpleNamespace(input_overflow=False, output_underflow=False),
    )
    assert job.done.is_set()
    assert obj.job is None
    assert not output.any()


def test_stream_is_reused_and_reopened_only_for_changed_settings(monkeypatch):
    import sounddevice as sd

    streams = []

    class Stream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active = False
            self.closed = False
            streams.append(self)

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(sd, "Stream", Stream)
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda *a: dict(max_input_channels=8, max_output_channels=8),
    )
    obj = engine()
    obj._open(settings())
    obj._open(settings())
    assert len(streams) == 1
    assert streams[0].kwargs["blocksize"] == 0
    assert streams[0].kwargs["latency"] == 2048 / 48000
    changed = dict(settings(), blocksize=64)
    obj._open(changed)
    assert len(streams) == 2
    assert streams[0].closed
    obj._close_stream()
    assert streams[1].closed


def test_owner_closes_on_its_own_thread(monkeypatch):
    import nam.capture._asio_stream as module
    from contextlib import nullcontext

    monkeypatch.setattr(module, "asio_com_apartment", nullcontext)
    threads = []
    monkeypatch.setattr(
        AsioRecorder, "_capture", lambda self, **kw: threading.get_ident()
    )
    monkeypatch.setattr(
        AsioRecorder,
        "_close_stream",
        lambda self: threads.append(threading.get_ident()),
    )
    obj = AsioRecorder()
    owner = obj.playrec()
    assert obj.playrec() == owner
    obj.close()
    assert threads == [owner]
    assert not obj.thread.is_alive()


def test_public_recorder_closes_persistent_owner():
    recorder = SounddeviceRecorder()
    closed = []
    recorder._asio = SimpleNamespace(close=lambda: closed.append(True))
    recorder.close()
    recorder.close()
    assert closed == [True]


def test_initially_stable_measurements_do_not_skip_startup_observation(
    monkeypatch, signal
):
    import nam.capture._asio_stream as module

    clock = [0.0]
    count = [0]
    reference = transfer(signal[:48000], signal[:48000], 48000)
    obj = engine()
    obj.settings = settings()
    obj.opened = 0.0
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def probe(cancel):
        clock[0] += 2.5
        count[0] += 1
        return (450 if count[0] <= 5 else 443), reference

    monkeypatch.setattr(obj, "_probe", probe)
    obj._stabilize(None)
    assert count[0] == 9
    assert obj.reference[0] == 443


def test_probe_accepts_clean_fractional_filtered_response(monkeypatch):
    from scipy.signal import fftconvolve
    from nam.capture.resample import fractional_delay_kernel

    obj = engine()
    obj.settings = settings()

    def exchange(main, loop, cancel):
        filtered = fftconvolve(loop, fractional_delay_kernel(0.5), mode="same")
        return main, np.concatenate((np.zeros(450), filtered))[: len(loop)]

    monkeypatch.setattr(obj, "_exchange", exchange)
    delay, reference = obj._probe(None)
    assert delay in (450, 451)
    assert np.median(reference[3][10:300]) > 0.99


def test_program_timing_check_can_be_cancelled(signal):
    from nam.capture.audio import CaptureCancelled

    reference = transfer(signal[:48000], signal[:48000], 48000)
    with pytest.raises(CaptureCancelled):
        check_timing(signal, signal, 48000, 0, reference, cancel=lambda: True)


def test_audit_does_not_report_training_safe_for_an_unstable_stream():
    from nam.capture.session import CaptureAudit, audit_problems

    audit = CaptureAudit(
        y_path="capture.wav",
        offset=3.0,
        stream_timing_problem="Loopback timing changed by -7 samples",
    )
    problems = audit_problems([audit])
    assert any("timing changed" in message for message in problems)
    assert not any(
        "Training on them as they are is fine" in message for message in problems
    )
