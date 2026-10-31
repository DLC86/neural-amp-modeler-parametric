"""A single COM-owning audio thread and a verified timebase for ASIO captures."""

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .audio import AudioDeviceError, CaptureCancelled, asio_com_apartment


class _UnstableReference(AudioDeviceError):
    pass


def transfer(x, y, rate):
    from scipy.signal import coherence, csd, welch

    f, power = welch(x, fs=rate, nperseg=2048)
    _, cross = csd(x, y, fs=rate, nperseg=2048)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, coherent = coherence(x, y, fs=rate, nperseg=2048)
    return f, cross / np.maximum(power, 1e-30), power, coherent


def relative_delay(current, reference, rate):
    """Compare only well-excited, coherent frequencies on the clean loopback."""
    f, response, power, coherent = current
    mask = (
        (coherent > 0.98)
        & (reference[3] > 0.98)
        & (power > max(power) * 1e-5)
        & (reference[2] > max(reference[2]) * 1e-5)
        & (f > 100)
        & (f < min(8000, rate * 0.45))
    )
    if np.count_nonzero(mask) < 12:
        return None
    ratio = response[mask] / reference[1][mask]
    ratio /= np.maximum(abs(ratio), 1e-30)
    shifts = np.arange(-256, 257) / 8
    errors = np.mean(
        abs(
            ratio[None, :]
            * np.exp(2j * np.pi * f[mask][None, :] * shifts[:, None] / rate)
            - 1
        )
        ** 2,
        axis=1,
    )
    i = int(np.argmin(errors))
    if errors[i] >= 0.01:
        raise _UnstableReference(
            "The clean loopback response changed and no longer provides a consistent timing reference. This capture was not saved."
        )
    return float(shifts[i])


def check_timing(
    playback, recorded, rate, delay, reference, require_measurable=True, cancel=None
):
    """Reject an inconsistent recording before it can be labelled or exported."""
    end = min(len(playback), len(recorded) - delay)
    measured = 0
    for start in range(0, end - rate + 1, rate):
        if cancel is not None and cancel():
            raise CaptureCancelled()
        x = playback[start : start + rate]
        y = recorded[start + delay : start + delay + rate]
        if np.mean(x.astype(np.float64) ** 2) < 1e-10:
            continue
        shift = relative_delay(transfer(x, y, rate), reference, rate)
        if shift is None:
            continue
        measured += 1
        if abs(shift) > 0.25:
            raise AudioDeviceError(
                f"Clean loopback timing changed by {shift:+.3f} samples near "
                f"{start / rate:.1f} seconds. This capture was not saved. "
                "Keep the audio routing unchanged and retry; the stream will be recalibrated."
            )
    if require_measurable and end >= rate and not measured:
        raise AudioDeviceError(
            "No reliable timing windows were found on the clean loopback. This capture was not saved."
        )
    return measured


@dataclass
class _BlockJob:
    main: np.ndarray
    loop: object
    recorded: np.ndarray
    loop_recorded: object
    position: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    error: object = None
    cancelled: bool = False


class AsioRecorder:
    """All stream operations stay on one long-lived COM thread, across GUI workers."""

    def __init__(self):
        self.commands = queue.Queue()
        self.thread = threading.Thread(target=self._serve, name="NAM ASIO", daemon=True)
        self.stream = None
        self.config = None
        self.job = None
        self.reference = None
        self.actual_blocksize = None
        self.last_status = None
        self.thread.start()

    def playrec(self, **settings):
        done = threading.Event()
        result = []
        self.commands.put((settings, done, result))
        while not done.wait(0.1):
            if not self.thread.is_alive():
                raise AudioDeviceError("The ASIO audio thread stopped unexpectedly.")
        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]

    def close(self):
        self.commands.put(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise AudioDeviceError(
                "The ASIO stream did not close; do not refresh the device yet."
            )

    def _serve(self):
        with asio_com_apartment():
            try:
                while True:
                    command = self.commands.get()
                    if command is None:
                        return
                    settings, done, result = command
                    try:
                        result.append(self._capture(**settings))
                    except BaseException as exc:
                        self.reference = None
                        result.append(exc)
                    finally:
                        done.set()
                        # Do not retain a completed recording or a retired GUI
                        # worker's bound callbacks while waiting for the next job.
                        del command, settings, done, result
            finally:
                self._close_stream()

    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
        self.config = None
        self.reference = None

    def _open(self, settings):
        import sounddevice as sd

        # Import DSP code before starting the real-time callback. First imports can
        # hold the interpreter long enough to starve a small ASIO buffer.
        from scipy import signal  # noqa: F401

        keys = (
            "sample_rate",
            "input_device",
            "output_device",
            "input_channel",
            "output_channel",
            "loopback_input_channel",
            "loopback_output_channel",
            "blocksize",
        )
        config = tuple(settings[k] for k in keys)
        if self.stream is not None and self.config == config and self.stream.active:
            return
        self._close_stream()
        rate = settings["sample_rate"]
        requested = settings["blocksize"]
        ins = int(sd.query_devices(settings["input_device"])["max_input_channels"])
        outs = int(sd.query_devices(settings["output_device"])["max_output_channels"])
        for name, maximum in (
            ("input_channel", ins),
            ("loopback_input_channel", ins),
            ("output_channel", outs),
            ("loopback_output_channel", outs),
        ):
            value = settings[name]
            if value is not None and not 1 <= value <= maximum:
                raise AudioDeviceError(f"{name} is outside the device's channel range.")
        self.settings = {key: settings[key] for key in keys}
        self.actual_blocksize = None
        self.last_status = None
        # With ASIO and no software block adapter, callback frames expose the native
        # host buffer. The explicit latency requests the selected native size, even
        # if PortAudio's cached default latency predates a control-panel change.
        self.stream = sd.Stream(
            samplerate=rate,
            device=(settings["input_device"], settings["output_device"]),
            channels=(ins, outs),
            dtype="float32",
            blocksize=0,
            latency=requested / rate if requested else "low",
            callback=self._callback,
        )
        self.config = config
        self.opened = time.monotonic()
        self.stream.start()

    def _callback(self, incoming, outgoing, frames, timing, status):
        outgoing.fill(0)
        self.actual_blocksize = frames
        job = self.job
        if status.input_overflow or status.output_underflow:
            self.last_status = str(status)
        if job is None:
            return
        if job.cancelled:
            self.job = None
            job.done.set()
            return
        try:
            requested = self.settings["blocksize"]
            if requested and frames != requested:
                raise AudioDeviceError(
                    f"ASIO requested {requested} samples but the driver uses {frames}. Choose a supported native buffer size."
                )
            if self.last_status:
                raise AudioDeviceError(
                    f"ASIO lost audio: {self.last_status}. This capture was not saved."
                )
            start = job.position
            n = min(frames, len(job.main) - start)
            stop = start + n
            outgoing[:n, self.settings["output_channel"] - 1] = job.main[start:stop]
            job.recorded[start:stop] = incoming[:n, self.settings["input_channel"] - 1]
            if job.loop is not None:
                outgoing[:n, self.settings["loopback_output_channel"] - 1] = job.loop[
                    start:stop
                ]
                job.loop_recorded[start:stop] = incoming[
                    :n, self.settings["loopback_input_channel"] - 1
                ]
            job.position = stop
            if stop == len(job.main):
                self.job = None
                job.done.set()
        except BaseException as exc:
            outgoing.fill(0)
            job.error = exc
            self.job = None
            job.done.set()

    def _exchange(self, main, loop, cancel, progress=None):
        job = _BlockJob(
            main,
            loop,
            np.zeros_like(main),
            None if loop is None else np.zeros_like(loop),
        )
        self.job = job
        while not job.done.wait(0.05):
            if cancel is not None and cancel():
                job.cancelled = True
                while self.stream.active and not job.done.wait(0.05):
                    pass
                self.job = None
                raise CaptureCancelled()
            if not self.stream.active:
                self.job = None
                raise AudioDeviceError("ASIO stopped before recording completed.")
            if progress is not None:
                progress(job.position / len(main))
        if job.error is not None:
            raise job.error
        return job.recorded, job.loop_recorded

    def _probe(self, cancel):
        from scipy.signal import correlate

        rate = self.settings["sample_rate"]
        rng = np.random.default_rng(917)
        pilot = np.concatenate(
            (rng.normal(0, 0.04, 2 * rate), np.zeros(rate // 2))
        ).astype(np.float32)
        _, returned = self._exchange(np.zeros_like(pilot), pilot, cancel)
        corr = correlate(returned, pilot, method="fft")
        lag = int(np.argmax(abs(corr))) - len(pilot) + 1
        if not 0 <= lag < rate // 2:
            raise _UnstableReference(
                "The clean ASIO loopback was not detected. Check its channels and cable."
            )
        x = pilot[: 2 * rate]
        y = returned[lag : lag + len(x)]
        reference = transfer(x, y, rate)
        band = (reference[0] > 100) & (reference[0] < min(8000, rate * 0.45))
        # Converter filters and fractional delay can lower waveform correlation
        # substantially even on a clean path. Spectral coherence tests linearity
        # without assuming an impulse-shaped hardware response.
        if (
            np.mean(y.astype(np.float64) ** 2) < 1e-10
            or np.count_nonzero(reference[3][band] > 0.98) < np.count_nonzero(band) // 2
        ):
            raise _UnstableReference(
                "The ASIO loopback is not a reliable clean reference. Check routing and levels."
            )
        return lag, reference

    def _stabilize(self, cancel):
        # A short initially stable interval is insufficient: the reproduced FM9
        # transition happened after ~14 s. This observation floor is paired with
        # measured stability and per-capture verification, not treated as a guarantee.
        stable = 0
        previous = None
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            try:
                lag, current = self._probe(cancel)
            except _UnstableReference:
                previous = None
                stable = 0
                continue
            if previous is not None:
                old_lag, old = previous
                try:
                    residual = relative_delay(
                        current, old, self.settings["sample_rate"]
                    )
                except _UnstableReference:
                    residual = None
                stable = (
                    stable + 1
                    if residual is not None and abs(lag - old_lag + residual) <= 0.125
                    else 0
                )
            previous = lag, current
            if time.monotonic() - self.opened >= 20 and stable >= 3:
                self.reference = previous
                return
        raise AudioDeviceError(
            "ASIO timing did not stabilize. Check the clean loopback and retry."
        )

    def _capture(self, **settings):
        main = settings["playback"]
        loop = settings["loopback_playback"]
        if main.ndim != 1 or not len(main):
            raise AudioDeviceError("ASIO playback must be non-empty mono audio.")
        if loop is not None:
            if np.shape(loop) != np.shape(main):
                raise AudioDeviceError(
                    "Loopback playback must have the same shape as playback."
                )
            if (
                settings["loopback_input_channel"] is None
                or settings["loopback_output_channel"] is None
            ):
                raise AudioDeviceError("Both clean loopback channels must be selected.")
            if (
                settings["loopback_input_channel"] == settings["input_channel"]
                or settings["loopback_output_channel"] == settings["output_channel"]
            ):
                raise AudioDeviceError(
                    "Clean loopback channels must differ from the capture channels."
                )
        self._open(settings)
        cancel = settings["cancel"]
        if loop is not None:
            if self.reference is None or self.last_status:
                self.last_status = None
                self._stabilize(cancel)
            else:
                lag, current = self._probe(cancel)
                old_lag, old = self.reference
                try:
                    shift = relative_delay(current, old, settings["sample_rate"])
                except _UnstableReference:
                    shift = None
                if shift is None or abs(lag - old_lag + shift) > 0.125:
                    self.reference = None
                    self._stabilize(cancel)
        main, recorded = self._exchange(
            settings["playback"], loop, cancel, settings["progress"]
        )
        if loop is not None:
            lag, reference = self.reference
            check_timing(
                loop, recorded, settings["sample_rate"], lag, reference, cancel=cancel
            )
        if settings["progress"] is not None:
            settings["progress"](1.0)
        return main, recorded
