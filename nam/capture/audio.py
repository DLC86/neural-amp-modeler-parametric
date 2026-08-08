"""
Audio device enumeration and simultaneous playback/recording.

Everything hardware-facing hides behind the :class:`PlaybackRecorder` protocol so the
capture session (and its tests) can run against a fake recorder without opening a
stream. ``sounddevice`` is imported lazily: enumerating or streaming only happens on
user action, and the GUI must be able to start even if PortAudio is unhappy.
"""

from __future__ import annotations

import os as _os
import sys as _sys
import time as _time
from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable
from typing import Literal
from typing import Optional as _Optional
from typing import Protocol as _Protocol
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np


# On Windows, python-sounddevice can load a PortAudio build with ASIO support when
# SD_ENABLE_ASIO is set before the first import of sounddevice. Keep sounddevice lazy,
# but enable ASIO automatically for the capture application so the GUI can enumerate
# ASIO alongside MME, DirectSound, WASAPI, and WDM-KS without requiring a shell variable.
# setdefault() deliberately preserves an explicit user choice (including disabling it).
if _sys.platform == "win32":
    _os.environ.setdefault("SD_ENABLE_ASIO", "1")


# Suggested stream latency: seconds, or one of PortAudio's per-device presets.
_Latency = _Union[float, Literal["low", "high"]]

LATENCY_CHOICES: _Tuple[_Tuple[str, _Latency], ...] = (
    ("System default (safest)", "high"),
    ("Low (device default)", "low"),
    ("5 ms", 0.005),
    ("2 ms", 0.002),
)


class CaptureCancelled(Exception):
    pass


class AudioDeviceError(RuntimeError):
    pass


class AudioDropoutError(AudioDeviceError):
    pass


DBFS_FLOOR = -120.0


def peak_to_dbfs(peak: float) -> float:
    if peak <= 0:
        return DBFS_FLOOR
    return max(20.0 * float(_np.log10(peak)), DBFS_FLOOR)


@_dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices(refresh: bool = False) -> list[DeviceInfo]:
    import sounddevice as sd

    if refresh:
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass

    host_apis = sd.query_hostapis()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append(
            DeviceInfo(
                index=index,
                name=device["name"],
                host_api=host_apis[device["hostapi"]]["name"],
                max_input_channels=device["max_input_channels"],
                max_output_channels=device["max_output_channels"],
                default_samplerate=device["default_samplerate"],
            )
        )
    return devices


def _initialize_audio_thread() -> None:
    """Initialize PortAudio on the thread that will create the audio stream.

    The Windows PortAudio ASIO backend has a thread-affinity requirement in some
    drivers: initializing PortAudio on the GUI thread and opening the ASIO stream on
    a worker thread can result in the host error ``Failed to load ASIO driver`` even
    though the same device opens successfully from a single-threaded program.

    The capture worker owns the stream exclusively, so it is safe to rebuild PortAudio
    immediately before resolving/opening the stream. This is deliberately kept out of
    device enumeration; enumeration can happen on the GUI thread, while the actual
    stream must be initialized on the worker thread.
    """
    import sounddevice as sd

    try:
        sd._terminate()
    except Exception:
        pass
    sd._initialize()


def current_device_sample_rates(allow_reinit: bool = False) -> dict[str, float]:
    import sys as _sys

    if _sys.platform == "darwin":
        try:
            return _coreaudio_sample_rates()
        except Exception:
            return {}

    if not allow_reinit:
        return {}
    import sounddevice as sd

    try:
        sd._terminate()
        sd._initialize()
        return {device.name: device.default_samplerate for device in list_devices()}
    except Exception:
        return {}


def _coreaudio_sample_rates() -> dict[str, float]:
    import ctypes
    import ctypes.util

    def fourcc(code: str) -> int:
        return (
            (ord(code[0]) << 24)
            | (ord(code[1]) << 16)
            | (ord(code[2]) << 8)
            | ord(code[3])
        )

    class _Addr(ctypes.Structure):
        _fields_ = [
            ("mSelector", ctypes.c_uint32),
            ("mScope", ctypes.c_uint32),
            ("mElement", ctypes.c_uint32),
        ]

    system_object = 1
    scope_global = fourcc("glob")
    element_main = 0
    prop_devices = fourcc("dev#")
    prop_name = fourcc("lnam")
    prop_nominal_rate = fourcc("nsrt")
    utf8 = 0x08000100

    core_audio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    core_foundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    core_foundation.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    core_foundation.CFStringGetCString.restype = ctypes.c_bool

    addr = _Addr(prop_devices, scope_global, element_main)
    size = ctypes.c_uint32(0)
    core_audio.AudioObjectGetPropertyDataSize(
        system_object, ctypes.byref(addr), 0, None, ctypes.byref(size)
    )
    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    device_ids = (ctypes.c_uint32 * count)()
    core_audio.AudioObjectGetPropertyData(
        system_object, ctypes.byref(addr), 0, None, ctypes.byref(size), device_ids
    )

    rates: dict[str, float] = {}
    for device_id in device_ids:
        name_addr = _Addr(prop_name, scope_global, element_main)
        cfstr = ctypes.c_void_p()
        name_size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        if (
            core_audio.AudioObjectGetPropertyData(
                device_id,
                ctypes.byref(name_addr),
                0,
                None,
                ctypes.byref(name_size),
                ctypes.byref(cfstr),
            )
            != 0
            or not cfstr.value
        ):
            continue
        buffer = ctypes.create_string_buffer(256)
        ok = core_foundation.CFStringGetCString(cfstr, buffer, 256, utf8)
        core_foundation.CFRelease(cfstr)
        if not ok:
            continue
        name = buffer.value.decode("utf-8", "replace")

        rate_addr = _Addr(prop_nominal_rate, scope_global, element_main)
        rate = ctypes.c_double(0)
        rate_size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_double))
        if (
            core_audio.AudioObjectGetPropertyData(
                device_id,
                ctypes.byref(rate_addr),
                0,
                None,
                ctypes.byref(rate_size),
                ctypes.byref(rate),
            )
            == 0
            and rate.value > 0
        ):
            rates[name] = rate.value
    return rates


def find_device(
    name: str,
    *,
    kind: str,
    host_api: _Optional[str] = None,
) -> DeviceInfo:
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output'; got {kind!r}")
    candidates = [
        device
        for device in list_devices()
        if device.name == name
        and (host_api is None or device.host_api == host_api)
        and (
            device.max_input_channels > 0
            if kind == "input"
            else device.max_output_channels > 0
        )
    ]
    if len(candidates) == 0:
        available = ", ".join(
            sorted(
                {
                    device.name
                    for device in list_devices()
                    if (
                        device.max_input_channels > 0
                        if kind == "input"
                        else device.max_output_channels > 0
                    )
                }
            )
        )
        raise AudioDeviceError(
            f"No {kind} device named {name!r}"
            + (f" on host API {host_api!r}" if host_api else "")
            + f". Available: {available}"
        )
    return candidates[0]


class PlaybackRecorder(_Protocol):
    def playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        *,
        output_device: _Optional[int] = None,
        input_device: _Optional[int] = None,
        output_channel: int = 1,
        input_channel: int = 1,
        loopback_output_channel: _Optional[int] = None,
        loopback_input_channel: _Optional[int] = None,
        loopback_playback: _Optional[_np.ndarray] = None,
        blocksize: int = 0,
        latency: _Latency = "low",
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _Tuple[_np.ndarray, _Optional[_np.ndarray]]:
        ...


def _device_channels(index: _Optional[int], *, kind: str) -> int:
    import sounddevice as sd

    info = sd.query_devices(kind=kind) if index is None else sd.query_devices(index)
    return int(info[f"max_{kind}_channels"])


def _raise_on_dropout(status, *, latency: _Latency, blocksize: int) -> None:
    lost_input = bool(getattr(status, "input_overflow", False))
    lost_output = bool(getattr(status, "output_underflow", False))
    if not (lost_input or lost_output):
        return
    what = []
    if lost_input:
        what.append("recorded samples were dropped")
    if lost_output:
        what.append("gaps were played into the output")
    raise AudioDropoutError(
        f"The audio stream could not keep up: {' and '.join(what)}. The capture is "
        "missing audio and was not saved. Raise 'Stream latency' (currently "
        f"{latency!r}) or the buffer size (currently "
        f"{'Auto' if blocksize == 0 else blocksize}) in Audio settings, close other "
        "audio applications, and capture again."
    )


class SounddeviceRecorder:
    _POLL_MS = 50

    def playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        *,
        output_device: _Optional[int] = None,
        input_device: _Optional[int] = None,
        output_channel: int = 1,
        input_channel: int = 1,
        loopback_output_channel: _Optional[int] = None,
        loopback_input_channel: _Optional[int] = None,
        loopback_playback: _Optional[_np.ndarray] = None,
        blocksize: int = 0,
        latency: _Latency = "low",
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _Tuple[_np.ndarray, _Optional[_np.ndarray]]:
        import sounddevice as sd

        # ASIO on Windows can fail with host error 0 when PortAudio was initialized
        # on the Qt GUI thread and the stream is subsequently created on the QThread
        # used for capture. Reinitialize PortAudio here, on the actual audio worker
        # thread, immediately before any device queries or stream creation.
        if _sys.platform == "win32":
            _initialize_audio_thread()

        playback = _np.asarray(playback, dtype=_np.float32)
        if playback.ndim != 1:
            raise ValueError("Playback signal must be mono (1-D).")
        if len(playback) == 0:
            raise ValueError("Playback signal is empty.")

        out_channels = _device_channels(output_device, kind="output")
        in_channels = _device_channels(input_device, kind="input")
        if not 1 <= output_channel <= out_channels:
            raise AudioDeviceError(
                f"Output channel {output_channel} is outside the device's "
                f"1..{out_channels} range."
            )
        if not 1 <= input_channel <= in_channels:
            raise AudioDeviceError(
                f"Input channel {input_channel} is outside the device's "
                f"1..{in_channels} range."
            )

        loopback_enabled = loopback_output_channel is not None or loopback_input_channel is not None
        if loopback_enabled:
            if loopback_output_channel is None or loopback_input_channel is None:
                raise AudioDeviceError("Loopback output and input channels must be set together.")
            if output_device != input_device:
                raise AudioDeviceError("Loopback requires the same input and output device.")
            if loopback_playback is None:
                raise AudioDeviceError("Loopback playback signal is missing.")
            loopback_playback = _np.asarray(loopback_playback, dtype=_np.float32)
            if loopback_playback.ndim != 1 or len(loopback_playback) != len(playback):
                raise AudioDeviceError("Loopback playback must be mono and match playback length.")
            if not 1 <= loopback_output_channel <= out_channels:
                raise AudioDeviceError(
                    f"Loopback output channel {loopback_output_channel} is outside the device's "
                    f"1..{out_channels} range."
                )
            if not 1 <= loopback_input_channel <= in_channels:
                raise AudioDeviceError(
                    f"Loopback input channel {loopback_input_channel} is outside the device's "
                    f"1..{in_channels} range."
                )
            if loopback_output_channel == output_channel:
                raise AudioDeviceError("Loopback output channel must differ from the primary output channel.")
            if loopback_input_channel == input_channel:
                raise AudioDeviceError("Loopback input channel must differ from the primary input channel.")

        out_max = max(output_channel, loopback_output_channel or 0)
        in_max = max(input_channel, loopback_input_channel or 0)
        out_buffer = _np.zeros((len(playback), out_max), dtype=_np.float32)
        in_buffer = _np.zeros((len(playback), in_max), dtype=_np.float32)
        out_buffer[:, output_channel - 1] = playback
        if loopback_enabled:
            out_buffer[:, loopback_output_channel - 1] = loopback_playback

        status = None
        chunks = []
        loopback_chunks = [] if loopback_enabled else None
        total = len(playback)
        position = 0

        def callback(indata, outdata, frames, _time_info, callback_status):
            nonlocal position, status
            if callback_status:
                status = callback_status
            if cancel and cancel():
                raise sd.CallbackAbort
            end = min(position + frames, total)
            count = end - position
            if count > 0:
                outdata[:count, :] = out_buffer[position:end, :]
                chunks.append(indata[:count, input_channel - 1].copy())
                if loopback_enabled:
                    loopback_chunks.append(indata[:count, loopback_input_channel - 1].copy())
            if frames > count:
                outdata[count:, :] = 0
            position = end
            if progress:
                progress(min(1.0, position / total))
            if position >= total:
                raise sd.CallbackStop

        try:
            with sd.Stream(
                samplerate=sample_rate,
                blocksize=blocksize,
                dtype="float32",
                channels=(in_max, out_max),
                device=(input_device, output_device),
                latency=latency,
                callback=callback,
            ):
                while position < total:
                    if cancel and cancel():
                        raise CaptureCancelled()
                    _time.sleep(self._POLL_MS / 1000.0)
        except sd.CallbackAbort:
            raise CaptureCancelled()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(f"Could not open audio stream: {exc}") from exc

        _raise_on_dropout(status, latency=latency, blocksize=blocksize)
        if position < total:
            raise AudioDeviceError("The audio stream ended before the full playback was captured.")
        recording = _np.concatenate(chunks) if chunks else _np.zeros(0, dtype=_np.float32)
        loopback_recording = (
            _np.concatenate(loopback_chunks) if loopback_chunks else None
        )
        return recording, loopback_recording
