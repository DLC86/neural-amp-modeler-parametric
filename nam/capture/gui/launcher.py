"""Compatibility launcher for the capture GUI.

The capture GUI historically expected PortAudio to expose a single device with both
input and output channels. On Windows, some drivers expose the input and output
endpoints as separate PortAudio devices with the same name and host API. This launcher
normalizes those endpoint pairs before constructing the GUI, without changing the
capture engine or project format.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from dataclasses import replace as _replace
from typing import Sequence as _Sequence

from ..audio import DeviceInfo as _DeviceInfo


def logical_duplex_devices(
    devices: _Sequence[_DeviceInfo],
) -> list[_DeviceInfo]:
    """Return devices that can be used as one input/output capture interface.

    PortAudio may expose one logical interface as two records: an input-only record and
    an output-only record. When their names and host APIs match, combine their channel
    capabilities into the input record and use it as the GUI representation. The actual
    input/output device indices are still resolved by ``CaptureSession.find_device()``
    from the stored name and host API, so the representative index is never used for
    opening the stream.

    Real duplex devices are returned unchanged. Unpaired input/output devices are not
    returned because the existing capture GUI requires one logical interface for both
    directions.
    """
    groups: dict[tuple[str, str], list[_DeviceInfo]] = {}
    for device in devices:
        groups.setdefault((device.name, device.host_api), []).append(device)

    result: list[_DeviceInfo] = []
    for group in groups.values():
        duplex = [
            device
            for device in group
            if device.max_input_channels > 0 and device.max_output_channels > 0
        ]
        if duplex:
            result.extend(duplex)
            continue

        inputs = [device for device in group if device.max_input_channels > 0]
        outputs = [device for device in group if device.max_output_channels > 0]
        if not inputs or not outputs:
            continue

        # Prefer the input record as the representative because its index is valid for
        # the input direction. CaptureSession resolves the output index independently.
        representative = inputs[0]
        output_channels = max(device.max_output_channels for device in outputs)
        result.append(_replace(representative, max_output_channels=output_channels))

    return result


def configure_asio(argv: _Sequence[str], *, platform: str | None = None) -> list[str]:
    """Enable sounddevice's ASIO-enabled PortAudio DLL when requested.

    ``sounddevice`` selects its PortAudio DLL while the module is imported, so the
    environment variable must be set before the GUI imports any code that imports
    sounddevice. ``--asio`` is deliberately opt-in because sounddevice ships a
    non-ASIO DLL by default for compatibility. ``NAM_ENABLE_ASIO=1`` provides the same
    behavior for launchers that cannot pass command-line options.
    """
    platform = _sys.platform if platform is None else platform
    args = list(argv)
    requested = "--asio" in args or _os.environ.get("NAM_ENABLE_ASIO") == "1"
    if platform == "win32" and requested:
        _os.environ["SD_ENABLE_ASIO"] = "1"
        args = [arg for arg in args if arg != "--asio"]
    return args


def main() -> None:
    """Launch the normal GUI after installing the device-list compatibility shim."""
    _sys.argv[:] = configure_asio(_sys.argv)

    # On Windows, the GUI's periodic sample-rate check historically reinitialised
    # PortAudio every few seconds. That is disruptive for PortAudio backends such as
    # MOD Desktop's JACK bridge, which can emit a client-creation dialog whenever the
    # backend is torn down and recreated. The initial device refresh still performs
    # the explicit PortAudio refresh, but the background poll must not reinitialise it.
    # This wrapper preserves the live-rate API while making its periodic calls cheap.
    if _sys.platform == "win32":
        from .. import audio as _audio

        _current_device_sample_rates = _audio.current_device_sample_rates

        def _poll_device_sample_rates(*, allow_reinit: bool = False):
            return _current_device_sample_rates(allow_reinit=False)

        _audio.current_device_sample_rates = _poll_device_sample_rates

    from . import main as _gui

    _gui.duplex_devices = logical_duplex_devices
    _gui.main()


if __name__ == "__main__":
    main()
