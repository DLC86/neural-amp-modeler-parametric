"""Compatibility launcher for the capture GUI.

The capture GUI historically expected PortAudio to expose a single device with both
input and output channels. On Windows, some drivers expose the input and output
endpoints as separate PortAudio devices with the same name and host API. This launcher
normalizes those endpoint pairs before constructing the GUI, without changing the
capture engine or project format.
"""

from __future__ import annotations

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
        result.append(
            _replace(representative, max_output_channels=output_channels)
        )

    return result


def main() -> None:
    """Launch the normal GUI after installing the device-list compatibility shim."""
    from . import main as _gui

    _gui.duplex_devices = logical_duplex_devices
    _gui.main()


if __name__ == "__main__":
    main()
