from nam.capture.audio import DeviceInfo
from nam.capture.gui.launcher import logical_duplex_devices


def _device(index, name, host_api, inputs, outputs):
    return DeviceInfo(
        index=index,
        name=name,
        host_api=host_api,
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_samplerate=48000.0,
    )


def test_logical_duplex_devices_combines_split_input_output_endpoints():
    devices = [
        _device(1, "1-2 (QUAD-CAPTURE)", "MME", 2, 0),
        _device(12, "1-2 (QUAD-CAPTURE)", "MME", 0, 2),
        _device(13, "Altoparlanti", "MME", 0, 2),
    ]

    result = logical_duplex_devices(devices)

    assert len(result) == 1
    assert result[0].name == "1-2 (QUAD-CAPTURE)"
    assert result[0].host_api == "MME"
    assert result[0].max_input_channels == 2
    assert result[0].max_output_channels == 2
    assert result[0].index == 1


def test_logical_duplex_devices_keeps_native_duplex_devices():
    native = _device(4, "Interface", "ASIO", 8, 8)

    result = logical_duplex_devices([native])

    assert result == [native]


def test_logical_duplex_devices_does_not_pair_different_host_apis():
    devices = [
        _device(1, "Interface", "MME", 2, 0),
        _device(2, "Interface", "ASIO", 0, 2),
    ]

    assert logical_duplex_devices(devices) == []
