# Input and output level calibration

The capture app's **Project** tab has optional **Input level at 0 dBFS** and
**Output level at 0 dBFS** fields. Check **Known** only when the respective
level has been measured or established for the actual gain settings.
Unknown values are omitted from export; the displayed +12 dBu is not
silently treated as a measurement.

* Input level describes the analog dBu at the device input when the digital
  playback/reamp path carries 0 dBFS. Include any reamp attenuation.
* Output level describes the analog dBu at the recording input corresponding
  to 0 dBFS in the saved output WAV. Include the actual input gain setting.

The values are saved in `capture_project.json`, survive plan regeneration,
and are included in the generated ConcatWaveNet, HyperWaveNet and HyperA2
model configs:

```json
"metadata": {
    "input_level_dbu": 18.0,
    "output_level_dbu": 12.5
}
```

These are the existing standard NAM `UserMetadata` names. The parametric
trainer now forwards this top-level model-config metadata to final exports
and checkpoint NAM exports. Both parametric and fixed-setting baked files
receive the levels. Output-normalization compensation restores the original
capture signal level, so the original calibration values remain correct.

The playback interface has its own calibration, which can differ from the
capture chain. In a compatible plugin, set **Input calibration** to the
analog level corresponding to 0 dBFS at the playback interface input. NAM's
input correction is `interface calibration - model input_level_dbu` dB.
Calibrated output adds `model output_level_dbu - interface calibration` dB.
The controls in the [DLC86 parametric plugin](https://github.com/DLC86/NamParametricPlugin)
also provide the standard Raw, Normalized, Calibrated and Auto output modes.
