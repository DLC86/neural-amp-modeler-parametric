# HyperA2: parametric NAM A2-Full and A2-Lite

`HyperA2` uses the same two audio topologies as stock A2. Knob values
condition each topology through the existing `HyperWaveNet` hypernetwork.
One training run and one parametric file contain both sizes.

| Audio setting | Channels | Bottleneck | Receptive field |
| --- | ---: | ---: | ---: |
| A2-Lite (`lite`) | 3 | 3 | 6347 samples |
| A2-Full (`full`) | 8 | 8 | 6347 samples |

Both sizes retain all 23 layers, LeakyReLU with slope 0.01, ungated
activations, the residual 1x1 projections, mono audio conditioning and the
16-tap output head. There are no extra audio-rate knob channels.

```text
layer:       1  2  3  4  5   6   7  8  9 10 11 12  13  14 15 16 17 18 19 20 21  22  23
kernel:      6  6  6  6  6   6   6  6  6  6  6  6   6   6 15 15  6  6  6  6  6   6   6
dilation:    1  3  7 17 41 101 239  1  3  7 17 41 101 239  1 13  1  3  7 17 41 101 239
```

The reference is `nam/train/_resources/config_model_packed.json`.
The original fork's `nam_full_configs/parametric/model.json` already uses
the Full audio topology; the addition here is the dual-size model,
training, validation and export integration. Topology equality with the
stock A2 configurations is covered by a test.

## Conditioning and training

Each branch computes `W(p) = W_base + delta_W(p)` through the existing
hypernetwork. As in packed A2, Lite and Full have independent weights;
Lite is not obtained by cutting three channels from a trained Full model.
This implementation evaluates the two branches sequentially during each
training step rather than using the masked packed-convolution optimization.
Both branch losses are averaged before backpropagation.

Both branches are also evaluated during validation. Logs include separate
`ESR/unseen_audio_unseen_params/lite` and
`ESR/unseen_audio_unseen_params/full` metrics, in addition to the
combined metrics. Best-checkpoint selection uses the combined validation
metric, so a checkpoint always preserves both branches from the same step.

The default selector keeps the dilated convolution weights shared across
knob settings, matching the original HyperWaveNet recipe. Biases and other
selected projections receive generated deltas. Base weights remain trainable.
`hypernet.mode` (`low_rank` or `full`) and `hypernet.selector` are independent
of the audio size selection. Changing `size` never changes those options.

## Capture and training

Start the capture app with `nam-capture --hyperwavenet`. The
**Export HyperA2 Full + Lite Configs** button writes `model_hyper_a2.json`
and `learning_hyper_a2.json` using the project's real knob definitions.
Then use the existing trainer:

```sh
nam-full-parametric data.json model_hyper_a2.json learning_hyper_a2.json outputs
```

For a hand-written setup, start with
`nam_full_configs/parametric/model_hyper_a2.json` and the existing
`nam_full_configs/parametric/learning.json`. Replace the example Gain/Tone
controls with the controls in the capture data. Learning-rate and
hypernetwork-capacity defaults are starting points, not measured quality
claims for a particular amp.

In Python:

```python
from nam.models.parametric import HyperA2

model = HyperA2.init_from_config(config)
model.set_size("full")  # 8 audio channels
y_full = model(audio, params)
with model.using_size("lite"):
    y_lite = model(audio, params)  # 3 audio channels; restores Full on exit
```

Inference runs only the selected branch. This Python implementation
generates weights at each forward, like HyperWaveNet. A realtime runtime
should regenerate them when controls change and then run the selected
ordinary A2 audio network.

## Export and playback

The trainer and its checkpoint callback write:

* `model_parametric.nam`: architecture `HyperA2`, with both HyperWaveNet
  states and editable knob definitions. A runtime must implement HyperA2
  loading to play this file with live controls.
* `model.nam`: a stock `SlimmableContainer` containing two ordinary
  `WaveNet` models, baked at the knobs' defaults. It supports normal A2
  Full/Lite playback in stock A2-capable players, with fixed controls.

Existing NAM-Oversampler and NamParametricPlugin builds do not gain support
for the new parametric architecture just by updating the trainer. The
runtime addition is separate from this repository's training implementation.
This change also retains HyperWaveNet's existing behavior during control
changes; it does not implement a realtime transition/crossfade solution.

To bake another fixed setting:

```python
model.bake({"Gain": 6.0, "Tone": 4.0}).export(output_directory)
```

For normalized training targets, pass the dataset's `output_scale` to
`bake` or `export_parametric`. Both audio sizes are compensated, including
the head-scale values in the config and the flattened weights.

### Parametric file layout

HyperA2 uses the parametric version namespace (`1.0.0`). Its config contains
`params`, `size`, `head_scale`, `hypernet` and `submodels` in Lite/Full order.
Each submodel entry contains:

* `name`, `max_value` (0.5 for Lite; 1.0 for Full), `architecture`;
* its ordinary HyperWaveNet `config`, including the `hypernet.delta_map`;
* `weight_offset`, `weight_count` and `base_weight_count`.

The top-level weights are concatenated Lite then Full. Each slice uses the
unchanged HyperWaveNet layout: base WaveNet weights (including trailing
head scale), followed by hypernetwork state. Offsets are relative to the
top-level weights. A runtime selects the size first, then loads that slice
through its HyperWaveNet loader. The A2 size threshold is 0.5, with values
below 0.5 selecting Lite and values at or above 0.5 selecting Full.

Python round-trip loading:

```python
import json

payload = json.loads(model_path.read_text())
model = HyperA2.init_from_config({
    **payload["config"], "sample_rate": payload.get("sample_rate")
})
model.import_weights(payload["weights"])
```

Sources: [original HyperWaveNet](https://github.com/phillipmself/neural-amp-modeler-parametric/blob/a3684411076be69dab53505df14922487697f2cf/nam/models/parametric/_hyperwavenet.py),
[stock A2 training configuration](https://github.com/phillipmself/neural-amp-modeler-parametric/blob/a3684411076be69dab53505df14922487697f2cf/nam/train/_resources/config_model_packed.json),
[A2 architecture from TONE3000](https://github.com/tone-3000/embedded).
