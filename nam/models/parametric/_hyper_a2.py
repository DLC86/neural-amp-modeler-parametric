"""Knob-conditioned A2 with the official Full and Lite audio topologies.

The branches are independent, as in packed A2 training. Both are scored against
each capture; inference evaluates only the selected branch.
"""

from contextlib import contextmanager as _contextmanager
from copy import deepcopy as _deepcopy

import numpy as _np
import torch as _torch
import torch.nn as _nn

from ..wavenet import PackedWaveNet as _PackedWaveNet
from ._base import ParametricNet as _ParametricNet
from ._hyperwavenet import HyperWaveNet as _HyperWaveNet
from ._spec import ParamSpec as _ParamSpec

_CHANNELS = {"lite": 3, "full": 8}
_KERNELS = (6,) * 14 + (15, 15) + (6,) * 7
_DILATIONS = (1, 3, 7, 17, 41, 101, 239) * 2 + (1, 13) + (1, 3, 7, 17, 41, 101, 239)


def _template_config(channels, head_scale):
    return {
        "layers": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "bottleneck": channels,
                "kernel_sizes": list(_KERNELS),
                "dilations": list(_DILATIONS),
                "activation": "LeakyReLU",
                "gated": False,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
            }
        ],
        "head_scale": head_scale,
    }


class HyperA2(_ParametricNet):
    """Two HyperWaveNets, selectable as A2-Lite (3ch) or A2-Full (8ch).

    ``size`` is a playback selection, independent of the hypernetwork's
    ``mode`` (full or low_rank) and tensor selector.
    """

    def __init__(self, *, branches, size="full", sample_rate=None):
        if tuple(branches) != tuple(_CHANNELS):
            raise ValueError("HyperA2 requires branches ordered lite, full")
        specs = branches["full"].param_specs
        if any(branch.param_specs != specs for branch in branches.values()):
            raise ValueError(
                "HyperA2 branches must have identical parameter definitions"
            )
        super().__init__(param_specs=specs, sample_rate=sample_rate)
        self._branches = _nn.ModuleDict(branches)
        self.set_size(size)

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        size = config.pop("size", "full")
        raw_specs = config.pop("params")
        specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_specs)
        hypernet = config.pop("hypernet", {})
        head_scale = config.pop("head_scale", 0.01)
        exported_submodels = config.pop("submodels", None)
        if config:
            raise ValueError(f"Unexpected HyperA2 config keys: {sorted(config)}")
        branches = {}
        for name, channels in _CHANNELS.items():
            branch_config = _template_config(channels, head_scale)
            branch_config.update(
                params=[spec.to_dict() for spec in specs],
                hypernet=_deepcopy(hypernet),
                sample_rate=sample_rate,
            )
            branches[name] = _HyperWaveNet.init_from_config(branch_config)
        if exported_submodels is not None:
            expected = cls(branches=branches, size=size, sample_rate=sample_rate)
            if exported_submodels != expected._export_submodels():
                raise ValueError(
                    "HyperA2 submodel layout does not match its configuration"
                )
        return dict(branches=branches, size=size, sample_rate=sample_rate)

    @property
    def sizes(self):
        return tuple(_CHANNELS)

    @property
    def size(self):
        return self._size

    def set_size(self, size):
        if size not in _CHANNELS:
            raise ValueError(f"HyperA2 size must be 'lite' or 'full'; got {size!r}")
        self._size = size

    @_contextmanager
    def using_size(self, size):
        previous = self.size
        self.set_size(size)
        try:
            yield
        finally:
            self.set_size(previous)

    def branch(self, size):
        if size not in _CHANNELS:
            raise ValueError(f"Unknown HyperA2 size {size!r}")
        return self._branches[size]

    @property
    def pad_start_default(self):
        return True

    @property
    def receptive_field(self):
        return self.branch("full").receptive_field

    @property
    def requires_uniform_batch_params(self):
        return self.branch(self.size).requires_uniform_batch_params

    def set_uniform_batch_params(self, uniform):
        for branch in self._branches.values():
            branch.set_uniform_batch_params(uniform)

    def set_compiled(self, enabled, **kwargs):
        for branch in self._branches.values():
            branch.set_compiled(enabled, **kwargs)

    @property
    def supports_compiled_step(self):
        return True

    def _run_conditioned(self, x, p):
        return self.branch(self.size)._run_conditioned(x, p)

    def _export_inner_config(self):
        branch = self.branch("full")
        return {
            "size": self.size,
            "head_scale": branch._template._head_scale,
            "hypernet": _deepcopy(branch._hypernet.config),
            "submodels": self._export_submodels(),
        }

    def _export_submodels(self):
        entries = []
        offset = 0
        for size in self.sizes:
            branch = self.branch(size)
            base_count = len(branch._template.export_weights())
            count = base_count + branch._hypernet.state_count()
            entries.append(
                {
                    "name": size,
                    "max_value": 0.5 if size == "lite" else 1.0,
                    "architecture": "HyperWaveNet",
                    "config": branch._export_config(),
                    "weight_offset": offset,
                    "weight_count": count,
                    "base_weight_count": base_count,
                }
            )
            offset += count
        return entries

    def _export_weights(self):
        # Lite followed by Full, each with its complete base + hypernetwork state.
        return _np.concatenate(
            [self.branch(size)._export_weights() for size in self.sizes]
        )

    def import_weights(self, weights, i=0):
        weights = _torch.as_tensor(weights)
        if weights.ndim != 1:
            raise ValueError("HyperA2 weights must be a flat 1-D sequence")
        counts = [
            len(self.branch(size)._template.export_weights())
            + self.branch(size)._hypernet.state_count()
            for size in self.sizes
        ]
        if len(weights) - i != sum(counts):
            raise ValueError(f"Expected {sum(counts)} HyperA2 weights after offset {i}")
        for size, count in zip(self.sizes, counts):
            # HyperWaveNet owns its entire tail, so give each branch its own slice.
            self.branch(size).import_weights(weights[i : i + count])
            i += count
        return i

    def bake(self, params, *, output_scale=None):
        """Bake both sizes into a stock A2 SlimmableContainer export surface."""
        from ._export import bake

        baked = []
        for size in self.sizes:
            branch = self.branch(size)
            branch.sample_rate = self.sample_rate
            baked.append(bake(branch, params))
        packed = _PackedWaveNet.init_from_config(
            {
                "sample_rate": self.sample_rate,
                "submodels": [
                    {
                        "name": f"channels_{_CHANNELS[size]}",
                        "config": model._net.export_config(),
                    }
                    for size, model in zip(self.sizes, baked)
                ],
                "export": {"container_max_values": "uniform"},
            }
        )
        if self.device is not None:
            packed.to(self.device)
        for index, model in enumerate(baked):
            packed.import_submodel(index, model)
        if output_scale is not None:
            from ...data import Dataset
            from ._export import _normalize_output_scale

            output_scale = _normalize_output_scale(output_scale)
            packed.export_model_dict_post_hooks.append(
                Dataset._ScaleOutputHook(scale=1.0 / output_scale)
            )
        packed.eval()
        return packed
