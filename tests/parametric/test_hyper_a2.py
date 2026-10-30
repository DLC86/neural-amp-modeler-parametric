import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from nam.models import factory
from nam.models.parametric import HyperA2, export_parametric
from nam.models.wavenet._wavenet import WaveNet as InnerWaveNet
from nam.train.parametric import _ParametricLightningModule, _ParametricLossConfig


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        "params": [
            {"name": "Gain", "type": "continuous", "min": 0, "max": 10, "default": 5}
        ],
        "sample_rate": 48000,
        "hypernet": {"hidden_sizes": [4], "mode": "low_rank", "rank": 2},
    }


def make_model():
    return factory.init("HyperA2", args=(config(),))


def test_exact_stock_topologies_and_receptive_field():
    model = make_model()
    with open("nam/train/_resources/config_model_packed.json") as fp:
        stock = json.load(fp)["net"]["config"]["submodels"]
    assert model.sizes == ("lite", "full")
    assert model.receptive_field == 6347
    assert model.size == "full"
    for size, entry in zip(model.sizes, stock):
        actual = model.branch(size)._template
        expected = InnerWaveNet.init_from_config(entry["config"])
        assert actual.export_config() == expected.export_config()
    assert [len(model.branch(s)._template.export_weights()) for s in model.sizes] == [
        1871,
        12146,
    ]


def test_selection_runs_only_selected_branch_and_restores_on_error(mocker):
    model = make_model()
    x = torch.randn(2, model.receptive_field + 7)
    p = torch.tensor([4.0])
    unused = mocker.spy(model.branch("full"), "_run_conditioned")
    with model.using_size("lite"):
        actual = model(x, p, pad_start=False)
    expected = model.branch("lite")(x, p, pad_start=False)
    torch.testing.assert_close(actual, expected)
    unused.assert_not_called()
    assert model.size == "full"
    with pytest.raises(RuntimeError):
        with model.using_size("lite"):
            raise RuntimeError("test")
    assert model.size == "full"
    with pytest.raises(ValueError, match="size"):
        model.set_size("invalid")


def test_training_scores_both_sizes_and_backpropagates():
    model = make_model()
    module = _ParametricLightningModule(model, loss_config=_ParametricLossConfig())
    module.train()
    module.on_train_epoch_start()
    x = torch.randn(1, model.receptive_field + 31)
    p = torch.tensor([[3.0]])
    target = torch.randn(1, 32)
    # The parametric batch order is audio, controls, target.
    loss = module.training_step((x, p, target), 0)
    assert torch.isfinite(loss)
    loss.backward()
    for size in model.sizes:
        branch = model.branch(size)
        assert any(
            v.grad is not None and v.grad.abs().sum() > 0
            for v in branch._template.parameters()
        )
        assert any(
            v.grad is not None and v.grad.abs().sum() > 0
            for v in branch._hypernet.parameters()
        )
    assert set(module._train_size_metrics) == {"lite", "full"}
    module.eval()
    module.on_validation_epoch_start()
    with torch.no_grad():
        module.validation_step((x, p, target), 0)
    assert set(module._val_size_metrics) == {"lite", "full"}
    assert model.size == "full"


def test_parametric_export_restore_and_scaling(tmp_path):
    model = make_model().eval()
    with torch.no_grad():
        for size in model.sizes:
            model.branch(size)._hypernet._final.weight.normal_(0, 0.01)
    export_parametric(model, tmp_path, output_scale=2.0)
    exported = json.loads((tmp_path / "model.nam").read_text())
    assert exported["architecture"] == "HyperA2"
    restored = HyperA2.init_from_config({**exported["config"], "sample_rate": 48000})
    assert restored.import_weights(exported["weights"]) == len(exported["weights"])
    x = torch.randn(128)
    for size in model.sizes:
        model.set_size(size)
        restored.set_size(size)
        with torch.no_grad():
            torch.testing.assert_close(
                restored(x, torch.tensor([7.0])), model(x, torch.tensor([7.0])) / 2
            )
    # Export must not alter the live training model.
    assert model.branch("full")._template._head_scale == 0.01
    assert exported["config"]["submodels"][0]["max_value"] == 0.5
    assert exported["config"]["submodels"][1]["max_value"] == 1.0
    with pytest.raises(ValueError, match="weights"):
        restored.import_weights(exported["weights"][:-1])
    bad = deepcopy(exported["config"])
    bad["submodels"][0]["weight_offset"] = 1
    with pytest.raises(ValueError, match="layout"):
        HyperA2.init_from_config(bad)


def test_bake_both_sizes_matches_conditioned_model_and_stock_export(tmp_path, mocker):
    model = make_model().eval()
    p = torch.tensor([6.0])
    with torch.no_grad():
        for size in model.sizes:
            model.branch(size)._hypernet._final.weight.normal_(0, 0.01)
    baked = model.bake(p, output_scale=2.0)
    x = torch.randn(1, model.receptive_field + 31)
    for index, size in enumerate(model.sizes):
        model.set_size(size)
        with torch.no_grad():
            torch.testing.assert_close(
                baked.extract_submodel(index)(x, pad_start=False),
                model(x, p, pad_start=False),
                rtol=1e-5,
                atol=1e-6,
            )
    # Skip costly standard loudness probes; the waveform comparison is above.
    mocker.patch("nam.models.base._Base._metadata_loudness", return_value=-20.0)
    mocker.patch("nam.models.base._Base._metadata_gain", return_value=0.5)
    baked.export(tmp_path)
    exported = json.loads((tmp_path / "model.nam").read_text())
    assert exported["architecture"] == "SlimmableContainer"
    assert exported["weights"] == []
    for entry in exported["config"]["submodels"]:
        child = entry["model"]
        assert child["architecture"] == "WaveNet"
        assert child["config"]["head_scale"] == pytest.approx(0.005)
        assert child["weights"][-1] == pytest.approx(0.005)


def test_training_main_writes_dual_size_checkpoints_and_final_exports(tmp_path, mocker):
    from nam.data import np_to_wav
    from nam.train.parametric import main
    from tests.parametric.test_train import _learning_config

    mocker.patch("nam.models.base._Base._metadata_loudness", return_value=-20.0)
    mocker.patch("nam.models.base._Base._metadata_gain", return_value=0.5)
    x = (0.1 * np.sin(np.arange(6500) * 0.03)).astype(np.float32)
    np_to_wav(x, tmp_path / "input.wav", rate=48000)
    np_to_wav(np.tanh(2 * x), tmp_path / "output.wav", rate=48000)
    common = {
        "x_path": str(tmp_path / "input.wav"),
        "y_path": str(tmp_path / "output.wav"),
        "delay": 0,
        "ny": 32,
        "require_input_pre_silence": None,
    }
    data = {
        "type": "parametric",
        "common": common,
        "train": [{"params": {"Gain": 3.0}}],
        "validation": [{"params": {"Gain": 7.0}}],
        "joint": [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ],
    }
    outdir = tmp_path / "run"
    outdir.mkdir()
    main(
        data_config=data,
        model_config={
            "net": {"name": "HyperA2", "config": config()},
            "loss": {"val_loss": "esr"},
            "optimizer": {"lr": 0.001},
            "metadata": {"input_level_dbu": 18.0, "output_level_dbu": 12.5},
            "lr_scheduler": None,
        },
        learning_config=_learning_config(),
        outdir=outdir,
        no_show=True,
        make_plots=False,
    )
    final = json.loads((outdir / "model_parametric.nam").read_text())
    assert final["architecture"] == "HyperA2"
    assert final["metadata"]["input_level_dbu"] == 18.0
    assert final["metadata"]["output_level_dbu"] == 12.5
    assert len(final["config"]["submodels"]) == 2
    baked = json.loads((outdir / "model.nam").read_text())
    assert baked["architecture"] == "SlimmableContainer"
    assert baked["metadata"]["input_level_dbu"] == 18.0
    assert baked["metadata"]["output_level_dbu"] == 12.5
    assert [
        entry["model"]["config"]["layers"][0]["channels"]
        for entry in baked["config"]["submodels"]
    ] == [3, 8]
    checkpoints = list(outdir.rglob("*.ckpt"))
    assert checkpoints
    for checkpoint in checkpoints:
        assert checkpoint.with_suffix(".nam").exists()
        assert checkpoint.with_name(checkpoint.stem + "_parametric.nam").exists()
        checkpoint_nam = json.loads(
            checkpoint.with_name(checkpoint.stem + "_parametric.nam").read_text()
        )
        assert checkpoint_nam["metadata"]["input_level_dbu"] == 18.0
        assert checkpoint_nam["metadata"]["output_level_dbu"] == 12.5
    restored = HyperA2.init_from_config(final["config"])
    restored.import_weights(final["weights"])
    assert restored.branch("full")._template._head_scale != pytest.approx(0.01)
