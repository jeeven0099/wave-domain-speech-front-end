from pathlib import Path

import numpy as np
import pytest
import torch

import benchmark_audiomnist_raspberry_pi as benchmark


def test_exact_model_mac_counts_match_frozen_reports() -> None:
    expected_16k = {
        "bc_resnet_1": (9_166, 2_389_632),
        "bc_resnet_2": (27_154, 7_021_824),
        "edgespeechnet_a": (108_058, 409_786_560),
        "matchboxnet_3x1x64": (62_850, 5_909_760),
        "mlperf_tiny_dscnn": (23_050, 20_428_800),
        "tc_resnet8": (25_530, 650_880),
    }
    expected_400 = {
        "bc_resnet_1": (8_734, 480_000),
        "bc_resnet_2": (31_154, 1_774_400),
        "edgespeechnet_a": (36_922, 14_407_200),
        "matchboxnet_3x1x64": (57_429, 22_455_600),
        "mlperf_tiny_dscnn": (19_594, 3_686_400),
        "tc_resnet8": (23_658, 1_963_200),
    }
    for name in benchmark.DIGITAL_MODEL_NAMES:
        model_16k = benchmark.DIGITAL_16K_FACTORIES[name]().eval()
        assert sum(parameter.numel() for parameter in model_16k.parameters()) == expected_16k[name][0]
        assert benchmark.profile_model(
            model_16k, benchmark.PreparedInput((torch.zeros(1, 96, 40),))
        ) == expected_16k[name][1]

        model_400 = benchmark.DIGITAL_400_FACTORIES[name]().eval()
        assert sum(parameter.numel() for parameter in model_400.parameters()) == expected_400[name][0]
        assert benchmark.profile_model(
            model_400, benchmark.PreparedInput((torch.zeros(1, 400, 1),))
        ) == expected_400[name][1]

    fabricated = benchmark.FabricatedPooledTcn(
        channels=32,
        frontend_kernel_size=32,
        frontend_stride=4,
        tcn_kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.1,
        classes=10,
    ).eval()
    macs_fabricated = benchmark.profile_model(
        fabricated,
        benchmark.PreparedInput(
            (torch.zeros(1, 400, 5), torch.tensor([400], dtype=torch.long))
        ),
    )
    assert macs_fabricated == 816_640


def test_load_model_validates_all_supported_checkpoint_contracts(tmp_path: Path) -> None:
    checkpoint_16k = tmp_path / "tc_resnet8_best.pt"
    torch.save(
        {
            "state_dict": benchmark.TCResNet8().state_dict(),
            "metadata": {"architecture": "tc_resnet8"},
        },
        checkpoint_16k,
    )
    np.savez(
        tmp_path / "compact_train_normalizer.npz",
        mean=np.zeros(40, dtype=np.float32),
        std=np.ones(40, dtype=np.float32),
    )
    assert benchmark.load_model("digital_16k", checkpoint_16k).architecture == "tc_resnet8"

    checkpoint_400 = tmp_path / "tc_resnet8_1d.pt"
    torch.save(
        {
            "model_state": benchmark.TCResNet8Envelope().state_dict(),
            "model_name": "tc_resnet8_1d",
            "train_mean": np.float32(0.1),
            "train_std": np.float32(0.2),
        },
        checkpoint_400,
    )
    assert (
        benchmark.load_model("digital_envelope_400hz", checkpoint_400).architecture
        == "tc_resnet8_1d"
    )

    config = {
        "channels": 32,
        "frontend_kernel_size": 32,
        "frontend_stride": 4,
        "tcn_kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16, 32, 64),
        "dropout": 0.1,
        "classes": 10,
    }
    model = benchmark.FabricatedPooledTcn(**config)
    checkpoint_fabricated = tmp_path / "fabricated.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "envelope_400hz_c32_fk32_long_tcn_stats_pool_softmax",
            "model_config": config,
            "input_mean": np.zeros(5, dtype=np.float32),
            "input_std": np.ones(5, dtype=np.float32),
        },
        checkpoint_fabricated,
    )
    loaded = benchmark.load_model("fabricated_400hz", checkpoint_fabricated)
    assert loaded.architecture == "envelope_400hz_c32_fk32_long_tcn_stats_pool_softmax"


def test_digital_envelope_preprocessing_is_exact_shape_and_finite() -> None:
    waveform = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
    example = benchmark.RawExample("sample", waveform, 48_000, 3)
    prepared = benchmark.preprocess_400hz_envelope(
        example, np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    )
    assert prepared.args[0].shape == (1, 400, 1)
    assert torch.isfinite(prepared.args[0]).all()


def test_fabricated_preprocessing_includes_channel_normalization() -> None:
    example = benchmark.FabricatedExample(
        "sample",
        np.full((400, 5), 3.0, dtype=np.float32),
        2,
        1.0,
    )
    prepared = benchmark.preprocess_fabricated(
        example,
        np.ones(5, dtype=np.float32),
        np.full(5, 2.0, dtype=np.float32),
    )
    assert prepared.args[0].shape == (1, 400, 5)
    assert torch.equal(prepared.args[0], torch.ones(1, 400, 5))
    assert prepared.args[1].item() == 400


def test_timing_schedule_covers_the_full_test_order() -> None:
    indices = [benchmark.scheduled_example_index(index, 500, 6_000) for index in range(500)]
    assert indices[0] == 0
    assert indices[-1] >= 5_980
    assert len(set(indices)) == 500


def test_fabricated_raw_frontend_matches_ac_rms_definition() -> None:
    time_axis = np.arange(20, dtype=np.float32)
    waveform = np.column_stack(
        [time_axis + 10.0 * channel for channel in range(5)]
    ).astype(np.float32)
    envelope = benchmark._frame_ac_rms_400hz(waveform, sample_rate_hz=400)
    assert envelope.shape == (17, 5)
    expected = np.std(waveform[:4], axis=0, ddof=0)
    np.testing.assert_allclose(envelope[0], expected, rtol=1e-6, atol=1e-6)

    example = benchmark.FabricatedRawExample("sample", waveform, 400, 1, 0.05)
    prepared = benchmark.preprocess_fabricated_raw(
        example, np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)
    )
    assert prepared.args[0].shape == (1, 17, 5)
    assert prepared.args[1].item() == 17


def test_final_cli_contract_times_every_heldout_example_once() -> None:
    assert benchmark.parse_args([]).runs == 6_000
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--runs", "500"])


def test_all_runs_digital_pairs_and_fabricated_models() -> None:
    workers = benchmark._requested_workers(benchmark.parse_args([]))
    assert len(workers) == 19
    assert workers[12] == ("fabricated_400hz", None)
    for model_name in benchmark.DIGITAL_MODEL_NAMES:
        assert ("digital_16k", model_name) in workers
        assert ("digital_envelope_400hz", model_name) in workers
        assert ("fabricated_400hz_baselines", model_name) in workers
