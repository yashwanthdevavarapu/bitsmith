from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitsmith.calib import (
    METHODS,
    awq_search,
    calibrate_and_quantize,
    collect_activation_stats,
    damped_inverse_cholesky,
    gptq_quantize,
    output_error,
    smooth_weight,
    smoothing_factors,
)
from bitsmith.eval import TinyLM, TinyLMConfig, sqnr_db
from bitsmith.nn import QLinear, make_codec
from bitsmith.quant import QuantSpec, fake_quantize, quantize_activations_per_token


def _correlated_inputs(n: int = 2048, d: int = 128, seed: int = 0) -> torch.Tensor:
    """Inputs with strong cross-channel correlation and a few loud channels, like real activations."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(d, d, generator=g) / d**0.5
    x = torch.randn(n, d, generator=g) @ (torch.eye(d) + 2 * mix)
    x[:, :4] *= 8
    return x


def _rtn(w: torch.Tensor, group: int) -> torch.Tensor:
    codec = make_codec("int4", group)
    return codec.decode(codec.encode(w), *w.shape)


# SmoothQuant ---------------------------------------------------------------------------


def test_smoothing_is_mathematically_invariant() -> None:
    w = torch.randn(32, 64)
    x = torch.randn(10, 64) * torch.rand(64) * 10
    s = smoothing_factors(x.abs().amax(0), w, alpha=0.5)
    ws, inv = smooth_weight(w, s)
    torch.testing.assert_close((x * inv) @ ws.T, x @ w.T, rtol=1e-4, atol=1e-4)


def test_smoothing_helps_w8a8_with_activation_outliers() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(256, 128)
    x[:, [3, 50, 97]] *= 60  # outlier channels, as seen in LLM activations
    ref = x @ w.T

    def w8a8(xx: torch.Tensor, ww: torch.Tensor) -> torch.Tensor:
        return quantize_activations_per_token(xx, 8) @ fake_quantize(ww, QuantSpec(bits=8)).T

    naive = sqnr_db(ref, w8a8(x, w))
    s = smoothing_factors(x.abs().amax(0), w, 0.5)
    ws, inv = smooth_weight(w, s)
    smoothed = sqnr_db(ref, w8a8(x * inv, ws))
    assert smoothed > naive + 6


def test_smoothing_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        smoothing_factors(torch.ones(4), torch.ones(2, 4), alpha=1.5)


# AWQ -----------------------------------------------------------------------------------


def test_awq_never_worse_than_rtn_and_helps_on_salient_channels() -> None:
    x = _correlated_inputs()
    w = torch.randn(64, 128) * 0.02
    res = awq_search(w, x.abs().mean(0), x[:512], group_size=64, n_grid=10)
    assert 0.0 <= res.alpha <= 1.0
    assert res.loss <= res.baseline_loss
    assert res.improvement > 1.05
    assert len(res.grid) == 11
    assert res.scale.shape == (128,)


# GPTQ ----------------------------------------------------------------------------------


def test_gptq_equals_rtn_when_inputs_are_uncorrelated() -> None:
    w = torch.randn(16, 64) * 0.02
    res = gptq_quantize(w, torch.eye(64), group_size=32, block_size=32)
    torch.testing.assert_close(res.qtensor.dequantize(), _rtn(w, 32))


def test_gptq_beats_rtn_on_correlated_inputs() -> None:
    x = _correlated_inputs()
    h = 2 * x.T @ x / x.shape[0]
    w = torch.randn(64, 128) * 0.02
    gptq = gptq_quantize(w, h, group_size=64)
    err_rtn = output_error(w, _rtn(w, 64), h)
    err_gptq = output_error(w, gptq.qtensor.dequantize(), h)
    assert err_gptq < 0.8 * err_rtn
    assert gptq.proxy_loss > 0


def test_gptq_codes_are_valid_and_packable() -> None:
    w = torch.randn(8, 100) * 0.02  # 100 is not a multiple of the group size
    x = torch.randn(512, 100)
    res = gptq_quantize(w, 2 * x.T @ x / 512, group_size=32)
    codes = res.qtensor.codes.to(torch.int32)
    assert codes.min() >= 0 and codes.max() <= 15
    assert res.qtensor.dequantize().shape == w.shape
    lin = nn.Linear(100, 8)
    q = QLinear.from_qtensor(lin, res.qtensor)
    torch.testing.assert_close(q.dequantized_weight(), res.qtensor.dequantize(), rtol=1e-3, atol=1e-6)


def test_gptq_handles_dead_inputs() -> None:
    x = torch.randn(256, 32)
    x[:, 5] = 0.0
    w = torch.randn(4, 32)
    res = gptq_quantize(w, 2 * x.T @ x / 256, group_size=16)
    assert torch.isfinite(res.qtensor.dequantize()).all()
    assert torch.all(res.qtensor.dequantize()[:, 5] == 0)


def test_damped_cholesky_is_upper_and_inverts() -> None:
    a = torch.randn(16, 16)
    h = a @ a.T + torch.eye(16)
    u, dead = damped_inverse_cholesky(h, damp=0.0)
    assert not dead.any()
    assert torch.allclose(u, torch.triu(u))
    torch.testing.assert_close(u.T @ u, torch.linalg.inv(h), rtol=1e-3, atol=1e-4)


# Capture and pipeline ------------------------------------------------------------------


def test_activation_capture_statistics() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    batches = [torch.randn(5, 8) for _ in range(3)]
    stats = collect_activation_stats(model, batches, ["0"], max_samples=4)
    x = torch.cat(batches)
    st = stats["0"]
    assert st.n_rows == 15
    torch.testing.assert_close(st.absmax, x.abs().amax(0))
    torch.testing.assert_close(st.abs_mean, x.abs().mean(0))
    torch.testing.assert_close(st.hessian, 2 * x.T @ x / 15, rtol=1e-5, atol=1e-5)
    assert st.samples.shape == (4, 8)
    assert not model[0]._forward_pre_hooks  # hooks are removed


def test_capture_without_hessian() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    st = collect_activation_stats(model, [torch.randn(2, 4)], ["0"], track_hessian=False)["0"]
    with pytest.raises(RuntimeError):
        _ = st.hessian
    with pytest.raises(TypeError):
        collect_activation_stats(nn.Sequential(nn.ReLU()), [], ["0"])


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_on_tiny_model(method: str) -> None:
    torch.manual_seed(0)
    model = TinyLM(TinyLMConfig(vocab_size=32, dim=32, n_layers=1, n_heads=2, max_len=32)).eval()
    ids = torch.randint(0, 32, (1, 32))
    ref = model(ids)
    scheme, act_bits = ("int8", 8) if method == "smoothquant" else ("int4", None)
    calib = [torch.randint(0, 32, (1, 32)) for _ in range(4)]
    report = calibrate_and_quantize(
        model,
        calib,
        method=method,
        scheme=scheme,
        group_size=32,
        act_bits=act_bits,  # type: ignore[arg-type]
    )
    assert len(report.converted) == 6
    assert report.scheme.startswith(scheme)
    out = model(ids)
    assert F.cosine_similarity(out.flatten(), ref.flatten(), dim=0) > 0.98


def test_pipeline_argument_validation() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="unknown method"):
        calibrate_and_quantize(model, [], method="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gptq supports"):
        calibrate_and_quantize(model, [], method="gptq", scheme="int8")


# act-order GPTQ and block-sequential calibration --------------------------------------


def test_act_order_gptq_equals_rtn_for_identity_hessian() -> None:
    w = torch.randn(16, 64) * 0.02
    h = torch.diag(torch.rand(64) + 0.5)  # diagonal: reorders columns but nothing to compensate
    res = gptq_quantize(w, h, group_size=32, act_order=True)
    torch.testing.assert_close(res.qtensor.dequantize(), _rtn(w, 32))


def test_act_order_gptq_beats_rtn_and_keeps_layout() -> None:
    x = _correlated_inputs()
    h = 2 * x.T @ x / x.shape[0]
    w = torch.randn(64, 128) * 0.02
    res = gptq_quantize(w, h, group_size=64, act_order=True)
    assert output_error(w, res.qtensor.dequantize(), h) < 0.8 * output_error(w, _rtn(w, 64), h)
    q = QLinear.from_qtensor(nn.Linear(128, 64), res.qtensor)
    torch.testing.assert_close(q.dequantized_weight(), res.qtensor.dequantize(), rtol=1e-3, atol=1e-6)


def test_block_grouping() -> None:
    from bitsmith.calib.pipeline import block_of, group_by_block

    assert block_of("model.layers.12.mlp.up_proj") == "model.layers.12"
    assert block_of("lm_head") == ""
    names = ["blocks.0.q", "blocks.0.k", "proj", "blocks.1.q"]
    assert group_by_block(names) == [
        ("blocks.0", ["blocks.0.q", "blocks.0.k"]),
        ("blocks.1", ["blocks.1.q"]),
        ("", ["proj"]),
    ]


def test_stop_after_skips_the_rest_of_the_network() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    calls: list[int] = []
    model[1].register_forward_pre_hook(lambda *_: calls.append(1))
    stats = collect_activation_stats(model, [torch.randn(2, 4)], ["0"], stop_after=model[0])
    assert stats["0"].n_rows == 2
    assert calls == []


@pytest.mark.parametrize("sequential", [True, False])
def test_sequential_and_one_shot_gptq(sequential: bool) -> None:
    torch.manual_seed(0)
    model = TinyLM(TinyLMConfig(vocab_size=32, dim=32, n_layers=2, n_heads=2, max_len=32)).eval()
    ids = torch.randint(0, 32, (1, 32))
    ref = model(ids)
    calib = [torch.randint(0, 32, (1, 32)) for _ in range(4)]
    report = calibrate_and_quantize(model, calib, method="gptq", group_size=32, sequential=sequential)
    assert len(report.converted) == 12
    assert report.skipped == ["lm_head"]
    assert F.cosine_similarity(model(ids).flatten(), ref.flatten(), dim=0) > 0.98
