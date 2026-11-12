"""GPU tests: every format on CUDA matches CPU, and the fused Triton kernel matches the
reference runtime. Skipped automatically on machines without CUDA."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitsmith.calib import calibrate_and_quantize, gptq_quantize, gptq_quantize_table
from bitsmith.eval import TinyLM, TinyLMConfig, sqnr_db
from bitsmith.kernels import int4_linear, triton_available
from bitsmith.nn import SCHEMES, QLinear, make_codec, quantize_model, set_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CUDA = torch.device("cuda")


@pytest.mark.parametrize("scheme", SCHEMES)
def test_codecs_match_cpu(scheme: str) -> None:
    w = torch.randn(64, 200) * 0.02
    codec = make_codec(scheme, 64)
    cpu = codec.decode(codec.encode(w), 64, 200)
    gpu = codec.decode(codec.encode(w.to(CUDA)), 64, 200)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-5, atol=1e-7)


def test_gptq_on_gpu_matches_cpu() -> None:
    x = torch.randn(1024, 128)
    h = 2 * x.T @ x / 1024
    w = torch.randn(32, 128) * 0.02
    cpu = gptq_quantize(w, h, group_size=64, act_order=True).qtensor.dequantize()
    gpu = gptq_quantize(w.to(CUDA), h.to(CUDA), group_size=64, act_order=True).qtensor.dequantize()
    assert sqnr_db(cpu, gpu.cpu()) > 40


@pytest.mark.parametrize("scheme", ["nf4", "mxfp4"])
def test_table_gptq_on_gpu_matches_cpu(scheme: str) -> None:
    x = torch.randn(1024, 128)
    h = 2 * x.T @ x / 1024
    w = torch.randn(32, 128) * 0.02
    codec = make_codec(scheme, 64)
    cpu = gptq_quantize_table(w, h, codec).weight  # type: ignore[arg-type]
    gpu = gptq_quantize_table(w.to(CUDA), h.to(CUDA), codec).weight  # type: ignore[arg-type]
    assert sqnr_db(cpu, gpu.cpu()) > 40


def test_calibration_pipeline_on_gpu() -> None:
    torch.manual_seed(0)
    model = TinyLM(TinyLMConfig(vocab_size=32, dim=64, n_layers=2, n_heads=2, max_len=32)).to(CUDA).eval()
    ids = torch.randint(0, 32, (1, 32), device=CUDA)
    ref = model(ids)
    calib = [torch.randint(0, 32, (1, 32), device=CUDA) for _ in range(4)]
    for method in ("awq", "gptq"):
        m = TinyLM(model.cfg).to(CUDA).eval()
        m.load_state_dict(model.state_dict())
        calibrate_and_quantize(m, calib, method=method, group_size=32)  # type: ignore[arg-type]
        assert F.cosine_similarity(m(ids).flatten(), ref.flatten(), dim=0) > 0.98


needs_triton = pytest.mark.skipif(not triton_available(), reason="needs Triton")


@needs_triton
@pytest.mark.parametrize("rows", [1, 3, 8, 9, 64, 300])
@pytest.mark.parametrize("scheme", ["int4", "int4-sym"])
@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_kernel_matches_reference(rows: int, scheme: str, group: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    lin = nn.Linear(320, 200, bias=True).to(CUDA, dtype)
    q = QLinear.from_linear(lin, scheme, group)
    x = torch.randn(rows, 320, device=CUDA, dtype=dtype)
    ref = q(x)
    q.set_backend("triton")
    out = q(x)
    assert out.dtype == dtype
    assert sqnr_db(ref.float(), out.float()) > 35


@needs_triton
def test_triton_kernel_handles_padding_and_batches() -> None:
    lin = nn.Linear(100, 48).to(CUDA, torch.float16)  # 100 is not a multiple of the group
    q = QLinear.from_linear(lin, "int4", 64)
    x = torch.randn(2, 5, 100, device=CUDA, dtype=torch.float16)
    ref = q(x)
    q.set_backend("triton")
    assert q(x).shape == (2, 5, 48)
    assert sqnr_db(ref.float(), q(x).float()) > 35


@needs_triton
def test_triton_model_end_to_end() -> None:
    torch.manual_seed(0)
    model = TinyLM(TinyLMConfig(vocab_size=64, dim=128, n_layers=2, n_heads=4, max_len=64)).to(CUDA, torch.float16)
    quantize_model(model, "int4", group_size=64)
    ids = torch.randint(0, 64, (1, 64), device=CUDA)
    ref = model(ids)
    assert set_backend(model, "triton") == 12
    assert sqnr_db(ref.float(), model(ids).float()) > 30


@needs_triton
def test_triton_kernel_validation() -> None:
    q = QLinear.from_linear(nn.Linear(64, 32).to(CUDA), "int4", 64)
    bufs = q.packed_buffers()
    with pytest.raises(TypeError, match="fp16 or bf16"):
        int4_linear(torch.randn(1, 64, device=CUDA), bufs["qweight"], bufs["scale"], bufs["zero"], 64, 64)
    with pytest.raises(ValueError, match="power of two"):
        int4_linear(torch.randn(1, 64, device=CUDA).half(), bufs["qweight"], bufs["scale"], bufs["zero"], 64, 48)
