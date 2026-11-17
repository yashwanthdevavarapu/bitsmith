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


@pytest.mark.parametrize("case", ["int4", "int4-act-order", "nf4", "mxfp4"])
def test_gptq_cuda_graph_matches_eager(case: str) -> None:
    x = torch.randn(1024, 200, device=CUDA)  # 200 columns: the last inner block is partial
    h = 2 * x.T @ x / 1024
    w = torch.randn(48, 200, device=CUDA) * 0.02
    if case.startswith("int4"):
        act = case.endswith("order")
        a = gptq_quantize(w, h, group_size=32, act_order=act, use_cuda_graph=True).qtensor
        b = gptq_quantize(w, h, group_size=32, act_order=act, use_cuda_graph=False).qtensor
        assert torch.equal(a.codes, b.codes) and torch.equal(a.scale, b.scale)
    else:
        codec = make_codec(case, 32)
        ga = gptq_quantize_table(w, h, codec, use_cuda_graph=True)  # type: ignore[arg-type]
        gb = gptq_quantize_table(w, h, codec, use_cuda_graph=False)  # type: ignore[arg-type]
        assert torch.equal(ga.weight, gb.weight)


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


@pytest.mark.parametrize("backend", ["none", "reference", "triton"])
def test_graph_decode_matches_eager(backend: str) -> None:
    """Replaying the captured decode step gives the same tokens as running it eagerly."""
    transformers = pytest.importorskip("transformers")
    from bitsmith.decode import GraphDecoder

    if backend == "triton" and not triton_available():
        pytest.skip("needs Triton")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    m = transformers.LlamaForCausalLM(cfg).to(CUDA, torch.float16).eval()
    if backend != "none":
        quantize_model(m, scheme="int4", group_size=64, exclude=["lm_head"])
        set_backend(m, backend)
    prompt = torch.randint(0, 128, (1, 9), device=CUDA)
    eager = GraphDecoder(m, 64, use_graph=False).generate(prompt, 20)
    graph = GraphDecoder(m, 64, use_graph=True)
    assert torch.equal(graph.generate(prompt, 20), eager)
    assert torch.equal(graph.generate(prompt, 20), eager)  # replay again after a reset


@needs_triton
@pytest.mark.parametrize("rows", [1, 16, 17, 300])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_int8_fused_matches_pytorch_path(rows: int, dtype: torch.dtype) -> None:
    from bitsmith.kernels.int8_mm import _quantize_rows_triton, int8_linear, quantize_rows_int8

    torch.manual_seed(0)
    x = (torch.randn(rows, 1536, device=CUDA) * torch.rand(1536, device=CUDA) * 4).to(dtype)
    codes_ref, scale_ref = quantize_rows_int8(x)
    codes, scale = _quantize_rows_triton(x)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    assert torch.equal(codes[:rows], codes_ref)  # same rounding, half to even
    assert not codes[rows:].any()  # padding rows are zero
    q = QLinear.from_linear(nn.Linear(1536, 512, device=CUDA, dtype=dtype), "int8", act_bits=8)
    bufs = q.packed_buffers()
    fused = int8_linear(x, bufs["qweight"], bufs["scale"], q.bias, fused=True)
    plain = int8_linear(x, bufs["qweight"], bufs["scale"], q.bias, fused=False)
    assert fused.dtype == dtype
    torch.testing.assert_close(fused.float(), plain.float(), rtol=1e-2, atol=1e-2)


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_int8_fused_input_scale_matches_separate_multiply(dtype: torch.dtype) -> None:
    from bitsmith.kernels.int8_mm import _quantize_rows_triton, quantize_rows_int8

    torch.manual_seed(0)
    x = (torch.randn(40, 1536, device=CUDA) * torch.rand(1536, device=CUDA) * 4).to(dtype)
    s = (torch.rand(1536, device=CUDA) + 0.25).to(torch.float16)
    codes, scale = _quantize_rows_triton(x, s)
    codes_ref, scale_ref = quantize_rows_int8(x * s.to(dtype))
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    assert torch.equal(codes[:40], codes_ref)
