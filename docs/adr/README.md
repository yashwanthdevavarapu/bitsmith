# Architecture Decision Records

Each record captures one decision: the context, what was decided, and what it costs.
Records are immutable once accepted; a later record can supersede an earlier one.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-quantization-conventions.md) | Quantization conventions | Accepted |
| [0003](0003-ternary-storage.md) | Ternary storage: 2-bit default, base-3 optional | Accepted |
| [0004](0004-reference-runtime.md) | Pure PyTorch reference runtime, no custom kernels | Amended by 0010 |
| [0005](0005-one-shot-calibration.md) | One-shot calibration from full precision activations | Superseded by 0009 |
| [0006](0006-synthetic-benchmark.md) | Synthetic corpus and tiny model for CI | Accepted |
| [0007](0007-input-scale-buffer.md) | Explicit input scale instead of folding into norms | Accepted |
| [0008](0008-minimal-dependencies.md) | Minimal runtime dependencies | Accepted |
| [0009](0009-block-sequential-calibration.md) | Block-sequential calibration and act-order GPTQ | Accepted |
| [0010](0010-fused-int4-triton-kernel.md) | Fused INT4 kernel in Triton as an optional backend | Accepted |
| [0011](0011-lookup-and-microscaling-formats.md) | Lookup-table and microscaling formats: NF4, MXFP4, MXINT8 | Accepted |
| [0012](0012-gptq-on-grid-formats.md) | GPTQ on non-uniform grids through a TableCodec interface | Accepted |
| [0013](0013-fast-gptq-loop.md) | Fast GPTQ: transposed two-level loop, CUDA graphs, cheaper Hessian factor | Accepted |
| [0014](0014-int3-int2-packing.md) | INT3 and INT2: asymmetric groups, 24-bit words for INT3 | Accepted |
| [0015](0015-kv-cache-quantization.md) | KV-cache quantization: per-channel keys, per-token values, exact chunked evaluation | Accepted |
| [0016](0016-cuda-graph-decoding.md) | Decode with a static cache and a CUDA graph per token | Accepted |
| [0017](0017-int8-w8a8-backend.md) | Real INT8 W8A8 through torch._int_mm | Accepted |
| [0018](0018-smoothquant-per-layer-search.md) | SmoothQuant chooses per layer, and may choose not to smooth | Accepted |
| [0019](0019-end-to-end-w8a8-prefill.md) | W8A8 measured end to end on prefill, with the input scale fused | Accepted |

New records start from [`template.md`](template.md).
