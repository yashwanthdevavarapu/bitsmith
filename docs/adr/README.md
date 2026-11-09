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

New records start from [`template.md`](template.md).
