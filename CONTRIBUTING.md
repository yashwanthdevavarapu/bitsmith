# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,plot]"
pre-commit install
```

## Everyday commands

| Command | What it does |
|---------|--------------|
| `make check` | lint, format check, mypy, full test suite |
| `make test-fast` | tests without the slow end to end ones (about 3 s) |
| `make coverage` | test suite with an HTML coverage report |
| `make reproduce` | regenerate `results/` and `docs/assets/` (CPU, no downloads) |
| `make reproduce-gpu` | also run the GPU model sweeps and kernel benchmark (CUDA + Triton) |

## Standards

- **Every numeric change comes with a test that pins a number**: an error bound, a known
  byte layout, a bit-exact comparison, or an invariance. "It runs" is not a test.
- Public functions have type hints and a docstring. `mypy` runs in strict-ish mode in CI.
- Changes to numerics, storage formats or the public API need an ADR in `docs/adr/`.
- If results change, regenerate them with `make reproduce` and commit the new files.
- Plain ASCII punctuation in docs and code; CI rejects em and en dashes.

## Adding a weight format

1. Write the quantizer in `src/bitsmith/quant/` and any packing in `src/bitsmith/packing/`.
2. Add a codec class in `src/bitsmith/nn/codecs.py` with `encode` and `decode`, then
   register its name in `SCHEMES` and `make_codec`.
3. Add its minimum SQNR and exact bits per weight to the tables in `tests/unit/test_nn.py`.
   Every parametrized test (round trip, forward, state dict, awkward widths, accounting)
   then covers it automatically.

## Commit messages

Use the imperative mood and keep the subject under 72 characters, for example
`Add NF4 codec with lookup table decode`.
