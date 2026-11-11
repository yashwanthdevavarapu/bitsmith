# Contributing to bitsmith

Thanks for your interest! Contributions of every size are welcome: a typo fix, a new GPU in
the benchmark table, a new weight format, or a faster kernel.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). By
submitting a pull request you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE), the same license as the project.

## Ways to contribute

| You want to... | Do this |
|----------------|---------|
| Report a bug | Open a [bug report](https://github.com/yashwanthdevavarapu/bitsmith/issues/new?template=bug_report.yml) |
| Propose a format or algorithm | Open a [new format request](https://github.com/yashwanthdevavarapu/bitsmith/issues/new?template=new_format.yml) |
| Share numbers from your GPU | Open a [benchmark result](https://github.com/yashwanthdevavarapu/bitsmith/issues/new?template=benchmark_result.yml) |
| Ask a question | Start a [discussion](https://github.com/yashwanthdevavarapu/bitsmith/discussions) |
| Write code or docs | Pick an issue (below) and open a pull request |
| Report a security problem | Follow [SECURITY.md](SECURITY.md); please do not open a public issue |

## Finding something to work on

- [`good first issue`](https://github.com/yashwanthdevavarapu/bitsmith/labels/good%20first%20issue):
  small, well-scoped tasks with the files to touch already listed.
- [`help wanted`](https://github.com/yashwanthdevavarapu/bitsmith/labels/help%20wanted):
  bigger features where help is especially welcome.
- [ROADMAP.md](ROADMAP.md): what is planned for the next releases, grouped by milestone.

**Before you start, comment on the issue** ("I'd like to take this") so the maintainer can
assign it to you and nobody duplicates work. If you have not opened a pull request within
two weeks, the issue may be offered to someone else; just comment if you need more time.

## Your first contribution, step by step

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/bitsmith.git
   cd bitsmith
   git remote add upstream https://github.com/yashwanthdevavarapu/bitsmith.git
   ```

2. **Set up** a development environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
   pip install -e ".[dev,plot]"
   pre-commit install
   make check                           # everything should pass before you change anything
   ```

3. **Create a branch** named after the issue:

   ```bash
   git switch -c 42-json-output-for-sqnr
   ```

4. **Make the change**, with tests (see the standards below).

5. **Check** locally:

   ```bash
   make check            # ruff, mypy and the full test suite
   ```

   If you touched GPU code and have a CUDA machine, also run `pytest tests/gpu`.

6. **Commit and push** to your fork, then open a pull request against `main`. Put
   `Closes #<issue number>` in the description so the issue closes when it is merged.

7. **Review.** CI runs on Linux, Windows and macOS. The maintainer will review, usually
   within a week. Small follow-up changes are normal; push them to the same branch.

## Everyday commands

| Command | What it does |
|---------|--------------|
| `make check` | lint, format check, mypy, full test suite |
| `make test-fast` | tests without the slow end to end ones (about 3 s) |
| `make format` | auto-fix lint issues and format the code |
| `make coverage` | test suite with an HTML coverage report |
| `make reproduce` | regenerate `results/` and `docs/assets/` (CPU, no downloads) |
| `make reproduce-gpu` | also run the GPU model sweeps and kernel benchmark (CUDA + Triton) |

## Standards

- **Every numeric change comes with a test that pins a number**: an error bound, a known
  byte layout, a bit-exact comparison, or an invariance. "It runs" is not a test.
- Public functions have type hints and a docstring. `mypy` runs in CI.
- Changes to numerics, storage formats or the public API need an ADR in `docs/adr/`
  (copy `docs/adr/template.md`).
- If results change, regenerate them with `make reproduce` and commit the new files.
- GPU-only code must be skipped cleanly on machines without CUDA, like `tests/gpu/`.
- Plain ASCII punctuation in docs and code; CI rejects em and en dashes.
- Add a line to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md) for anything a
  user would notice.

## Adding a weight format

1. Write the quantizer in `src/bitsmith/quant/` and any packing in `src/bitsmith/packing/`.
2. Add a codec class in `src/bitsmith/nn/codecs.py` with `encode` and `decode`, then
   register its name in `SCHEMES` and `make_codec`.
3. Add its minimum SQNR and exact bits per weight to the tables in `tests/unit/test_nn.py`.
   Every parametrized test (round trip, forward, state dict, awkward widths, accounting)
   then covers it automatically.
4. Mention it in the README format list and the technical guide.

## Adding a GPU to the benchmark table

Run `python scripts/bench_kernels.py --json results/kernels_<gpu>.json` on your machine and
open a [benchmark result](https://github.com/yashwanthdevavarapu/bitsmith/issues/new?template=benchmark_result.yml)
issue with the output, or a pull request that adds the JSON file.

## Commit messages

Use the imperative mood and keep the subject under 72 characters, for example
`Add NF4 codec with lookup table decode`. Reference the issue in the body when there is one.

## Labels

| Label | Meaning |
|-------|---------|
| `good first issue` | small and well scoped, good for newcomers |
| `help wanted` | larger work where outside help is welcome |
| `bug` | something gives a wrong result or crashes |
| `enhancement` | a new feature or improvement |
| `performance` | makes something faster or use less memory |
| `documentation` | docs, examples, notebooks |
| `area: kernels`, `area: calibration`, `area: formats`, `area: models`, `area: benchmarks`, `area: cli` | which part of the code |
| `status: needs triage` | new, not reviewed by a maintainer yet |
| `status: in progress` | someone is working on it |

## Releases

The maintainer bumps the version in `pyproject.toml` and `src/bitsmith/__init__.py`, moves
the `Unreleased` changelog entries under the new version, and pushes a `vX.Y.Z` tag. The
release workflow builds the package and publishes the GitHub release.
