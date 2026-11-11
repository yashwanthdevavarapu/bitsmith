## What and why

<!-- One or two sentences on what this changes and why. -->

Closes #

## How it was tested

<!-- Commands you ran and what they showed. -->

## Checklist

- [ ] `make check` passes (lint, format, mypy, tests)
- [ ] New behaviour has a test that pins a number (error bound, known value or invariance)
- [ ] GPU code was tested with `pytest tests/gpu` on a CUDA machine, or is not touched
- [ ] `results/` regenerated if outputs changed (`make reproduce`)
- [ ] An ADR is added or updated if this changes numerics, storage format or public API
- [ ] A line is added to the `Unreleased` section of `CHANGELOG.md` if users will notice it
