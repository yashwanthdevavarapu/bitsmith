# Security policy

## Supported versions

Security fixes go into the latest release only.

| Version | Supported |
|---------|-----------|
| 0.2.x | Yes |
| below 0.2 | No |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report them privately through GitHub:
[Security > Report a vulnerability](https://github.com/yashwanthdevavarapu/bitsmith/security/advisories/new).

Include what you found, how to reproduce it, and the impact you expect. You will get an
acknowledgement within 7 days and a plan or fix as soon as one is ready. Once a fix is
released, you are welcome to be credited in the advisory.

## Scope notes

- **Loading checkpoints.** `torch.load` can run arbitrary code from untrusted files. The
  test suite uses `weights_only=True`; do the same in your own code, and only load
  quantized checkpoints you trust.
- **Model downloads.** `bitsmith hf` downloads models and datasets from the Hugging Face Hub
  through `transformers` and `datasets`. Issues in those libraries should be reported to
  their maintainers.
