"""Benchmark a Hugging Face model on WikiText-2 (needs: pip install -e '.[hf]').

Run: python examples/04_real_model.py --model HuggingFaceTB/SmolLM2-135M --eval-tokens 4096
"""

from __future__ import annotations

import sys

from bitsmith.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["hf", *sys.argv[1:]]))
