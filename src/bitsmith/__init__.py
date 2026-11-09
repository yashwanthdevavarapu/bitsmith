"""bitsmith: LLM weight quantization from first principles in pure PyTorch.

Subpackages
-----------
quant    uniform integer, FP8 and ternary quantizers
packing  INT4 nibbles, ternary (2-bit and base-3) and 2:4 sparse storage
nn       QLinear and in-place model conversion
calib    SmoothQuant, AWQ and GPTQ calibration
eval     error metrics, sliding-window perplexity, synthetic corpus, tiny LM
bench    Pareto sweep, decode roofline, Hugging Face evaluation
"""

from bitsmith.calib import calibrate_and_quantize
from bitsmith.nn import SCHEMES, QLinear, quantize_model
from bitsmith.quant import QuantSpec, fake_quantize, quantize

__version__ = "0.2.0"

__all__ = [
    "SCHEMES",
    "QLinear",
    "QuantSpec",
    "__version__",
    "calibrate_and_quantize",
    "fake_quantize",
    "quantize",
    "quantize_model",
]
