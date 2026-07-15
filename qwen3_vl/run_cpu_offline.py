#!/usr/bin/env python3
"""CPU correctness path: locally dequantize the FP8 checkpoint to FP32."""

import os


# Set offline mode before importing the shared runner (and thus Transformers).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from .qwen3_vl_offline import main


if __name__ == "__main__":
    raise SystemExit(main("cpu"))
