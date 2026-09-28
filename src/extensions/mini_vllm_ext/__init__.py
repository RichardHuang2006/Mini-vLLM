"""The Metal kernels, built by src/extensions/build.py. Every op here has a pure-MLX
counterpart in mini_vllm that is its test oracle."""

from pathlib import Path

from ._ext import *

load_library(str(Path(__file__).parent))
