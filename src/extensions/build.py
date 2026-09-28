"""Build the Metal extension in place: uv run python src/extensions/build.py"""

import os
from pathlib import Path

from mlx import extension
from setuptools import Distribution

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    distribution = Distribution(
        {
            "name": "mini_vllm_ext",
            "ext_modules": [extension.CMakeExtension("mini_vllm_ext._ext")],
            "package_data": {"mini_vllm_ext": ["*.so", "*.dylib", "*.metallib"]},
        }
    )
    command = extension.CMakeBuild(distribution)
    command.initialize_options()
    command.build_temp = Path("build")
    command.build_lib = Path("build") / "lib"
    command.inplace = True
    command.ensure_finalized()
    command.run()
