"""
FP8 MPS Metal support for Apple Silicon.

This module provides FP8 (8-bit floating point) support for Apple's MPS (Metal Performance Shaders)
backend, enabling ComfyUI to run FP8-quantized models (like FLUX and SD3.5) on Apple Silicon.

The implementation uses custom Metal compute shaders to perform FP8 dequantization and matrix
multiplication directly on the GPU, working around PyTorch's lack of native FP8 support on MPS.

Source: https://github.com/tashiscool/fp8-mps-metal
"""

from .fp8_mps_patch import install, uninstall, is_installed

__all__ = ['install', 'uninstall', 'is_installed']
