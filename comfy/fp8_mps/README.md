# FP8 MPS Metal Support

This directory contains the FP8 MPS Metal patch that enables FP8 model support on Apple Silicon.

## Overview

PyTorch's MPS backend does not natively support FP8 (8-bit floating point) operations, which prevents modern FP8-quantized models like FLUX and SD3.5 from running on Apple Silicon. This module provides custom Metal compute shaders that implement:

- FP8 dequantization (e4m3fn format)
- FP8 scaled matrix multiplication
- Float to FP8 quantization

These operations are exposed through a monkey-patch of `torch._scaled_mm`, making them transparent to existing code.

## Files

- `__init__.py` - Module initialization and public API
- `fp8_mps_patch.py` - Monkey-patch for torch._scaled_mm
- `fp8_mps_native.py` - Native Metal kernel interface using PyTorch's compile_shader API
- `fp8_matmul.metal` - Metal compute shaders for FP8 operations

## Usage

The patch is automatically installed when ComfyUI detects MPS is available. You can verify it's active by checking the logs for:
```
FP8 MPS patch installed - FLUX/SD3.5 FP8 models should now work on Apple Silicon
```

## Source

This integration is based on [tashiscool/fp8-mps-metal](https://github.com/tashiscool/fp8-mps-metal).

## License

The core fp8-mps-metal code (fp8_mps_patch.py, fp8_mps_native.py, fp8_matmul.metal) is licensed under the MIT License. See the LICENSE file in this directory for details. The integration code (__init__.py) is part of ComfyUI and licensed under GPLv3.
