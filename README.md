# Blackwell-Ready PyTorch Build Script

## Summary

This script builds a custom **PyTorch wheel with Compute Capability 12.0 (`sm_120`)** support for Blackwell GPUs (e.g. RTX 5080).  
It patches required CMake logic to allow successful CUDA 12.x builds targeting `sm_120`.

## Output

- ✅ A PyTorch `.whl` package with `TORCH_CUDA_ARCH_LIST=12.0`
- ✅ Runtime validated on Blackwell GPU via matrix multiplication and TFLOPS test
- ✅ Optional install into current Python environment
- ✅ Copies the final wheel to:  
  - `$WHEEL_OUT` (default: `~/Downloads/torch-blackwell`)

## Generated Folders

| Path                         | Purpose                                 |
|------------------------------|------------------------------------------|
| `$WORKDIR` (default: `~/pytorch_blackwell_build`) | Build workspace with cloned PyTorch repo |
| `$WORKDIR/pytorch/dist/`     | Local output folder containing `.whl` build artifacts |
| `$WHEEL_OUT`                 | Final wheel copy location                |

You may override these via environment variables before execution.

## Modifications

- Patches:
  - `select_compute_arch.cmake` — adds support for `arch_bin` and `arch_ptx` 12.0
  - `cuda.cmake` — appends `12.0` to `TORCH_CUDA_ARCH_LIST`
- Optional: removal of all `sm_100` references (set `SKIP_SM100=1`)  
  ⚠️ *This is generally unnecessary if building with selected libraries only.*

## Tested Environment

- **CUDA 12.8**
- **NVIDIA RTX 5080 OC (sm_120)**
- **PyTorch main branch (rev: 2025-07)**

## Reference

Based on:  
https://github.com/kentstone84/pytorch-rtx5080-support

