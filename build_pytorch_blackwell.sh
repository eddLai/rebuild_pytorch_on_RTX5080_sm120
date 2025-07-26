#!/usr/bin/env bash
# build_blackwell_pytorch.sh ? make a Blackwell-ready (sm_120) PyTorch wheel
# rev3 ? 2025-07-06

set -euo pipefail

##############################################################################
# 0. GLOBALS
##############################################################################
WORKDIR="${WORKDIR:-$HOME/pytorch_blackwell_build}"
PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
CUDA_ARCH="120"                         # Compute capability 12.0 (sm_120)
DEFAULT_CMAKE_VERSION="3.29.*"
NUM_CORES=$(nproc)
MAX_JOBS="${MAX_JOBS:-$(( NUM_CORES * 70 / 100 ))}"  
WHEEL_OUT="${WHEEL_OUT:-$HOME/Downloads/torch-blackwell}"
SKIP_SM100="${SKIP_SM100:-0}"           # 1 = comment-out all sm_100 code

##############################################################################
# helper
##############################################################################
ask() { local q="$1" d="${2:-y}"
  [[ "${NONINTERACTIVE:-0}" == 1 ]] && { REPLY="$d"; echo "$q ? $d"; return; }
  printf "%s [%s/%s] " "$q" "$(tr yn YN <<<"$d")" "$(tr yn NY <<<"$d")"
  read -r REPLY; REPLY="${REPLY:-$d}"
}
msg() { printf '==> %s\n' "$*"; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 2; }

##############################################################################
# 1. PYTHON BUILD DEPS
##############################################################################
ask "Install/upgrade cmake & ninja?" y
if [[ $REPLY =~ ^[Yy]$ ]]; then
  pip install -qU "cmake==${DEFAULT_CMAKE_VERSION}" ninja \
      pyyaml typing_extensions numpy six requests wheel future
fi
export PATH="$(python - <<'PY'
import os, cmake, sys; print(os.path.dirname(cmake.__file__))
PY
):$PATH"
cmake --version

##############################################################################
# 2. CLONE / UPDATE PYTORCH
##############################################################################
mkdir -p "$WORKDIR"; cd "$WORKDIR"
[[ -d pytorch ]] || { msg "Cloning PyTorch ?"; git clone --recursive "$PYTORCH_REPO" pytorch; }
cd pytorch
ask "Pull latest changes?" y
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git pull --rebase
  git submodule sync --recursive
  git submodule update --init --recursive
fi

##############################################################################
# 2.1 PATCH select_compute_arch.cmake  (enable 12.0 / 120)
##############################################################################
ARCH_CMAKE="cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
if ! grep -q 'STREQUAL "120"' "$ARCH_CMAKE"; then
  msg "Patching select_compute_arch.cmake for 12.0 / 120 ?"
  sed -i '/STREQUAL "Blackwell")/a\
  \  elseif("${arch_name}" STREQUAL "120")\n\
  \    set(arch_bin 12.0)\n\
  \    set(arch_ptx 12.0)\n\
  \  elseif("${arch_name}" STREQUAL "12.0")\n\
  \    set(arch_bin 12.0)\n\
  \    set(arch_ptx 12.0)' "$ARCH_CMAKE"
else
  msg "select_compute_arch.cmake already patched."
fi

##############################################################################
# 2.5 PATCH cuda.cmake  (append 12.0 to TORCH_CUDA_ARCH_LIST)
##############################################################################
CUDA_CMAKE="cmake/public/cuda.cmake"
PATCH_TAG="Blackwell CUDA support begin"
if ! grep -q "$PATCH_TAG" "$CUDA_CMAKE"; then
  msg "Appending 12.0 to TORCH_CUDA_ARCH_LIST in cuda.cmake ?"
  sed -i '/list(APPEND TORCH_CUDA_ARCH_LIST "8.9")/a\
  \  # === Blackwell CUDA support begin ===\n\
  \  list(APPEND TORCH_CUDA_ARCH_LIST "12.0")\n\
  \  # === Blackwell CUDA support end ===' "$CUDA_CMAKE"
else
  msg "cuda.cmake already contains 12.0 entry."
fi

##############################################################################
# 3. OPTIONAL ? COMMENT-OUT ALL sm_100 REFERENCES
##############################################################################
if [[ "$SKIP_SM100" == 1 ]]; then
  msg "Commenting out obsolete sm_100 code ?"
  find third_party/nccl third_party/gloo -type f -print0 | \
      xargs -0 sed -i -e '/\b\(compute_100\|sm_100\)\b/d'
  sed -i '/STREQUAL[[:space:]]*"100a"/,/endif()/s/^/# DISABLED_SM100A: /' cmake/Codegen.cmake
fi

##############################################################################
# 4. CLEAN PREVIOUS BUILD ARTEFACTS (INTERACTIVE)
##############################################################################
ask "Clean previous build artifacts (build/, dist/, *.egg-info)?" y
if [[ $REPLY =~ ^[Yy]$ ]]; then
  msg "Cleaning previous build artifacts ?"
  rm -rf build/ dist/ torch.egg-info
else
  msg "Skipping clean step."
fi

##############################################################################
# 5. ENVIRONMENT FOR BUILD
##############################################################################
export USE_CUDA=1
export MAX_JOBS
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
export TORCH_CUDA_ARCH_LIST="12.0"
export USE_DISTRIBUTED=0
export USE_SYSTEM_NCCL=1
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-$(python - <<'PY'
import sys, os; print(os.path.dirname(os.path.dirname(sys.exec_prefix)))
PY
)}"

##############################################################################
# 6. BUILD
##############################################################################
ask "Build type: wheel (w) or develop (d) ?" w
if [[ $REPLY =~ ^[Dd]$ ]]; then
  BUILD_CMD="python setup.py develop"; INSTALL_WHEEL=0
else
  BUILD_CMD="python setup.py bdist_wheel"; INSTALL_WHEEL=1
fi
msg "Compiling with ${MAX_JOBS} threads ?"
eval "$BUILD_CMD"

##############################################################################
# 7. INSTALL / COPY WHEEL
##############################################################################
pip install --no-deps "numpy>=2.0,<2.1"
if [[ $INSTALL_WHEEL -eq 1 ]]; then
  WHEEL=$(ls dist/torch-*.whl | sort | tail -n1) || die "Wheel not built"
  ask "Install ${WHEEL##*/} into current env?" y
  [[ $REPLY =~ ^[Yy]$ ]] && pip install -q --force-reinstall "$WHEEL"
  cp -u "$WHEEL" "$WHEEL_OUT/"
  msg "Wheel copied to $WHEEL_OUT/"
fi

##############################################################################
# 8. RUNTIME SANITY CHECK
##############################################################################
python - <<'PY'
import torch, time, numpy as np

# --- Environment check -------------------------------------------------
assert torch.cuda.is_available(), "CUDA is not available."
device = torch.device("cuda")
props  = torch.cuda.get_device_properties(device)
assert (props.major, props.minor) == (12, 0), (
    f"Unexpected compute capability: sm_{props.major}{props.minor}"
)
print(f"[ENV ] torch {torch.__version__} | GPU {props.name} (sm_{props.major}{props.minor})")

# --- Numerical correctness (smoke test) --------------------------------
A = torch.randn(1024, 1024, device=device)
B = torch.randn(1024, 1024, device=device)
C = A @ B
ref = A.cpu().numpy() @ B.cpu().numpy()
assert np.allclose(C.cpu().numpy(), ref, atol=1e-4), "GEMM result mismatch."
print("[CHECK] 1024x1024 GEMM matches NumPy")

# --- Performance benchmark --------------------------------------------
N = 4096
A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)

# single run
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = A @ B
torch.cuda.synchronize()
t_single = time.perf_counter() - t0
flops_single = 2 * N**3 / t_single / 1e12  # TFLOPS

# 100 consecutive runs
iters = 100
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    _ = A @ B
torch.cuda.synchronize()
t_total = time.perf_counter() - t0
flops_avg = 2 * N**3 * iters / t_total / 1e12

print(f"[PERF] {N}x{N} GEMM:")
print(f"        single run      : {t_single*1e3:7.2f} ms  (~{flops_single:6.2f} TFLOPS)")
print(f"        avg over {iters} : {flops_avg:6.2f} TFLOPS")
PY


msg "? Blackwell-ready PyTorch build complete."
