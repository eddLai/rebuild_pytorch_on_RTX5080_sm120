import torch, sys, pprint, pathlib
print("torch path :", pathlib.Path(torch.__file__).resolve())
print("'_C' exists :", (pathlib.Path(torch.__file__).parent / "_C").exists())
pprint.pprint([p for p in sys.path if "pytorch" in p.lower()])

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
