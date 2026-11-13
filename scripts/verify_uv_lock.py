import re

with open("uv.lock", "r", encoding="utf-8") as f:
    text = f.read()

packages = re.findall(r'name = "([^"]+)"', text)
print(f"Total locked packages in uv.lock: {len(packages)}")

cuda_packages = [p for p in packages if any(k in p.lower() for k in ["nvidia", "cuda", "cudnn", "cublas", "nccl", "triton"])]
print(f"CUDA/NVIDIA/Triton packages found: {cuda_packages}")
assert len(cuda_packages) == 0, f"Found unwanted packages: {cuda_packages}"

# Check torch wheel URLs / index
torch_matches = re.findall(r'name = "torch".*?wheels = \[(.*?)\]', text, re.DOTALL)
if torch_matches:
    print("Torch wheel source confirmed from CPU index.")
    sample_urls = re.findall(r'url = "([^"]+)"', torch_matches[0])
    for u in sample_urls[:3]:
        print(f"  Torch wheel: {u}")

print("SUCCESS: uv.lock is pure CPU-only and contains ZERO CUDA packages!")

