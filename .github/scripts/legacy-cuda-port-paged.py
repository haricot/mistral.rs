#!/usr/bin/env python3
"""Exact replay of historical 2026 Pascal BF16 paged-attention fixes.

Only the three files changed by commit adc54c5 are handled. Preserve upstream
CUDA/FlashInfer headers and use the software BF16 operations on sm_61.
"""
from pathlib import Path
import subprocess

COMMIT = "adc54c5bdbd69b0e738241b78357919af9b0c6e0"
PATHS = {
    "mistralrs-paged-attn/src/cuda/attention/dtype_bfloat16.cuh",
    "mistralrs-paged-attn/src/cuda/backend/paged_attention.rs",
    "mistralrs-paged-attn/src/cuda/flashinfer/utils.cuh",
}

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def require(test, message):
    if not test:
        raise RuntimeError(message)

require(git("rev-parse", "REBASE_HEAD") == COMMIT, "Unexpected paged-attn commit")
conflicts = set(git("diff", "--name-only", "--diff-filter=U").splitlines())
require(bool(conflicts) and conflicts <= PATHS,
        f"Unexpected conflict set: {sorted(conflicts)}")
# Use the last committed rebase state for ALL files: it includes modern
# upstream plus earlier ports. This also resets a cleanly auto-merged portion
# of the historical patch before semantic replay, avoiding duplicate flags.
for path in sorted(PATHS):
    Path(path).write_text(subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True))

header = Path("mistralrs-paged-attn/src/cuda/attention/dtype_bfloat16.cuh")
text = header.read_text()
signatures = [
    "inline __device__ float2 bf1622float2(",
    "inline __device__ __nv_bfloat162 bf162bf162(",
    "inline __device__ __nv_bfloat16 add(",
    "inline __device__ __nv_bfloat162 add(",
    "inline __device__ __nv_bfloat16 mul(",
    "inline __device__ __nv_bfloat162 mul(",
    "inline __device__ __nv_bfloat162 fma(__nv_bfloat162",
    "inline __device__ __nv_bfloat162 fma(__nv_bfloat16",
    "inline __device__ void from_float(__nv_bfloat162&",
    "inline __device__ void from_float(bf16_4_t&",
    "inline __device__ void from_float(bf16_8_t&",
    "inline __device__ void zero(__nv_bfloat16&",
]
positions = [text.find(sig) for sig in signatures]
require(all(pos >= 0 for pos in positions) and positions == sorted(positions),
        "BF16 helper ordering changed")
require(text.count("  assert(false);") == 12, "Not the expected twelve BF16 SM61 guards")
software = [
    """  float2 result;
  result.x = __bfloat162float(val.x);
  result.y = __bfloat162float(val.y);
  return result;""",
    "  return __nv_bfloat162{val, val};",
    "  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));",
    """  return __nv_bfloat162{
      __float2bfloat16(__bfloat162float(a.x) + __bfloat162float(b.x)),
      __float2bfloat16(__bfloat162float(a.y) + __bfloat162float(b.y))};""",
    "  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));",
    """  return __nv_bfloat162{
      __float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x)),
      __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y))};""",
    """  return __nv_bfloat162{
      __float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(c.x)),
      __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y) + __bfloat162float(c.y))};""",
    "  return fma(bf162bf162(a), b, c);",
    """  dst.x = __float2bfloat16(src.x);
  dst.y = __float2bfloat16(src.y);""",
    """  from_float(dst.x, src.x);
  from_float(dst.y, src.y);""",
    """  from_float(dst.x, src.x);
  from_float(dst.y, src.y);
  from_float(dst.z, src.z);
  from_float(dst.w, src.w);""",
    "  dst = __float2bfloat16(0.0f);",
]
require(len(signatures) == len(software), "Missing software BF16 replacement")
for old, replacement in zip(signatures, software):
    start = text.index(old)
    limit = text.index("#else", start)
    section = text[start:limit]
    require(section.count("  assert(false);") == 1, f"{old}: architecture guard changed")
    text = text[:start] + section.replace("  assert(false);", replacement, 1) + text[limit:]
require("assert(false);" not in text, "Unexpected unported BF16 assertion")
header.write_text(text)

# FP8 scale pointer lifetime/validity: only interpret scales for FP8 cache.
backend = Path("mistralrs-paged-attn/src/cuda/backend/paged_attention.rs")
text = backend.read_text()
first = "        let (k_scale_ptr, v_scale_ptr) =\n"
a = text.find(first)
require(a >= 0 and text.count(first) == 1, "Unexpected paged-attention scale structure")
b = text.find("\n\n        let sinks_ptr", a)
require(b > a, "Missing paged-attention sinks marker")
part = text[a:b]
require(part.endswith("            };"), "Unexpected paged scale closure")
part = part.replace(
    "        let (k_scale_ptr, v_scale_ptr) =\n"
    "            if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {",
    "        let (k_scale_ptr, v_scale_ptr) = if cache_dtype == 3 {\n"
    "            if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {",
    1,
)
require("if cache_dtype == 3 {\n" in part, "Could not guard paged FP8 scales")
part = part[:-len("            };")] + (
    "            }\n"
    "        } else {\n"
    "            (std::ptr::null(), std::ptr::null())\n"
    "        };"
)
text = text[:a] + part + text[b:]
second = "    let (k_scale_ptr, v_scale_ptr) = if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {"
require(text.count(second) == 1, "Unexpected cache-update scale structure")
a = text.index(second)
b = text.index("\n\n    let (num_tokens", a)
part = text[a:b]
require(part.endswith("    };"), "Unexpected update-cache closure")
part = part.replace(second,
    "    let (k_scale_ptr, v_scale_ptr) = if cache_dtype == 3 {\n"
    "        if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {", 1)
part = part[:-len("    };")] + (
    "        }\n"
    "    } else {\n"
    "        (std::ptr::null(), std::ptr::null())\n"
    "    };"
)
text = text[:a] + part + text[b:]
backend.write_text(text)

utils = Path("mistralrs-paged-attn/src/cuda/flashinfer/utils.cuh")
text = utils.read_text()
anchor = '#include "exception.h"\n'
require(text.count(anchor) == 1 and "__grid_constant__" not in text,
        "Unexpected FlashInfer __grid_constant__ macro layout")
text = text.replace(anchor,
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)\n"
    "#undef __grid_constant__\n"
    "#define __grid_constant__\n"
    "#endif\n\n" + anchor, 1)
utils.write_text(text)

for path in sorted(PATHS):
    subprocess.run(["git", "add", path], check=True)
subprocess.run(["git", "diff", "--cached", "--check"], check=True)
print("PORT_PAGED_R4_READY: 12 BF16 ops, FP8 cache scales and FlashInfer SM61 guard")
