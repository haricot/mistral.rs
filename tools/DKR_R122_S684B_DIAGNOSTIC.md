# S6.8.4-B.1 — Offline differential diagnosis

S684B c796899 executed all three legs successfully, but the numerical gate
failed (RMS 0.1686913446, max 1.00059509, cosine 0.99906416575, same argmax).
The original gate reports only its first failing comparison, without identifying
prompt or leg. Do not relax thresholds or rebuild. Evaluate all ten pairs of
the five configurations, including the intermediate typed-only contiguous
configuration from S684A. Cross-cache comparisons are diagnostic (not identical
execution contracts). Record each repetition and each prompt. Check raw
outputs independently of generation; log first differing generated token.
Audit actual typed adapter success counts, native routing and cache allocation
separately. The absence of a module-load trace alone is not proof of no
execution; use telemetry. The original archives and report remain unchanged.

The static build recompiled 54 quant and 15 core kernels although the paged
cache hit 15/15; do not assume cache reuse across different Cargo feature sets.
Preserve both binaries; isolate build caches before future feature switches.
Run the offline diagnostic with explicit prior and run directories, and
a fresh --output. An exit of zero means the diagnostic completed, not
qualification. Production approval is always false. The original numerical
thresholds are preserved. No new GPU execution or compilation.
