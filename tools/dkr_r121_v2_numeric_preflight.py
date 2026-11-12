#!/usr/bin/env python3
"""Pure-Python numeric oracle for DKR-R1.21/S6 two-stage paged v2.

Tests the partition online-softmax and log-sum-exp merge identities with
ragged/empty partitions, large logits, multiple values and 0..8193 tokens.
This is NOT a CUDA/PTX or Rust correctness test and never invokes nvcc.
"""

import math
import random
import sys

PARTITION = 256


def reference(logits, values):
    if not logits:
        return [0.0] * 4
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    denom = math.fsum(weights)
    return [
        math.fsum(weight * value[dim] for weight, value in zip(weights, values))
        / denom
        for dim in range(4)
    ]


def partitioned(logits, values, max_context):
    parts = (max_context + PARTITION - 1) // PARTITION
    partial = []
    lse = []
    for part in range(parts):
        lo = part * PARTITION
        hi = min(len(logits), lo + PARTITION)
        if lo >= hi:
            partial.append([0.0] * 4)
            lse.append(float("-inf"))
            continue
        running_max = float("-inf")
        denominator = 0.0
        accumulator = [0.0] * 4
        for logit, value in zip(logits[lo:hi], values[lo:hi]):
            maximum = max(running_max, logit)
            previous = math.exp(running_max - maximum)
            current = math.exp(logit - maximum)
            denominator = denominator * previous + current
            accumulator = [
                accumulator[d] * previous + value[d] * current
                for d in range(4)
            ]
            running_max = maximum
        partial.append([x / denominator for x in accumulator])
        lse.append(running_max + math.log(denominator))

    global_max = max(lse, default=float("-inf"))
    weights = [
        (math.exp(value - global_max) if value != float("-inf") else 0.0)
        for value in lse
    ] if global_max != float("-inf") else [0.0] * parts
    denom = math.fsum(weights)
    return [
        math.fsum(weight * p[d] for weight, p in zip(weights, partial))
        / denom if denom else 0.0
        for d in range(4)
    ]


def main():
    rng = random.Random(610121)
    cases = 0
    for count in (0, 1, 2, 255, 256, 257, 511, 512, 513, 777, 1024, 4096, 8193):
        for amplitude in (1.0, 10.0, 1000.0):
            logits = [rng.uniform(-amplitude, amplitude) for _ in range(count)]
            values = [[rng.uniform(-3.0, 3.0) for _ in range(4)]
                      for _ in range(count)]
            # An actual context may be much shorter than the configured maximum.
            for max_context in (max(count, 257), max(count, 257) + 256):
                expected = reference(logits, values)
                observed = partitioned(logits, values, max_context)
                for d, (want, got) in enumerate(zip(expected, observed)):
                    if not math.isclose(want, got, abs_tol=3e-11, rel_tol=3e-11):
                        raise AssertionError(
                            f"count={count}, amplitude={amplitude}, "
                            f"max_context={max_context}, dim={d}: "
                            f"reference={want}, partitioned={got}"
                        )
                cases += 1
    print(
        f"DKR_R121_V2_NUMERIC_PREFLIGHT=PASS cases={cases} "
        "partitions=256 empty_and_tail=checked "
        "logsumexp=stable no_cuda_compilation"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OverflowError, ValueError) as error:
        print(f"DKR_R121_V2_NUMERIC_PREFLIGHT=FAIL {error}", file=sys.stderr)
        sys.exit(1)
