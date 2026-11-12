#!/usr/bin/env python3
"""DKR R1.21/S7 CPU-only KV page layout and copy safety oracle.

Test copy-byte exactness, allowed/disallowed copy dependency graphs and
VLLM K-blocked-x / V-token-strided slot storage. Pure Python; no NVCC,
Cargo, CUDA allocations or changes to CUDA_COMPUTE_CAP.
"""

import itertools
import sys


def independent(mapping):
    sources = set(mapping)
    destinations = set()
    for source, targets in mapping.items():
        for target in targets:
            if target in destinations or (
                target != source and target in sources
            ):
                return False
            destinations.add(target)
    return True


def copy_blocks(pages, mapping):
    result = list(pages)
    for source, destinations in mapping.items():
        for dest in destinations:
            result[dest] = pages[source]
    return result


def reshape(tokens, heads, hd, blocks, block_size, x, slots, key, value):
    total = blocks * heads * hd * block_size
    kcache = [0] * total
    vcache = [0] * total
    for token, slot in enumerate(slots):
        if slot < 0 or slot >= blocks * block_size:
            continue
        block, within = divmod(slot, block_size)
        for h in range(heads):
            for d in range(hd):
                source = (token * heads + h) * hd + d
                key_index = (
                    block * heads * hd * block_size +
                    h * hd * block_size +
                    (d // x) * block_size * x +
                    within * x + d % x
                )
                value_index = (
                    block * heads * hd * block_size +
                    h * hd * block_size +
                    d * block_size + within
                )
                kcache[key_index] = key[source]
                vcache[value_index] = value[source]
    return kcache, vcache


def read_back(heads, hd, block_size, x, slots, key_cache, value_cache):
    keys, values = [], []
    for slot in slots:
        if slot < 0:
            continue
        page, within = divmod(slot, block_size)
        for h in range(heads):
            for d in range(hd):
                key_index = (
                    page * heads * hd * block_size +
                    h * hd * block_size +
                    (d // x) * block_size * x +
                    within * x + d % x
                )
                value_index = (
                    page * heads * hd * block_size +
                    h * hd * block_size +
                    d * block_size + within
                )
                keys.append(key_cache[key_index])
                values.append(value_cache[value_index])
    return keys, values


def main():
    copy_cases = [
        ({0: [2], 1: [3]}, True),
        ({0: [0], 1: [3]}, True),
        ({0: [1], 1: [3]}, False),
        ({0: [2], 1: [2]}, False),
        ({0: [1], 1: [0]}, False),
        ({0: [2, 3], 1: []}, True),
    ]
    for mapping, expected in copy_cases:
        assert independent(mapping) == expected, mapping

    byte_cases = 0
    for bytes_per_element, stride in itertools.product((1, 2, 4), (3, 16, 65)):
        count = bytes_per_element * stride
        pages = [bytes((i * 17 + j * 3) % 256 for j in range(count))
                 for i in range(4)]
        result = copy_blocks(pages, {0: [2], 1: [3]})
        assert result == [pages[0], pages[1], pages[0], pages[1]]
        assert pages[0] == result[2] and pages[1] == result[3]
        byte_cases += 1

    reshape_cases = 0
    for heads, hd, x, block_size in itertools.product(
        (1, 2), (8, 16), (1, 2, 4, 8), (2, 4)
    ):
        slots = [block_size + 1, -1, 0, block_size]
        blocks = 3
        count = len(slots) * heads * hd
        keys = [(n * 7 + 1) % 251 for n in range(count)]
        values = [(n * 5 + 3) % 253 for n in range(count)]
        kcache, vcache = reshape(
            len(slots), heads, hd, blocks, block_size, x,
            slots, keys, values
        )
        output_key, output_value = read_back(
            heads, hd, block_size, x, slots, kcache, vcache
        )
        reference_keys = []
        reference_values = []
        for token, slot in enumerate(slots):
            if slot < 0:
                continue
            start = token * heads * hd
            end = start + heads * hd
            reference_keys.extend(keys[start:end])
            reference_values.extend(values[start:end])
        assert output_key == reference_keys, (heads, hd, x, block_size)
        assert output_value == reference_values, (heads, hd, x, block_size)
        reshape_cases += 1
    print(
        f"DKR_R121_S7_CPU_PREFLIGHT=PASS copy_dependencies={len(copy_cases)} "
        f"copy_byte_cases={byte_cases} reshape_layout_cases={reshape_cases} "
        "no_cuda_compilation"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, OverflowError) as error:
        print(f"DKR_R121_S7_CPU_PREFLIGHT=FAIL {error}", file=sys.stderr)
        sys.exit(1)
