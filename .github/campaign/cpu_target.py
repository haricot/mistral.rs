#!/usr/bin/env python3
import argparse
import json
import platform
from pathlib import Path

PROFILES = {"portable": "-C target-cpu=x86-64", "avx2": "-C target-cpu=x86-64-v3"}
AVX2_FEATURES = frozenset("avx avx2 bmi1 bmi2 cx16 f16c fma lahf_lm lzcnt movbe popcnt sse3 sse4_1 sse4_2 ssse3 xsave".split())


def profile(config):
    name = config.get("cpu_profile", "portable")
    if name not in PROFILES:
        raise ValueError("unsupported explicit CPU profile")
    return name


def rustflags(config):
    return PROFILES[profile(config)]


def host_features(text):
    processors = [set(line.split(":", 1)[1].split()) for line in text.splitlines()
                  if line.split(":", 1)[0].strip() == "flags"]
    if not processors:
        raise ValueError("cannot establish the host CPU instruction set")
    for flags in processors:
        for alias, canonical in (("pni", "sse3"), ("abm", "lzcnt"), ("bmi", "bmi1")):
            if alias in flags:
                flags.add(canonical)
    return set.intersection(*processors)


def check_host(config, *, cpuinfo=None):
    if profile(config) == "portable":
        return
    if platform.machine() != "x86_64":
        raise ValueError("AVX2 binaries require an x86_64 host")
    flags = host_features(Path("/proc/cpuinfo").read_text() if cpuinfo is None else cpuinfo)
    missing = AVX2_FEATURES - flags
    if missing:
        raise ValueError("host cannot execute x86-64-v3 binaries; missing " + ",".join(sorted(missing)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--check-host", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.plan.read_text())["config"] if args.plan else {}
    if args.check_host:
        check_host(config)
    print(rustflags(config))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit("CPU_TARGET_REJECT=" + str(error))
