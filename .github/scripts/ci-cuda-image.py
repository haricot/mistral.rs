#!/usr/bin/env python3
"""Resolve a CUDA environment by its inputs; publish only after its smoke test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
PIN = re.compile(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}\Z")


def identity(root, spec_path):
    spec = json.loads((root / spec_path).read_text())
    if spec.get("schema") != 1 or not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+", spec["image"]):
        raise RuntimeError("invalid image specification")
    if not DIGEST.fullmatch(spec["base_image"].rsplit("@", 1)[-1]):
        raise RuntimeError("base image must be pinned by digest")
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in spec["files"]}
    config = {}
    if "config" in spec:
        source = json.loads((root / spec["config"]["path"]).read_text())
        config = {key: source[key] for key in spec["config"]["keys"]}
    inputs = {"schema": 1, "base_image": spec["base_image"], "files": files,
              "build_args": spec.get("build_args", {}), "config": config,
              "platform": "linux/amd64"}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return spec, key, f'{spec["image"]}:env-{key}'


def docker(*args, capture=False):
    return subprocess.run(["docker", *args], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def registry_digest(reference):
    try:
        value = json.loads(docker("buildx", "imagetools", "inspect", reference,
                                  "--format", "{{json .Manifest.Digest}}", capture=True).stdout)
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or "").lower()
        if any(marker in error for marker in ("manifest unknown", "name unknown", "not found")):
            return None
        raise RuntimeError("registry lookup failed; refusing to rebuild an inaccessible environment: " + error) from exc
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise RuntimeError("invalid registry digest")
    return value


def smoke(root, spec, image):
    docker("run", "--rm", "--network=none", "--entrypoint", "/bin/bash",
           "--volume", f'{root / spec["smoke"]}:/ci-image-smoke.sh:ro', image,
           "/ci-image-smoke.sh")


def ensure(root, spec_path):
    spec, key, tag = identity(root, spec_path)
    current = registry_digest(tag)
    rebuilt = False
    if current is None:
        bootstrap = spec.get("bootstrap", {})
        if bootstrap.get("environment_key") == key:
            candidate = bootstrap["image"]
            if not PIN.fullmatch(candidate):
                raise RuntimeError("invalid bootstrap image digest")
            docker("pull", candidate)
        else:
            candidate = "ci-cuda-environment:candidate"
            args = ["build", "--provenance=false", "--platform", "linux/amd64",
                    "--build-arg", "BASE_IMAGE=" + spec["base_image"],
                    "--label", "org.opencontainers.image.source=https://github.com/" + spec["repository"],
                    "--label", "io.haricot.cuda-environment=" + key,
                    "--file", str(root / spec["recipe"]), "--tag", candidate]
            for name, value in spec.get("build_args", {}).items():
                args += ["--build-arg", f"{name}={value}"]
            docker(*args, str(root / spec["context"]))
            rebuilt = True
        smoke(root, spec, candidate)
        docker("tag", candidate, tag)
        docker("push", tag)
        current = registry_digest(tag)
        if current is None:
            raise RuntimeError("published image cannot be resolved")
    image = spec["image"] + "@" + current
    docker("pull", image)
    smoke(root, spec, image)
    return {"image": image, "environment_key": key, "rebuilt": rebuilt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("identity", "ensure"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "identity":
        spec, key, tag = identity(root, args.spec)
        result = {"environment_key": key, "tag": tag}
    else:
        result = ensure(root, args.spec)
    print(json.dumps(result, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for name, value in result.items():
                stream.write(f"{name}={str(value).lower() if isinstance(value, bool) else value}\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("CUDA_ENVIRONMENT=FAIL " + str(exc), file=sys.stderr)
        sys.exit(1)
