#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import release_r1 as release

ROOT = Path(__file__).resolve().parent
CUDA_ERROR_STUB_LIBRARY = 34
IMAGE_ARCHIVE = "image.tar.zst"


def run(*args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def payload(image, destination):
    container = run("docker", "create", "--network=none", image)
    try:
        subprocess.run(["docker", "cp", container + ":/usr/local/bin/mistralrs", str(destination)], check=True)
    finally:
        subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)


def inspect_payload(image, manifest):
    details = json.loads(run("docker", "image", "inspect", image))[0]
    if (details["Config"].get("User") != "10001:10001"
            or details["Config"].get("Entrypoint") != ["/usr/local/bin/mistralrs"]
            or details["Config"].get("WorkingDir") != "/agent/work"
            or details["Config"].get("Labels", {}).get("org.haricot.binary-sha256") != manifest["binary_sha256"]):
        raise ValueError("runtime image configuration does not match the delivered executable")
    with tempfile.TemporaryDirectory(prefix="release-image-payload-") as directory:
        binary = Path(directory) / "mistralrs"
        payload(image, binary)
        if release.hash_file(binary) != manifest["binary_sha256"]:
            raise ValueError("runtime image contains a different executable")
    return details["Id"]


def loader_smoke(image, directory, output):
    stub = Path(directory) / "loader-only/libcuda.so.1"
    if stub.is_symlink() or not stub.is_file():
        raise ValueError("temporary CUDA link stub is missing")
    driver = ctypes.CDLL(str(stub.resolve()))
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    if driver.cuInit(0) != CUDA_ERROR_STUB_LIBRARY:
        raise ValueError("loader-only image smoke must not use a physical CUDA driver")
    base = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--volume", str(stub.parent.resolve()) + ":/run/driver-link-stub:ro",
            "--env", "LD_LIBRARY_PATH=/run/driver-link-stub"]
    with (Path(output) / "loader-help.txt").open("w") as stream:
        subprocess.run([*base, image, "--help"], check=True, stdout=stream, stderr=subprocess.STDOUT)
    dependencies = subprocess.check_output([*base, "--entrypoint", "/bin/sh", image, "-ec",
        'test "$(id -u)" = 10001; ldd /usr/local/bin/mistralrs'], text=True)
    (Path(output) / "loader-ldd.txt").write_text(dependencies)
    if "not found" in dependencies:
        raise ValueError("runtime image has unresolved libraries")


def build(plan, directory, output):
    manifest = release.inspect(plan, directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tag = "haricot-release-r1:" + manifest["binary_sha256"][:24]
    subprocess.run(["docker", "build", "--label", "org.haricot.binary-sha256=" + manifest["binary_sha256"],
                    "--label", "org.opencontainers.image.revision=" + manifest["candidate"],
                    "-f", str(ROOT / "Containerfile.release-r1"), "-t", tag, str(directory)], check=True)
    image_id = inspect_payload(tag, manifest)
    loader_smoke(image_id, directory, output)
    archive = output / "image.tar"
    subprocess.run(["docker", "save", "--output", str(archive), tag], check=True)
    subprocess.run(["zstd", "-T2", "--rm", str(archive)], check=True)
    record = {"schema": release.IMAGE_SCHEMA, "release": manifest, "image_id": image_id,
              "binary_sha256": manifest["binary_sha256"], "containerfile_sha256": release.hash_file(ROOT / "Containerfile.release-r1"),
              "archive": IMAGE_ARCHIVE, "archive_sha256": release.hash_file(output / IMAGE_ARCHIVE),
              "loader_smoke": "PASS", "gpu_executed": False,
              "loader_help_sha256": release.hash_file(output / "loader-help.txt"),
              "loader_ldd_sha256": release.hash_file(output / "loader-ldd.txt")}
    (output / "oci.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    release.image_record(plan, directory, output, archive=True)
    print("EXTENDED_RELEASE_R1_IMAGE_LOADER=PASS; physical inference uses this same binary SHA256")


def publish(plan, directory, image_directory):
    record = release.image_record(plan, directory, image_directory, archive=True)
    image_directory = Path(image_directory)
    with tempfile.TemporaryDirectory(prefix="release-publish-") as temporary:
        archive = Path(temporary) / "image.tar"
        with archive.open("wb") as stream:
            subprocess.run(["zstd", "-d", "--stdout", str(image_directory / record["archive"])], check=True, stdout=stream)
        subprocess.run(["docker", "load", "--input", str(archive)], check=True)
    if inspect_payload(record["image_id"], record["release"]) != record["image_id"]:
        raise ValueError("qualified image identity changed after transport")
    manifest = record["release"]
    repository = "ghcr.io/haricot/mistralrs-extended-sm61"
    tag = repository + ":r1-" + manifest["candidate"] + "-" + record["image_id"].split(":")[1][:16]
    subprocess.run(["docker", "tag", record["image_id"], tag], check=True)
    output = run("docker", "push", tag)
    digests = re.findall(r"digest:\s*(sha256:[0-9a-f]{64})", output)
    if not digests:
        raise ValueError("registry did not return the published OCI manifest digest")
    result = {"image": repository + "@" + digests[-1], "tag": tag, "image_id": record["image_id"],
              "binary_sha256": manifest["binary_sha256"], "candidate": manifest["candidate"]}
    (image_directory / "published.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("EXTENDED_RELEASE_R1_OCI_PUBLISHED=" + result["image"])
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("Qualified release image: `" + result["image"] + "`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify", "publish"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        release.image_record(args.plan, args.dir, args.image_dir, archive=True)
        print("EXTENDED_RELEASE_R1_IMAGE_REUSE=PASS")
    else:
        (build if args.action == "build" else publish)(args.plan, args.dir, args.image_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit("RELEASE_IMAGE_R1_REJECT=" + str(error))
