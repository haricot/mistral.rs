#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("images", Path(__file__).with_name("ci-cuda-image.py"))
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "Containerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
        (self.root / "config.json").write_text(json.dumps({"toolchain": "1.99.0", "sources": ["old"]}))
        self.config = {"schema": 1, "repository": "test/repo", "image": "ghcr.io/test/repo/ci",
                       "base_image": "ubuntu@sha256:" + "a" * 64,
                       "recipe": "Containerfile", "context": ".", "files": ["Containerfile"],
                       "config": {"path": "config.json", "keys": ["toolchain"]},
                       "smoke": "smoke.sh"}
        self.write_spec()

    def write_spec(self):
        (self.root / "image.json").write_text(json.dumps(self.config))

    def key(self):
        return images.identity(self.root, "image.json")[1]

    def test_only_environment_inputs_change_the_key(self):
        first = self.key()
        (self.root / "config.json").write_text(json.dumps({"toolchain": "1.99.0", "sources": ["new"]}))
        self.assertEqual(first, self.key())
        (self.root / "config.json").write_text(json.dumps({"toolchain": "1.100.0", "sources": ["new"]}))
        self.assertNotEqual(first, self.key())

    def test_base_and_recipe_changes_invalidate_the_environment(self):
        first = self.key()
        self.config["base_image"] = "ubuntu@sha256:" + "b" * 64
        self.write_spec()
        second = self.key()
        self.assertNotEqual(first, second)
        (self.root / "Containerfile").write_text("FROM new-recipe\n")
        self.assertNotEqual(second, self.key())

    def test_mutable_base_is_rejected(self):
        self.config["base_image"] = "ubuntu:latest"
        self.write_spec()
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            self.key()

    def test_two_campaigns_build_only_once_and_use_the_same_digest(self):
        registry = {}
        calls = []
        def command(*args, **kwargs):
            calls.append(args)
            if args[0] == "push":
                registry[args[1]] = "sha256:" + "c" * 64
        with patch.object(images, "registry_digest", side_effect=lambda tag: registry.get(tag)), \
             patch.object(images, "docker", side_effect=command):
            first = images.ensure(self.root, "image.json")
            second = images.ensure(self.root, "image.json")
        self.assertTrue(first["rebuilt"])
        self.assertFalse(second["rebuilt"])
        self.assertEqual(first["image"], second["image"])
        self.assertEqual(sum(c[0] == "build" for c in calls), 1)

    def test_failed_smoke_never_publishes(self):
        calls = []
        def command(*args, **kwargs):
            calls.append(args)
            if args[0] == "run":
                raise subprocess.CalledProcessError(1, args)
        with patch.object(images, "registry_digest", return_value=None), \
             patch.object(images, "docker", side_effect=command):
            with self.assertRaises(subprocess.CalledProcessError):
                images.ensure(self.root, "image.json")
        self.assertFalse(any(c[0] == "push" for c in calls))

    def test_registry_outage_and_denied_access_do_not_trigger_builds(self):
        for message in ("denied", "unauthorized", "connection timed out", "500 server error"):
            error = subprocess.CalledProcessError(1, ["docker"], stderr=message)
            with patch.object(images, "docker", side_effect=error) as mocked:
                with self.assertRaisesRegex(RuntimeError, "refusing to rebuild"):
                    images.ensure(self.root, "image.json")
                self.assertEqual(mocked.call_count, 1)

    def test_only_a_missing_manifest_allows_building(self):
        error = subprocess.CalledProcessError(1, ["docker"], stderr="manifest unknown")
        with patch.object(images, "docker", side_effect=error):
            self.assertIsNone(images.registry_digest("ghcr.io/test/repo:missing"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
