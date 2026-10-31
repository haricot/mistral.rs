import json
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_image_r1 as image
import release_r1 as release


class ImagePayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "mistralrs"
        self.binary.write_bytes(b"\x7fELFunit-test")
        self.manifest = {"binary_sha256": release.hash_file(self.binary), "candidate": "a" * 40}
        self.details = {"Id": "sha256:" + "b" * 64, "Config": {"User": "10001:10001",
            "Entrypoint": ["/usr/local/bin/mistralrs"], "WorkingDir": "/agent/work",
            "Labels": {"org.haricot.binary-sha256": self.manifest["binary_sha256"]}}}

    def test_image_payload_is_hashed_instead_of_trusting_its_label(self):
        def extract(_image, target): target.write_bytes(self.binary.read_bytes())
        with mock.patch.object(image, "run", return_value=json.dumps([self.details])), mock.patch.object(image, "payload", side_effect=extract):
            self.assertEqual(image.inspect_payload("unit-test", self.manifest), self.details["Id"])

    def test_false_binary_label_does_not_qualify_a_different_payload(self):
        def extract(_image, target): target.write_bytes(b"different executable")
        with mock.patch.object(image, "run", return_value=json.dumps([self.details])), mock.patch.object(image, "payload", side_effect=extract):
            with self.assertRaisesRegex(ValueError, "different executable"): image.inspect_payload("unit-test", self.manifest)

    def test_root_image_is_rejected(self):
        self.details["Config"]["User"] = "0"
        with mock.patch.object(image, "run", return_value=json.dumps([self.details])):
            with self.assertRaises(ValueError): image.inspect_payload("unit-test", self.manifest)

    def loader(self):
        stub = self.root / "loader-only/libcuda.so.1"
        stub.parent.mkdir()
        stub.write_bytes(b"unit-test-stub")
        return stub

    def test_loader_probe_uses_only_a_temporary_readonly_link_stub(self):
        self.loader()
        driver = mock.Mock()
        driver.cuInit.return_value = 34
        with mock.patch.object(image.ctypes, "CDLL", return_value=driver), mock.patch.object(image.subprocess, "run") as execute:
            with mock.patch.object(image.subprocess, "check_output", return_value="libc.so.6 => /lib/libc.so.6"):
                image.loader_smoke("unit-image", self.root, self.root)
        args = execute.call_args.args[0]
        self.assertIn("--network=none", args)
        self.assertIn("--read-only", args)
        self.assertIn("--cap-drop=ALL", args)
        self.assertIn("LD_LIBRARY_PATH=/run/driver-link-stub", args)
        self.assertTrue(any(x.endswith(":/run/driver-link-stub:ro") for x in args))

    def test_loader_probe_cannot_use_a_real_gpu_driver(self):
        self.loader()
        driver = mock.Mock()
        driver.cuInit.return_value = 0
        with mock.patch.object(image.ctypes, "CDLL", return_value=driver), mock.patch.object(image.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "physical CUDA driver"): image.loader_smoke("unit-image", self.root, self.root)
            execute.assert_not_called()

    def test_unresolved_image_libraries_are_rejected(self):
        self.loader()
        driver = mock.Mock()
        driver.cuInit.return_value = 34
        with mock.patch.object(image.ctypes, "CDLL", return_value=driver), mock.patch.object(image.subprocess, "run"):
            with mock.patch.object(image.subprocess, "check_output", return_value="libssl.so.3 => not found"):
                with self.assertRaisesRegex(ValueError, "unresolved libraries"): image.loader_smoke("unit-image", self.root, self.root)

    def test_publication_loads_the_qualified_archive_without_rebuilding(self):
        record = {"image_id": self.details["Id"], "archive": "image.tar.zst", "release": self.manifest}
        published_digest = "sha256:" + "c" * 64
        with mock.patch.object(release, "image_record", return_value=record), mock.patch.object(image, "inspect_payload", return_value=record["image_id"]):
            with mock.patch.object(image.subprocess, "run") as command, mock.patch.object(image, "run", return_value="digest: " + published_digest):
                with contextlib.redirect_stdout(io.StringIO()): image.publish("unit-plan", self.root, self.root)
        calls = [c.args[0] for c in command.call_args_list]
        self.assertTrue(any(c[:2] == ["docker", "load"] for c in calls))
        self.assertFalse(any("build" in c for c in calls))
        result = json.loads((self.root / "published.json").read_text())
        self.assertEqual(result["binary_sha256"], self.manifest["binary_sha256"])
        self.assertEqual(result["image"], "ghcr.io/haricot/mistralrs-extended-sm61@" + published_digest)


class QualificationScopeTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT.parent / "workflows/mistral-extended-sm61.yml").read_text()

    def job(self, name):
        return re.search(r"(?ms)^  " + name + r":\n(.*?)(?=^  [a-z_]+:\n|\Z)", self.workflow).group(1)

    def enabled(self, name, event, ref, qualification_only):
        expression = re.search(r"    if: >-\n(.*?)(?=    [a-z_-]+:)", self.job(name), re.S).group(1)
        expression = expression.replace("github.event_name", "event").replace("github.ref", "ref").replace("inputs.qualification_only", "qualification_only")
        expression = expression.replace("&&", " and ").replace("||", " or ").replace("true", "True").replace("false", "False")
        return eval(" ".join(expression.split()), {"__builtins__": {}}, {"event": event, "ref": ref, "qualification_only": qualification_only})

    def test_branch_qualification_cannot_promote_or_publish(self):
        branch = "refs/heads/ci/extended-release-r1-single-build"
        self.assertTrue(self.enabled("prepare", "workflow_dispatch", branch, True))
        self.assertFalse(self.enabled("prepare", "workflow_dispatch", branch, False))
        for name in ("promote", "release_publish"):
            self.assertFalse(self.enabled(name, "workflow_dispatch", branch, True))
            self.assertFalse(self.enabled(name, "workflow_dispatch", branch, False))
            self.assertFalse(self.enabled(name, "workflow_dispatch", "refs/heads/mirror_orch", True))
            self.assertTrue(self.enabled(name, "workflow_dispatch", "refs/heads/mirror_orch", False))

    def test_unapproved_branch_and_push_never_publish(self):
        self.assertFalse(self.enabled("prepare", "workflow_dispatch", "refs/heads/unapproved", True))
        for name in ("promote", "release_publish"):
            self.assertFalse(self.enabled(name, "push", "refs/heads/mirror_orch", False))
        self.assertIn("release_image", self.job("gpu"))
        self.assertIn("verify-gpu", self.job("release_publish"))


if __name__ == "__main__":
    unittest.main()
