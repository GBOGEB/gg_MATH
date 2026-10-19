"""JADE/gg_MATH: exact-source release dependency and visual parity contract."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def classify_release_requirements(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_.-]*)==([A-Za-z0-9_.+-]+)", line)
        if match is None:
            raise ValueError("NON_PINNED_RELEASE_DEPENDENCY")
        name = match.group(1).lower()
        if name in pins:
            raise ValueError("DUPLICATED_RELEASE_DEPENDENCY")
        pins[name] = match.group(2)
    if "matplotlib" not in pins or "plotly" not in pins:
        raise ValueError("MISSING_VISUAL_BRIDGE_DEPENDENCY")
    return pins


class JadeReleaseDependencyContract(unittest.TestCase):
    def test_required_visual_producers_have_exact_pins(self):
        text = (ROOT / "requirements-release.txt").read_text(encoding="utf-8")
        pins = classify_release_requirements(text)
        self.assertEqual(pins["matplotlib"], "3.10.7")
        self.assertIn("plotly", pins)

    def test_missing_or_duplicated_pin_is_rejected(self):
        text = (ROOT / "requirements-release.txt").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "MISSING_VISUAL_BRIDGE"):
            classify_release_requirements(
                "\n".join(line for line in text.splitlines()
                          if not line.startswith("matplotlib=="))
            )
        with self.assertRaisesRegex(ValueError, "DUPLICATED_RELEASE"):
            classify_release_requirements(text + "\nmatplotlib==3.10.7\n")

    def test_exact_head_workflow_installs_canonical_file_before_tests(self):
        yml = (ROOT / ".github/workflows/release-infrastructure-v1.yml").read_text(
            encoding="utf-8"
        )
        install = yml.index("python -m pip install -r requirements-release.txt")
        tests = yml.index("python -m pytest -q")
        self.assertLess(install, tests)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha || github.sha }}", yml)
        self.assertIn("cache-dependency-path: requirements-release.txt", yml)


if __name__ == "__main__":
    unittest.main()
