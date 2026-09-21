#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mission" / "LM10" / "w260_prove.py"
SPEC = importlib.util.spec_from_file_location("w260_prove", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load w260_prove.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class W260ProofReceiptTests(unittest.TestCase):
    def test_working_tree_cleanliness_detects_real_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _git("init", cwd=root)
            source = root / "source.txt"
            source.write_text("baseline\n", encoding="utf-8")
            _git("add", "source.txt", cwd=root)
            _git(
                "-c",
                "user.name=Historian Test",
                "-c",
                "user.email=historian@example.invalid",
                "commit",
                "-m",
                "baseline",
                cwd=root,
            )
            self.assertFalse(MOD.git_working_tree_dirty(root))

            source.write_text("mutated\n", encoding="utf-8")
            self.assertTrue(MOD.git_working_tree_dirty(root))


if __name__ == "__main__":
    unittest.main()
