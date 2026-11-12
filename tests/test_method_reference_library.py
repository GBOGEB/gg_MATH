"""References bind existing method IDs to public sources and source tests."""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_build", ROOT / "release/build_release.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
LIBRARY = ROOT / "research/MATH_METHOD_REFERENCES_v1.yaml"


def test_reference_bindings_use_existing_source_and_challenges():
    library = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    assert len(library["references"]) >= 8
    for entry in library["references"]:
        for field in ("kernel", "challenge"):
            assert (ROOT / entry[field]).is_file(), entry[field]
        assert entry["assumptions"] and entry["boundary"]
    _, cards = build.load_methods(ROOT)
    rendered = build.reference_library(ROOT, cards, "a" * 40)
    assert "PROJECT_GOVERNED" not in rendered  # no new threshold vocabulary


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(authority_transfer=True),
    lambda data: data.update(formal_credit_delta=False),
    lambda data: data["references"][0].update(methods=["INVENTED"]),
    lambda data: data["references"][0].update(url="javascript:alert(1)"),
    lambda data: data["references"][0].update(kernel="kernels/../private.py"),
])
def test_library_rejects_authority_or_source_binding_mutations(tmp_path, mutation):
    data = copy.deepcopy(yaml.safe_load(LIBRARY.read_text(encoding="utf-8")))
    mutation(data)
    (tmp_path / "research").mkdir()
    (tmp_path / "research/MATH_METHOD_REFERENCES_v1.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    _, cards = build.load_methods(ROOT)
    with pytest.raises(ValueError):
        build.reference_library(tmp_path, cards, "a" * 40)


def test_reference_text_is_escaped_in_html(tmp_path):
    data = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    data["references"][0]["citation"] = '<script>alert("x")</script>'
    (tmp_path / "research").mkdir()
    (tmp_path / "research/MATH_METHOD_REFERENCES_v1.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    _, cards = build.load_methods(ROOT)
    rendered = build.reference_library(tmp_path, cards, "a" * 40)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
