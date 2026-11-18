"""Negative controls for the committed Drive census."""
import json
from pathlib import Path

import pytest

from kernels.drive_census_validation import validate_candidate_census


CENSUS = Path(__file__).resolve().parents[1] / "evidence/dw_spm/drive_census_candidate_20261116.json"


def altered(tmp_path, change):
    doc = json.loads(CENSUS.read_text(encoding="utf-8"))
    change(doc)
    path = tmp_path / "census.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_committed_census_is_quarantined():
    receipt = validate_candidate_census(CENSUS)
    assert receipt["observed_count"] == 25
    assert receipt["distinct_display_names"] == 21
    assert receipt["duplicate_name_groups"] == 2
    assert receipt["authority_transfer"] is False


def test_duplicate_drive_id_fails(tmp_path):
    def mutate(doc):
        doc["entries"][1]["drive_id"] = doc["entries"][0]["drive_id"]
    with pytest.raises(ValueError, match="duplicate Drive ID"):
        validate_candidate_census(altered(tmp_path, mutate))


@pytest.mark.parametrize("field,value", [
    ("role", "SOURCE"), ("destination", "src/module.py"),
    ("approved", True), ("byte_sha256", "0" * 64),
    ("export_mime", "text/plain"),
])
def test_unapproved_promotion_fails(tmp_path, field, value):
    def mutate(doc):
        doc["entries"][0][field] = value
    with pytest.raises(ValueError, match="must not promote"):
        validate_candidate_census(altered(tmp_path, mutate))


def test_wrong_count_fails(tmp_path):
    with pytest.raises(ValueError, match="observed_count"):
        validate_candidate_census(altered(tmp_path, lambda doc: doc.update(observed_count=19)))


def test_wrong_duplicate_classification_fails(tmp_path):
    def mutate(doc):
        doc["entries"][0]["classification"] = "DUPLICATE_BASENAME_UNRESOLVED"
    with pytest.raises(ValueError, match="classification"):
        validate_candidate_census(altered(tmp_path, mutate))
