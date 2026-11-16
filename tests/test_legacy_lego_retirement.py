import json

import pytest

from scripts.audit_legacy_lego_dependencies import FILES, digest
from scripts.retire_legacy_lego_directory import retire
from scripts.stage_training_reset_closure import stage
from scripts.relocate_legacy_lego_directory import relocate


def test_training_closure_is_staged_under_exact_canonical_parent_and_hash_checked(tmp_path):
    root = tmp_path / "lego"
    parent = root / "data/parent"
    child = root / "data/child"
    target = root / "piles/pile_0001"
    for path in (parent, child, target):
        path.mkdir(parents=True)
        for name in FILES:
            (path / name).write_text("{}")
    (child / "problem.json").write_text(json.dumps({
        "provenance": {"parent_entry": {"root": str(parent)}}}))
    bindings = {str(path): {"files_sha256": {name: digest(path / name) for name in FILES}}
                for path in (parent, child)}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"canonical": [{"source": str(parent), "canonical": str(target)}],
                                "dependencies": bindings}))
    assert stage(root, audit)["roots"] == 2
    mapping = json.loads((root / "piles/source-paths.json").read_bytes())
    staged = type(child)(mapping["roots"][str(child)])
    assert staged.is_relative_to(target / "checkpoints")
    assert (staged / "model.mjb").stat().st_ino == (child / "model.mjb").stat().st_ino
    (child / "model.xml").write_text("changed")
    with pytest.raises(ValueError, match="changed after audit"):
        stage(root, audit)


def test_retirement_never_deletes_active_sources_or_broad_data_root(tmp_path):
    root = tmp_path / "lego"
    candidate = root / "data/active"
    candidate.mkdir(parents=True)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"dependencies": {str(candidate): {}},
                                "active_assistance_roots": [str(candidate)]}))
    with pytest.raises(ValueError, match="retained"):
        retire(root, candidate, audit, apply=True, canonicalized=True)
    assert candidate.exists()
    with pytest.raises(ValueError, match="contained legacy"):
        retire(root, root / "data", audit, apply=True)


def test_atomic_relocation_preserves_open_files_and_historical_ancestor_paths(tmp_path):
    root = tmp_path / "lego"
    source = root / "data/old-run"
    source.mkdir(parents=True)
    original = source / "receipt.json"
    original.write_bytes(b"frozen receipt")
    destination = root / "piles/measurements/current"
    with original.open("rb") as stream:
        inode = source.stat().st_ino
        relocate(root, source, destination)
        assert stream.read() == b"frozen receipt"
    assert source.is_symlink()
    assert destination.stat().st_ino == inode
    assert original.read_bytes() == b"frozen receipt"
    (source / "new.json").write_bytes(b"next finalized receipt")
    assert (destination / "new.json").read_bytes() == b"next finalized receipt"
