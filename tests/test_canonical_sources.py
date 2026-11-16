import json

import pytest

from so101.canonical_sources import canonical_source_root


def test_historical_root_resolves_without_rewriting_frozen_source_binding(tmp_path):
    source = tmp_path / "data/retired/scene"
    target = tmp_path / "piles/pile_0001/checkpoints/training"
    target.mkdir(parents=True)
    manifest = tmp_path / "piles/source-paths.json"
    manifest.write_text(json.dumps({"schema_version": 1, "roots": {str(source): str(target)}}))
    assert not source.exists()
    assert canonical_source_root(source) == target
    assert json.loads(manifest.read_bytes())["roots"][str(source)] == str(target)
    target.rmdir()
    target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|symlink"):
        canonical_source_root(source)


def test_unmapped_sources_keep_existing_runtime_contract(tmp_path):
    assert canonical_source_root(tmp_path / "original") == tmp_path / "original"
    with pytest.raises(ValueError, match="absolute"):
        canonical_source_root(type(tmp_path)("relative"))
