import tarfile

import pytest

from so101 import aws_archive
from so101.aws_extract import extract


MODEL = "hub/models--org--model"


def cache(tmp_path):
    root = tmp_path / "hf"
    model = root / MODEL
    (model / "blobs").mkdir(parents=True)
    (model / "snapshots" / "revision").mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "blobs" / "abc").write_bytes(b"weights")
    (model / "snapshots" / "revision" / "model.bin").symlink_to("../../blobs/abc")
    (model / "refs" / "main").write_text("revision")
    (root / "token").write_text("NEVER PACKAGE")
    (root / "stored_tokens").write_text("NEVER PACKAGE")
    (root / "xet").mkdir()
    (root / "xet" / "private").write_text("NEVER PACKAGE")
    return root, model


def options(**updates):
    return dict(hf_cache=True, max_bytes=1024, max_members=20) | updates


def test_internal_links_flatten_and_credentials_are_excluded(tmp_path):
    root, _ = cache(tmp_path)
    target = tmp_path / "cache.tar"
    receipt = aws_archive.produce(root, [MODEL], target, **options())
    assert receipt["members"] == 3
    assert receipt["flattened_bytes"] == 2 * len(b"weights") + len(b"revision")
    with tarfile.open(target) as bundle:
        assert all(member.isfile() for member in bundle)
        assert set(bundle.getnames()) == set(receipt["files"])
        assert bundle.extractfile(MODEL + "/snapshots/revision/model.bin").read() == b"weights"
    restored = extract(target, tmp_path / "restored", max_bytes=1024, max_members=20)
    assert restored["files"] == receipt["files"]


@pytest.mark.parametrize("kind", ["outside", "directory", "private"])
def test_unsafe_link_rejected(tmp_path, kind):
    root, model = cache(tmp_path)
    link = model / "snapshots" / "revision" / "bad"
    if kind == "outside":
        link.symlink_to(root / "token")
    elif kind == "private":
        (model / "token").write_text("secret")
        link.symlink_to(model / "token")
    else:
        link.symlink_to(model / "blobs", target_is_directory=True)
    with pytest.raises(ValueError):
        aws_archive.inventory(root, [MODEL], **options())


@pytest.mark.parametrize("updates", [{"max_bytes": 21}, {"max_members": 2}])
def test_flattened_size_and_member_limits(tmp_path, updates):
    root, _ = cache(tmp_path)
    with pytest.raises(ValueError, match="ceiling"):
        aws_archive.inventory(root, [MODEL], **options(**updates))
    assert len(aws_archive.inventory(root, [MODEL], **options(max_bytes=22, max_members=3))) == 3


@pytest.mark.parametrize("selection", [".", "..", "/hub", "hub", "token"])
def test_explicit_hf_tree_required(tmp_path, selection):
    root, _ = cache(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        aws_archive.inventory(root, [selection], **options())


def test_source_mutation_after_inventory_rejected(tmp_path, monkeypatch):
    root, model = cache(tmp_path)
    original = aws_archive.inventory

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (model / "blobs" / "abc").write_bytes(b"changed")
        return result

    monkeypatch.setattr(aws_archive, "inventory", mutate)
    with pytest.raises(ValueError, match="changed"):
        aws_archive.produce(root, [MODEL], tmp_path / "bad.tar", **options())


def test_data_explicit_selection_and_private_path_rejection(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "collection.json").write_text("{}")
    (root / "token").write_text("secret")
    receipt = aws_archive.produce(root, ["collection.json"], tmp_path / "data.tar",
                                  **options(hf_cache=False))
    assert set(receipt["files"]) == {"collection.json"}
    with pytest.raises(ValueError, match="Private"):
        aws_archive.inventory(root, ["token"], **options(hf_cache=False))


def test_symlinked_parent_escape_rejected(tmp_path):
    root, _ = cache(tmp_path)
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "file").write_text("private")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        aws_archive.inventory(root, ["alias/file"], **options(hf_cache=False))
