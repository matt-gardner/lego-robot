import hashlib
import io
import json
import tarfile

import pytest

from so101.aws_extract import extract


def archive(path, name="collection/data.json", kind=tarfile.REGTYPE, body=b"data"):
    with tarfile.open(path, "w") as bundle:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.mode = 0o777
        member.linkname = "/outside"
        member.size = len(body) if kind == tarfile.REGTYPE else 0
        bundle.addfile(member, io.BytesIO(body) if member.size else None)


def test_extract_creates_only_new_readable_nonwritable_input_copies(tmp_path):
    source, target = tmp_path / "input.tar", tmp_path / "prepared"
    archive(source)
    original = source.read_bytes()
    result = extract(source, target, max_bytes=4, max_members=1)
    assert result["bytes"] == 4 and result["members"] == 1
    assert result["files"]["collection/data.json"]["sha256"] == hashlib.sha256(b"data").hexdigest()
    assert (target / "collection/data.json").stat().st_mode & 0o777 == 0o644
    assert (target / "collection").stat().st_mode & 0o777 == 0o755
    assert source.read_bytes() == original
    assert json.loads((target / ".extracted.json").read_text()) == result
    with pytest.raises(FileExistsError):
        extract(source, target, max_bytes=4, max_members=1)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "nested/../../outside",
                                 "nested\\outside", ".extracted.json"])
def test_unsafe_archive_paths_never_commit_success(tmp_path, name):
    source, target = tmp_path / "input.tar", tmp_path / "prepared"
    archive(source, name=name)
    with pytest.raises(ValueError):
        extract(source, target, max_bytes=100, max_members=10)
    assert not (target / ".extracted.json").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                                 tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_links_and_special_files_are_rejected(tmp_path, kind):
    source, target = tmp_path / "input.tar", tmp_path / "prepared"
    archive(source, kind=kind)
    with pytest.raises(ValueError):
        extract(source, target, max_bytes=100, max_members=10)


def test_archive_byte_and_member_ceilings_are_exact(tmp_path):
    source = tmp_path / "input.tar"
    archive(source)
    with pytest.raises(ValueError, match="byte ceiling"):
        extract(source, tmp_path / "too-small", max_bytes=3, max_members=1)
    with tarfile.open(source, "w") as bundle:
        for name in ("first", "second"):
            member = tarfile.TarInfo(name)
            bundle.addfile(member)
    with pytest.raises(ValueError, match="member ceiling"):
        extract(source, tmp_path / "too-many", max_bytes=1, max_members=1)


def test_expired_lease_never_commits_extraction_receipt(tmp_path):
    source, target = tmp_path / "input.tar", tmp_path / "prepared"
    archive(source)
    readings = iter((0, 0, 2))
    with pytest.raises(TimeoutError, match="lease expired"):
        extract(source, target, max_bytes=4, max_members=1,
                deadline=1, wall=lambda: next(readings))
    assert not (target / ".extracted.json").exists()
