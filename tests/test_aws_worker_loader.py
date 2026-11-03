import hashlib
import io
import runpy
import tarfile

import pytest


def archive(tmp_path, name="src/so101/__init__.py", kind=tarfile.REGTYPE):
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as bundle:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind == tarfile.REGTYPE:
            member.size = 4
            bundle.addfile(member, io.BytesIO(b"pass"))
        else:
            member.linkname = "/outside"
            bundle.addfile(member)
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_loader_pinned_regular_source_only_and_no_overwrite(tmp_path):
    unpack = runpy.run_path("scripts/so101-worker-install.py")["unpack"]
    source, digest = archive(tmp_path)
    root = unpack(source, digest, tmp_path / "worker")
    assert (root / "src/so101/__init__.py").read_bytes() == b"pass"
    with pytest.raises(FileExistsError):
        unpack(source, digest, tmp_path / "worker")


@pytest.mark.parametrize("name,kind", [
    ("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
    ("data/private", tarfile.REGTYPE), ("src/so101/link", tarfile.SYMTYPE),
    ("src/so101/link", tarfile.LNKTYPE), ("src/so101/fifo", tarfile.FIFOTYPE),
])
def test_loader_rejects_noncode_and_links(tmp_path, name, kind):
    unpack = runpy.run_path("scripts/so101-worker-install.py")["unpack"]
    source, digest = archive(tmp_path, name, kind)
    with pytest.raises(ValueError, match="non-code"):
        unpack(source, digest, tmp_path / "worker")
    assert not (tmp_path / "escape").exists()


def test_loader_digest_mismatch_makes_no_directory(tmp_path):
    unpack = runpy.run_path("scripts/so101-worker-install.py")["unpack"]
    source, _ = archive(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        unpack(source, "0" * 64, tmp_path / "worker")
    assert not (tmp_path / "worker").exists()
