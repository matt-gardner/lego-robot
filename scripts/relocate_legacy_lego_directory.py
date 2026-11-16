"""Move a legacy directory atomically while retaining its historical path as an alias."""
import argparse
import ctypes
import json
import os
from pathlib import Path


def relocate(root: Path, source: Path, destination: Path) -> dict:
    if (source.is_symlink() or not source.is_dir()
            or source.resolve() == (root / "data").resolve()
            or not source.resolve().is_relative_to((root / "data").resolve())
            or destination.exists() or destination.is_symlink()
            or not destination.absolute().is_relative_to((root / "piles").absolute())):
        raise ValueError("Relocation requires a real legacy subdirectory and an absent canonical destination")
    parent_exists = destination.parent.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to((root / "piles").resolve()):
        raise ValueError("Canonical relocation destination escapes through a symlink")
    if not parent_exists:
        owner = root.stat()
        os.chown(destination.parent, owner.st_uid, owner.st_gid)
        destination.parent.chmod(0o2750)
    before = source.stat()
    if before.st_dev != destination.parent.stat().st_dev:
        raise ValueError("Atomic relocation must stay on the same filesystem")
    destination.symlink_to(destination, target_is_directory=True)
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                      ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 2) != 0:
        error = ctypes.get_errno()
        destination.unlink()
        raise OSError(error, os.strerror(error))
    after = destination.stat()
    if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or not source.is_symlink() or source.resolve() != destination):
        raise RuntimeError("Directory identity or historical alias differs after relocation")
    return {"old_alias": str(source), "canonical_root": str(destination),
            "directory_inode_preserved": before.st_ino}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(relocate(args.root, args.source, args.destination)))


if __name__ == "__main__":
    main()
