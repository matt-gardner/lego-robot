"""Retire one explicitly named, dependency-audited obsolete LEGO directory."""
import argparse
import json
import os
from pathlib import Path
import shutil

from scripts.audit_legacy_lego_dependencies import digest


def retire(root: Path, candidate: Path, audit_path: Path, *, apply: bool,
           canonicalized: bool = False) -> dict:
    candidate = candidate.absolute()
    data = root.absolute() / "data"
    if (candidate.is_symlink() or not candidate.is_dir()
            or candidate == data or not candidate.resolve().is_relative_to(data.resolve())):
        raise ValueError("Retirement requires one existing contained legacy subdirectory")
    audit = json.loads(audit_path.read_bytes())
    protected = [Path(path) for path in audit["dependencies"]]
    protected.append(data / "simulation/model")
    for path in protected:
        if path != candidate and not path.is_relative_to(candidate):
            continue
        if (not canonicalized or str(path) in audit["active_assistance_roots"]
                or path == data / "simulation/model"):
            raise ValueError("Retirement target contains a retained reset or model dependency")
        mapping = json.loads((root / "piles/source-paths.json").read_bytes())
        target = Path(mapping["roots"][str(path)])
        if not target.resolve().is_relative_to((root / "piles").resolve()):
            raise ValueError("Retained source mapping escapes canonical piles")
        for name, sha in audit["dependencies"][str(path)]["files_sha256"].items():
            if digest(target / name) != sha:
                raise ValueError("Retained source has not been preserved byte-for-byte")
    references = []
    blindspots = []
    needle = str(candidate).encode()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if b"retire_legacy_lego_directory.py" in (process / "cmdline").read_bytes():
                continue
            for name in ("cwd", "exe", "root"):
                try:
                    value = os.readlink(process / name).encode()
                except FileNotFoundError:
                    continue
                if needle in value:
                    references.append((process.name, name))
            for name in ("maps", "cmdline", "environ"):
                try:
                    value = (process / name).read_bytes()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if needle in value:
                    references.append((process.name, name))
            for descriptor in (process / "fd").iterdir():
                try:
                    value = os.readlink(descriptor).encode()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if needle in value:
                    references.append((process.name, "fd"))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            blindspots.append(process.name)
    if references or blindspots:
        raise ValueError(f"Unresolved process references/blindspots: {references}, {blindspots}")
    files = 0
    logical = 0
    reclaimable = 0
    seen = set()
    for path in candidate.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            stat = path.stat()
            files += 1
            logical += stat.st_size
            key = (stat.st_dev, stat.st_ino)
            if key not in seen and stat.st_nlink == 1:
                reclaimable += stat.st_blocks * 512
            seen.add(key)
    result = {"candidate": str(candidate), "files": files, "logical_bytes": logical,
              "unique_single_link_bytes": reclaimable, "applied": apply}
    if apply:
        shutil.rmtree(candidate)
        if candidate.exists():
            raise RuntimeError("Legacy directory remains after retirement")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--canonicalized", action="store_true")
    args = parser.parse_args()
    for candidate in args.candidate:
        print(json.dumps(retire(args.root, candidate, args.audit, apply=args.apply,
                                canonicalized=args.canonicalized)), flush=True)


if __name__ == "__main__":
    main()
