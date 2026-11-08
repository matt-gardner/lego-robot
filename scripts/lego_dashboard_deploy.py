#!/usr/bin/env python3
"""Root-installed zero-argument deployment of published dashboard-only source."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

CONFIG = Path("/etc/lego-robot-control/dashboard-deploy.json")
CATALOG_URL = "http://127.0.0.1:8765/lego-robot/api/lego"
EVIDENCE_VERSION = "ldraw-2026-10-023083918da2-zup-v17-mj3"
API_LIMIT = 2 * 1024 * 1024
STATUS_LIMIT = 16 * 1024 - 1
SSH = (
    "ssh -i /mnt/dockyard/service-home/.ssh/dockyard-repo/id_ed25519 "
    "-o IdentitiesOnly=yes "
    "-o UserKnownHostsFile=/mnt/dockyard/service-home/.ssh/dockyard-repo/known_hosts "
    "-o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10"
)


def trusted(path: Path) -> None:
    for entry in (path, *path.parents):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(f"Untrusted deployment path: {entry}")


def run(*args: str, timeout: int = 30) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=timeout).stdout


def git(repo: Path, *args: str) -> bytes:
    return run("/usr/bin/git", "-c", f"core.sshCommand={SSH}", "--git-dir", str(repo),
               *args, timeout=40)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in immutable source: {path}")
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def stage_release(baseline: Path, destination: Path, repo: Path,
                  commit: str, paths: list[str]) -> dict[str, str]:
    shutil.copytree(baseline, destination / "src", ignore=shutil.ignore_patterns("__pycache__"))
    hashes = {}
    for name in paths:
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[:2] != ("src", "so101")):
            raise ValueError(f"Invalid scoped source path: {name}")
        mode = git(repo, "ls-tree", commit, "--", name).decode().split()
        if not mode or mode[0] != "100644":
            raise ValueError(f"Published dashboard source must be a regular file: {name}")
        content = git(repo, "show", f"{commit}:{name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (destination / "manifest.json").write_text(json.dumps({
        "published_commit": commit, "sha256": hashes,
    }, sort_keys=True, indent=2) + "\n")
    for entry in destination.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"Symlink in baseline release: {entry}")
        entry.chmod(0o755 if entry.is_dir() else 0o644)
    destination.chmod(0o755)
    return hashes


def atomic_write(path: Path, content: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".dashboard-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def health(url: str, timeout: int = 60) -> dict:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last = "not attempted"
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=min(10, deadline - time.monotonic())) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")
                result = json.loads(response.read(16 * 1024 * 1024))
                if not isinstance(result, dict):
                    raise ValueError("Health API did not return an object")
                return result
        except (URLError, TimeoutError, ValueError, ConnectionError) as error:
            last = str(error)
            time.sleep(.25)
    raise RuntimeError(f"Live dashboard health failed: {last}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"Live catalog redirect refused: HTTP {code}")


def catalog_api(query: dict[str, str] | None = None) -> dict:
    url = CATALOG_URL + ("?" + urlencode(query) if query else "")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=4) as response:
        if response.status != 200:
            raise ValueError(f"Live catalog HTTP {response.status}")
        body = response.read(API_LIMIT + 1)
    if len(body) > API_LIMIT:
        raise ValueError("Live catalog response exceeds 2 MiB")
    result = json.loads(body)
    if not isinstance(result, dict) or "error" in result:
        raise ValueError("Live catalog did not return a successful JSON object")
    return result


def status() -> dict:
    run("/usr/bin/systemctl", "is-active", "so101-web.service", timeout=4)
    pid = run("/usr/bin/systemctl", "show", "so101-web.service",
              "--property=MainPID", "--value", timeout=4).decode().strip()
    if not pid.isdecimal() or int(pid) == 0:
        raise ValueError("Dashboard has no live process")
    environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    sources = [item.removeprefix(b"PYTHONPATH=").decode() for item in environment
               if item.startswith(b"PYTHONPATH=")]
    if len(sources) != 1:
        raise ValueError("Dashboard live source is not uniquely identified")
    source = Path(sources[0])
    if source.parent.parent != Path("/opt/lego-dashboard/releases") or source.name != "src":
        raise ValueError("Dashboard is not running a managed published release")
    manifest = source.parent / "manifest.json"
    trusted(manifest)
    publication = json.loads(manifest.read_text())
    for name, expected in publication["sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("src", "so101"):
            raise ValueError("Invalid managed manifest source path")
        path = source.parent / relative
        trusted(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Managed source hash mismatch: {name}")
    releases = catalog_api().get("releases")
    if not isinstance(releases, list) or not 1 <= len(releases) <= 256:
        raise ValueError("Live catalog must contain 1..256 releases")
    fields = ("id", "kind", "version", "path", "catalog_sha256", "artifact_sha256",
              "generated_at", "validated_parts", "publication_mtime_ns",
              "qualification_notices", "qualification")
    summaries = []
    for release in releases:
        if not isinstance(release, dict):
            raise ValueError("Invalid live catalog release")
        for field in ("id", "catalog_sha256", "artifact_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(release.get(field, ""))):
                raise ValueError(f"Invalid live catalog {field}")
        summaries.append({key: release[key] for key in fields if key in release})
    selected = [release for release in summaries
                if release.get("kind") == "parts" and release.get("version") == EVIDENCE_VERSION]
    if len(selected) != 1:
        raise ValueError("Required live v17 catalog is missing or ambiguous")
    release = selected[0]
    parts = [row for row in summaries if row.get("kind") == "parts"]
    if any(type(row.get("publication_mtime_ns")) is not int for row in parts):
        raise ValueError("Live parts catalogs lack publication ordering evidence")
    recent = sorted(parts, key=lambda row: (row["publication_mtime_ns"], row["id"]),
                    reverse=True)[:8]
    if release not in recent:
        recent.append(release)
    counts = {}
    for stored_status in ("verified", "unavailable", "imported", "prepared"):
        result = catalog_api({"release": release["id"], "status": stored_status, "limit": "1"})
        if result.get("release") != next(row for row in releases if row["id"] == release["id"]):
            raise ValueError("Live catalog publication changed during status inspection")
        count = result.get("total")
        if type(count) is not int or count < 0 or result.get("limit") != 1:
            raise ValueError("Invalid live catalog filtered count")
        counts[stored_status] = count
    if counts["verified"] != release.get("validated_parts"):
        raise ValueError("Live catalog verified count disagrees with release summary")
    result = {"active": True, "healthy": True, "pid": pid, "source": str(source),
              "publication": publication, "catalog_api": CATALOG_URL,
              "release_count": len(releases), "parts_release_count": len(parts),
              "omitted_parts_releases": len(parts) - len(recent),
              "release_selection": "newest eight parts publications plus required v17",
              "releases": recent, "catalog_evidence": {
                  "release": release, "stored_status_counts": counts,
                  "total_parts": sum(counts.values()),
                  "verification_attempts": counts["verified"] + counts["unavailable"],
                  "verification_attempts_definition": "verified + unavailable",
              }}
    if len(json.dumps(result).encode()) > STATUS_LIMIT:
        raise ValueError("Managed status response exceeds native 16 KiB receipt limit")
    return result


def activate(override: Path, release: Path, url: str) -> str:
    previous = override.read_bytes() if override.exists() else None
    try:
        atomic_write(override, f"[Service]\nEnvironment=PYTHONPATH={release}/src\n".encode())
        run("/usr/bin/systemctl", "daemon-reload")
        run("/usr/bin/systemctl", "restart", "so101-web.service")
        run("/usr/bin/systemctl", "is-active", "so101-web.service")
        health(url)
        pid = run("/usr/bin/systemctl", "show", "so101-web.service",
                  "--property=MainPID", "--value").decode().strip()
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if f"PYTHONPATH={release}/src".encode() not in environment:
            raise RuntimeError("Live dashboard source does not match activated release")
        return pid
    except Exception as error:
        try:
            if previous is None:
                override.unlink(missing_ok=True)
            else:
                atomic_write(override, previous)
            run("/usr/bin/systemctl", "daemon-reload")
            run("/usr/bin/systemctl", "restart", "so101-web.service")
            run("/usr/bin/systemctl", "is-active", "so101-web.service")
        except Exception as rollback_error:
            raise RuntimeError(f"Activation failed: {error}; rollback failed: {rollback_error}") from error
        raise RuntimeError(f"Activation failed; previous release restored: {error}") from error


def deploy(config: Path = CONFIG) -> dict:
    stage = "manifest"
    try:
        trusted(config)
        settings = json.loads(config.read_text())
        repo = Path(settings["repository"])
        baseline = Path(settings["baseline"])
        releases = Path(settings["releases"])
        override = Path(settings["override"])
        for path in (repo, baseline, releases, override.parent):
            trusted(path)
        if tree_hash(baseline) != settings["baseline_sha256"]:
            raise ValueError("Frozen compatible baseline hash mismatch")
        stage = "fetch-published"
        if git(repo, "remote", "get-url", "origin").decode().strip() != "git@github.com:matt-gardner/lego-robot.git":
            raise ValueError("Unexpected published repository")
        git(repo, "fetch", "--no-tags", "origin", "refs/heads/main:refs/heads/published")
        commit = git(repo, "rev-parse", "refs/heads/published").decode().strip()
        stage = "stage"
        staging = Path(tempfile.mkdtemp(prefix=".staged-", dir=releases))
        try:
            hashes = stage_release(baseline, staging, repo, commit, settings["paths"])
            stage = "import-check"
            run("/usr/sbin/runuser", "-u", "ubuntu", "--", settings["python"], "-I", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import so101.rl_dashboard",
                str(staging / "src"))
            digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            release = releases / f"dashboard-{commit[:12]}-{digest[:12]}"
            if release.exists():
                if (release / "manifest.json").read_bytes() != (staging / "manifest.json").read_bytes():
                    raise ValueError("Existing immutable release manifest mismatch")
                for name, expected in hashes.items():
                    if hashlib.sha256((release / name).read_bytes()).hexdigest() != expected:
                        raise ValueError(f"Existing immutable release hash mismatch: {name}")
            else:
                os.replace(staging, release)
            stage = "activate-and-health"
            pid = activate(override, release, settings["health_url"])
            result = {"healthy": True, "stage": "complete", "published_commit": commit,
                      "release": str(release), "pid": pid, "sha256": hashes}
            atomic_write(releases / "last-success.json", json.dumps(result, sort_keys=True).encode())
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    except Exception as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            detail += ": " + error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Dashboard deployment failed at {stage}: {detail}") from error


def main() -> int:
    if sys.argv[1:] not in ([], ["--status"]) or os.geteuid() != 0:
        print("Requires root and no arguments or fixed --status", file=sys.stderr)
        return 2
    if sys.argv[1:] == ["--status"]:
        try:
            print(json.dumps(status(), sort_keys=True))
        except Exception as error:
            print(f"Live dashboard status failed: {error}", file=sys.stderr)
            return 1
        return 0
    with open("/run/lock/lego-dashboard-deploy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print(json.dumps(deploy(), sort_keys=True))
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
