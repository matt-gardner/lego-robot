"""Root-only registration for server-validated sessions; capabilities stay inside containers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

from so101.cloud_checkpoint import fsync_directory
from so101.control_broker import canonical, trusted_json, trusted_path, validate_policy


def private_json(path: Path, value: dict) -> None:
    trusted_path(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def session_client(session_id: str, token_sha256: str, preset: dict) -> dict:
    if str(uuid.UUID(session_id)) != session_id:
        raise ValueError("A canonical server-validated session UUID is required")
    if not re.fullmatch(r"[0-9a-f]{64}", token_sha256):
        raise ValueError("Registration accepts only a capability hash, never its secret")
    if (set(preset) != {"repository", "actions", "spec_sha256"}
            or preset["repository"] != "matt-gardner/lego-robot"
            or not isinstance(preset["actions"], list)
            or not isinstance(preset["spec_sha256"], list)
            or not preset["spec_sha256"]):
        raise ValueError("Session preset must name this repository and exact approved specifications")
    return {"id": "session-" + session_id, "token_sha256": token_sha256,
            "actions": preset["actions"], "spec_sha256": preset["spec_sha256"]}


def register(session_id: str, container_id: str, token_sha256: str, preset_id: str, *,
             policy_path: Path, presets_path: Path, lock_path: Path,
             run=subprocess.run) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("Session registration requires the trusted Dockyard launch integration")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", preset_id):
        raise ValueError("Invalid fixed session preset identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError("Registration requires the full server-verified container ID")
    trusted_path(lock_path.parent)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        trusted_path(lock_path)
        presets = trusted_json(presets_path)
        if preset_id not in presets:
            raise ValueError("Session preset is not installed")
        policy = trusted_json(policy_path)
        client = session_client(session_id, token_sha256, presets[preset_id])
        registry_path = lock_path.parent / "sessions.json"
        registry = trusted_json(registry_path) if registry_path.exists() else {}
        binding = {"container_id": container_id, "token_sha256": token_sha256, "preset_id": preset_id}
        if session_id in registry and registry[session_id] != binding:
            raise ValueError("Cannot rebind a registered session to another container or capability")
        existing = next((entry for entry in policy["clients"] if entry["id"] == client["id"]), None)
        if existing is not None:
            if existing != client:
                raise ValueError("Cannot rebind an existing session capability or preset")
        else:
            if any(entry["token_sha256"] == token_sha256 for entry in policy["clients"]):
                raise ValueError("Capability hash already belongs to another session")
            policy["clients"].append(client)
        validate_policy(policy)
        if existing is None:
            private_json(policy_path, policy)
        registry[session_id] = binding
        private_json(registry_path, registry)
        # A retry after uncertain reload repeats only reload, never issuance or rebinding.
        run(["/usr/bin/systemctl", "restart", "lego-robot-control.service"],
            check=True, capture_output=True, text=True, timeout=60)
        run(["/usr/bin/systemctl", "is-active", "--quiet", "lego-robot-control.service"],
            check=True, capture_output=True, text=True, timeout=10)
        return {"client_id": client["id"], "session_id": session_id,
                "container_id": container_id, "preset_id": preset_id,
                "policy_sha256": hashlib.sha256(canonical(policy)).hexdigest(),
                "registered": True, "reloaded": True}


def revoke(session_id: str, container_id: str, *, policy_path: Path, lock_path: Path,
           run=subprocess.run) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("Session revocation requires the trusted Dockyard launch integration")
    if str(uuid.UUID(session_id)) != session_id or not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError("Revocation requires the exact server-validated session and container")
    trusted_path(lock_path.parent)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        trusted_path(lock_path)
        registry_path = lock_path.parent / "sessions.json"
        registry = trusted_json(registry_path) if registry_path.exists() else {}
        if session_id in registry and registry[session_id]["container_id"] != container_id:
            raise ValueError("Cannot revoke another container's session binding")
        policy = trusted_json(policy_path)
        policy["clients"] = [client for client in policy["clients"] if client["id"] != "session-" + session_id]
        validate_policy(policy)
        private_json(policy_path, policy)
        registry.pop(session_id, None)
        private_json(registry_path, registry)
        run(["/usr/bin/systemctl", "restart", "lego-robot-control.service"],
            check=True, capture_output=True, text=True, timeout=60)
        run(["/usr/bin/systemctl", "is-active", "--quiet", "lego-robot-control.service"],
            check=True, capture_output=True, text=True, timeout=10)
        return {"client_id": "session-" + session_id, "session_id": session_id,
                "container_id": container_id, "revoked": True, "reloaded": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "revoke"))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--token-sha256")
    parser.add_argument("--preset-id")
    args = parser.parse_args()
    paths = {"policy_path": Path("/etc/lego-robot-control/policy.json"),
             "lock_path": Path("/var/lib/lego-robot-control/session-registration.lock")}
    if args.action == "register":
        if args.token_sha256 is None or args.preset_id is None:
            parser.error("Registration requires --token-sha256 and --preset-id")
        result = register(args.session_id, args.container_id, args.token_sha256, args.preset_id,
                          presets_path=Path("/etc/lego-robot-control/session-presets.json"), **paths)
    else:
        if args.token_sha256 is not None or args.preset_id is not None:
            parser.error("Revocation accepts only session and container identity")
        result = revoke(args.session_id, args.container_id, **paths)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
