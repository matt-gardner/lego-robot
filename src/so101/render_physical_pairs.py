"""Bounded CPU raster derivatives of immutable, physically validated pairs."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import threading
from time import perf_counter, time
from urllib.parse import urlencode

import cv2
import mujoco
import numpy as np
import torch

from so101.cloud_checkpoint import atomic_json, sha256
from so101.collect_catalog_sim import validate_recording
from so101.episode_storage import EpisodeStorage
from so101.lego_scene import restore_scene
from so101.pathtrace_recordings import restore_pose
from so101.produce_catalog import DiskBackpressure, qualify_pair
from so101.curriculum_workers import memory_state
from so101.renderer_appearance_v6 import WristAppearanceV6
from so101.robot_sim import BeamSimulation


LOGGER = logging.getLogger(__name__)
MEMORY_ALLOWANCE = 2 * 1024**3
MAX_RETRIES = 3
BROWSER_BASE = "http://127.0.0.1:8765/lego-robot"


def wait_memory(root: Path, stopping: threading.Event,
                service_publications: Callable[[], None] | None = None) -> bool:
    paused = False
    while not stopping.is_set():
        if service_publications is not None:
            service_publications()
        memory = memory_state()
        if not memory["pressure"] and memory["available_bytes"] >= MEMORY_ALLOWANCE:
            if paused:
                atomic_json(root / "resource_status.json", {"status": "ready", "memory": memory})
            return True
        atomic_json(root / "resource_status.json", {"status": "paused", "memory": memory})
        paused = True
        stopping.wait(5)
    return False


class ImageRejected(ValueError):
    """A rendered pair failed image-dependent recording/sample checks."""


def safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    target = root / relative
    if (relative.is_absolute() or ".." in relative.parts
            or not target.resolve().is_relative_to(root.resolve())
            or any(part.is_symlink() for part in (target, *target.parents))):
        raise ValueError(f"Unsafe handoff file: {name}")
    return target


def verify_handoff(path: Path) -> tuple[dict, dict]:
    if path.is_symlink():
        raise ValueError("Symlink handoff manifest")
    manifest = json.loads(path.read_bytes())
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "lego-physical-pair"
            or manifest.get("status") != "physically_validated"
            or manifest.get("images_pending") is not True
            or manifest.get("sft_qualified") is not False):
        raise ValueError("Not an unrendered physically validated v1 pair")
    release = manifest["release_id"]
    if not isinstance(release, str) or not re.fullmatch("[0-9a-f]{64}", release):
        raise ValueError("Invalid physical release identity")
    if path.name != f"{release}.json":
        raise ValueError("Manifest filename differs from release identity")
    root = Path(manifest["pair_root"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("Physical source root must be absolute and not a symlink")
    files = manifest["files_sha256"]
    actual = set()
    for source in root.rglob("*"):
        if source.is_symlink():
            raise ValueError(f"Symlink physical source: {source}")
        if source.is_file():
            actual.add(source.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError("Physical source file population differs from manifest")
    for name, digest in files.items():
        if sha256(safe_file(root, name)) != digest:
            raise ValueError(f"Physical source checksum mismatch: {name}")
    report = json.loads(safe_file(root, manifest["collection"]).read_bytes())
    for key in ("release_id", "scene_family_id", "split"):
        if report.get(key) != manifest[key]:
            raise ValueError(f"Physical collection/manifest differ: {key}")
    attempts = report["attempts"]
    if (report["status"] != "physically_validated" or len(attempts) != 2
            or [row["example_id"] for row in attempts] != manifest["example_ids"]
            or len(set(manifest["example_ids"])) != 2
            or report["successful_episodes"] != [row["source"] for row in attempts]
            or any(row["outcome"] != "success" or row["verification"]["success"] is not True
                   for row in attempts)):
        raise ValueError("Physical source is not a successful independent pair")
    first_pose = None
    for attempt in attempts:
        source = Path(attempt["source"])
        if not source.is_relative_to(root):
            raise ValueError("Physical episode escapes source root")
        metadata = json.loads((source / "episode.json").read_bytes())
        rows = [json.loads(line) for line in (source / "frames.jsonl").read_bytes().splitlines()]
        for key in ("example_id", "release_id", "scene_family_id", "split"):
            expected = attempt[key] if key == "example_id" else manifest[key]
            if metadata.get(key) != expected:
                raise ValueError(f"Physical episode identity differs: {key}")
        validate_recording(source, metadata, rows, len(metadata["initial"]["qpos"]),
                           physical_only=True)
        pose = rows[0]["qpos"]
        if first_pose is not None and pose != first_pose:
            raise ValueError("Paired commands did not independently start at the same pose")
        first_pose = pose
    return manifest, report


def renderer_identity() -> dict:
    appearance = WristAppearanceV6()
    directory = Path(__file__).parent
    names = (
        "render_physical_pairs.py", "pathtrace_recordings.py", "robot_sim.py",
        "collect_catalog_sim.py", "produce_catalog.py", "episode_storage.py",
        "lego_scene.py", "simulation_corpus.py", "control_model/dataset.py",
        "control_model/simulation_dataset.py", "control_model/command.py",
        "renderer_appearance.py", "renderer_appearance_v2.py",
        "renderer_appearance_v3.py", "renderer_appearance_v6.py",
    )
    return {
        "backend": "mujoco-cpu-raster", "mujoco_version": mujoco.__version__,
        "appearance": appearance.identity(),
        "implementation_sha256": {name: sha256(directory / name) for name in names},
        "camera_pair": ["wrist_cam", "recorder-free-overview"],
        "physics_replayed": False,
    }


def derivative_identity(manifest: dict, manifest_hash: str, renderer: dict) -> str:
    return hashlib.sha256(json.dumps(
        [manifest["release_id"], manifest_hash, renderer],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def check_image(image: np.ndarray) -> None:
    if image.shape != (240, 320, 3) or image.dtype != np.uint8 or np.ptp(image) == 0:
        raise ImageRejected("Missing, malformed or uniform paired-camera image")


def render_pair(manifest_path: Path, destination: Path, stopping: threading.Event,
                service_publications: Callable[[], None] | None = None) -> dict:
    def checkpoint() -> bool:
        if not stopping.is_set() and service_publications is not None:
            service_publications()
        return stopping.is_set()

    started = perf_counter()
    manifest_hash = sha256(manifest_path)
    manifest, report = verify_handoff(manifest_path)
    timings = {"source_validation": perf_counter() - started}
    renderer = renderer_identity()
    derivative = derivative_identity(manifest, manifest_hash, renderer)
    provenance = {
        "schema_version": 1, "kind": "lego-physical-pair-render",
        "source_release_id": manifest["release_id"], "derivative_release_id": derivative,
        "source_manifest": str(manifest_path.resolve()), "source_manifest_sha256": manifest_hash,
        "source_pair_root": manifest["pair_root"],
        "source_files_sha256": manifest["files_sha256"],
        "source_status": "physically_validated", "source_has_images": False,
        "scene_family_id": manifest["scene_family_id"], "split": manifest["split"],
        "example_ids": manifest["example_ids"], "renderer": renderer,
        "unique_training_data_claim": False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "physical_pair_render.json", provenance)
    disk = DiskBackpressure(destination)
    sources = []
    image_hashes = {}
    stage_started = perf_counter()
    for command_index, attempt in enumerate(report["attempts"]):
        if stopping.is_set():
            raise InterruptedError("Raster consumer stopped before paired completion")
        source = Path(attempt["source"])
        metadata = json.loads((source / "episode.json").read_bytes())
        rows = [json.loads(line) for line in (source / "frames.jsonl").read_bytes().splitlines()]
        snapshot = json.loads((source / "settled_state.json").read_bytes())
        if snapshot["physics_version"] != mujoco.__version__:
            raise RuntimeError("Physical model ABI differs from rendering runtime")
        physical_model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
        physical_data = mujoco.MjData(physical_model)
        xml = (source / "model.xml").read_text()
        restore_scene(physical_model, physical_data, snapshot, xml)
        appearance = WristAppearanceV6()
        render_xml = appearance.apply_xml(xml)
        model = mujoco.MjModel.from_xml_string(render_xml)
        if (model.nq != physical_model.nq
                or not np.array_equal(model.jnt_qposadr, physical_model.jnt_qposadr)):
            raise RuntimeError("Appearance model changed physical pose layout")
        appearance.apply_model(model)
        plan = json.loads((source / "scene_plan.json").read_bytes())
        sim = BeamSimulation(model=model, render=True,
                             object_names=tuple(obj["instance_id"] for obj in plan["objects"]),
                             distractor_names=())
        episode = destination / f"command_{command_index}"
        episode.mkdir()
        sources.append(str(episode.resolve()))
        try:
            sim.appearance = appearance
            sim.initial = metadata["initial"]
            sim.hide_unseen_centers = bool(metadata.get("perception", {}).get("hide_unseen_centers", False))
            for name in ("model.xml", "model.mjb", "scene_plan.json", "settled_state.json"):
                shutil.copyfile(source / name, episode / name)
            shutil.copytree(source / "source", episode / "source")
            (episode / "render_model.xml").write_text(render_xml)
            mujoco.mj_saveModel(model, str(episode / "render_model.mjb"), None)
            revised_rows = []
            with (episode / "frames.jsonl").open("w") as stream:
                for index, row in enumerate(rows):
                    if stopping.is_set():
                        raise InterruptedError("Raster consumer stopped; partial derivative retained")
                    if not wait_memory(destination, stopping, service_publications):
                        raise InterruptedError("Raster consumer stopped during memory pause")
                    if not disk.wait(1024**2, "rendering", checkpoint):
                        raise InterruptedError("Raster consumer stopped during storage pause")
                    sim.data.qvel[:] = row["qvel"]
                    sim.data.ctrl[:] = row["ctrl"]
                    sim.data.act[:] = row["act"]
                    sim.data.time = snapshot["time"] + row["elapsed"]
                    restore_pose(model, sim.data, row)
                    sim.configure_command(metadata["task"], list(zip(
                        row["command_entity_ids"], row["command_roles"], strict=True,
                    )))
                    revised = {**row, **sim._recording_command_labels()}
                    for key, image in (("image", sim.image()), ("external_image", sim.overview_image())):
                        check_image(image)
                        name = f"{'wrist' if key == 'image' else 'external'}/{index:06d}.png"
                        target = episode / name
                        target.parent.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                            raise OSError(f"Cannot save raster image: {target}")
                        revised[key] = name
                        image_hashes[f"command_{command_index}/{name}"] = sha256(target)
                    revised_rows.append(revised)
                    stream.write(json.dumps(revised, allow_nan=False) + "\n")
            revised_metadata = {
                **metadata, "release_id": derivative, "source_release_id": manifest["release_id"],
                "images_pending": False, "qualification_stage": "image-validation-pending",
                "sft_qualified": False, "physical_pair_render": provenance,
                "render_model": "render_model.xml",
                "render_model_sha256": sha256(episode / "render_model.xml"),
            }
            atomic_json(episode / "episode.json", revised_metadata)
            try:
                validate_recording(episode, revised_metadata, revised_rows, model.nq)
            except ValueError as error:
                raise ImageRejected(str(error)) from error
        finally:
            sim.close()
    timings["paired_rendering_and_image_validation"] = perf_counter() - stage_started
    collection = copy.deepcopy(report)
    collection.update(
        status="complete", release_id=derivative, source_release_id=manifest["release_id"],
        images_pending=False, sft_qualified=False, qualification_stage="image-validation-pending",
        physical_pair_render=provenance, successful_episodes=sources,
    )
    for attempt, source in zip(collection["attempts"], sources, strict=True):
        attempt["original_source"] = attempt["source"]
        attempt["source"] = source
    atomic_json(destination / "collection.json", collection)
    stage_started = perf_counter()
    if not disk.wait(2 * sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
                     "sample-qualification", checkpoint):
        raise InterruptedError("Stopped before sample qualification")
    try:
        qualification = qualify_pair(destination)
    except ValueError as error:
        raise ImageRejected(str(error)) from error
    timings["archive_and_sft_sample_validation"] = perf_counter() - stage_started
    verify_handoff(manifest_path)
    if sha256(manifest_path) != manifest_hash or renderer_identity() != renderer:
        raise RuntimeError("Immutable input or renderer changed during derivative creation")
    receipt = {
        **provenance, **qualification, "release_id": derivative,
        "status": "image_qualified", "image_qualified": True, "sft_qualified": True,
        "publication_status": "staged", "images_sha256": image_hashes,
        "timings_seconds": timings, "upstream_timings_seconds": manifest["timings_seconds"],
        "qualification_scope": "Finite paired SFT samples and image/label integrity; "
        "not hardware calibration, visual audit, unique data or train-split admission.",
    }
    atomic_json(destination / "qualification.json", receipt)
    return receipt


def verify_derivative(root: Path) -> dict:
    receipt = json.loads((root / "qualification.json").read_bytes())
    if receipt["status"] != "image_qualified" or receipt["sft_qualified"] is not True:
        raise ValueError("Derivative is not image/sample qualified")
    if sha256(root / "episodes.sqlite") != receipt["storage_sha256"]:
        raise ValueError("Derivative archive checksum mismatch")
    with EpisodeStorage(root / "episodes.sqlite") as storage:
        storage.verify()
        storage.verify_sources(root)
    for name, digest in receipt["images_sha256"].items():
        if sha256(safe_file(root, name)) != digest:
            raise ValueError("Derivative image checksum mismatch")
    return receipt


def publish_derivative(stage: Path, output: Path) -> Path:
    receipt = verify_derivative(stage)
    destination = output / "releases" / receipt["release_id"]
    if destination.exists():
        published = json.loads((destination / "qualification.json").read_bytes())
        if (sha256(destination / "episodes.sqlite") != receipt["storage_sha256"]
                or published["publication_status"] != "published"
                or published["source_manifest_sha256"] != receipt["source_manifest_sha256"]):
            raise ValueError("Immutable published derivative identity changed")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.parent / f".{receipt['release_id']}.building"
    interrupted = output / ".publication-staging" / receipt["release_id"]
    if interrupted.exists() and not building.exists():
        interrupted.chmod(0o755)
        interrupted.rename(building)
    building.mkdir(parents=True, exist_ok=True)
    for name in ("collection.json", "episodes.sqlite", "physical_pair_render.json"):
        target = building / name
        expected = sha256(stage / name)
        if target.exists():
            if sha256(target) != expected:
                raise ValueError(f"Interrupted publication checksum mismatch: {name}")
        else:
            temporary = building / f".{name}.copying"
            shutil.copyfile(stage / name, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f"Publication copy checksum mismatch: {name}")
            temporary.replace(target)
    catalog = stage / "command_0" / "source" / "parts_catalog.json"
    target = building / "parts_catalog.json"
    if target.exists() and sha256(target) != sha256(catalog):
        raise ValueError("Interrupted publication catalog changed")
    if not target.exists():
        shutil.copyfile(catalog, target)
    published = {
        **receipt, "publication_status": "published",
        "published_root": str(destination.resolve()),
        "staged_qualification_sha256": sha256(stage / "qualification.json"),
    }
    qualification = building / "qualification.json"
    if qualification.exists():
        if json.loads(qualification.read_bytes()) != published:
            raise ValueError("Interrupted publication qualification changed")
    else:
        atomic_json(qualification, published)
    with EpisodeStorage(building / "episodes.sqlite") as storage:
        storage.verify()
    for path in building.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    building.chmod(0o555)
    building.rename(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def publication_request(published: Path, episode_root: Path, output: Path) -> Path:
    receipt = json.loads((published / "qualification.json").read_bytes())
    archive = published / "episodes.sqlite"
    archive_hash = sha256(archive)
    if receipt["publication_status"] != "published" or archive_hash != receipt["storage_sha256"]:
        raise ValueError("Publication request requires a verified immutable derivative")
    relative = archive.relative_to(episode_root).as_posix()
    dashboard_release = hashlib.sha256(f"examples:{relative}:{archive_hash}".encode()).hexdigest()
    items = []
    with EpisodeStorage(archive) as storage:
        for index, source in enumerate(storage.catalog["successful_episodes"]):
            metadata = json.loads(storage.read(Path(source), "episode.json"))
            rows = storage.read(Path(source), "frames.jsonl").splitlines()
            example = metadata["example_id"]
            dashboard_item = hashlib.sha256(
                f"curriculum:{example}:derived:{dashboard_release}".encode(),
            ).hexdigest()[:20]
            query = {"release": dashboard_release, "item": example}
            first = json.loads(rows[0])
            items.append({
                "example_id": example, "dashboard_item_id": dashboard_item,
                "frame_count": len(rows),
                "lego_url": BROWSER_BASE + "/api/lego?" + urlencode(query),
                "data_url": BROWSER_BASE + "/api/data/all?" + urlencode({
                    "view": "examples", "item": dashboard_item,
                }),
                "media": [{
                    "url": BROWSER_BASE + "/api/lego?" + urlencode({
                        **query, "frame": 0, "camera": camera,
                    }),
                    "sha256": receipt["images_sha256"][f"command_{index}/{first[key]}"],
                } for camera, key in (("wrist", "image"), ("external", "external_image"))],
            })
    request = {
        "schema_version": 1, "kind": "lego-physical-render-publication-request",
        "source_release_id": receipt["source_release_id"],
        "derivative_release_id": receipt["release_id"],
        "archive_path": str(archive.resolve()), "archive_sha256": archive_hash,
        "qualification_sha256": sha256(published / "qualification.json"),
        "published_root": str(published.resolve()), "dashboard_release_id": dashboard_release,
        "example_ids": receipt["example_ids"], "items": items,
    }
    target = output / "publication_requests" / f"{receipt['source_release_id']}.json"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if json.loads(target.read_bytes()) != request:
            raise ValueError("Immutable browser publication request changed")
    else:
        atomic_json(target, request)
        target.chmod(0o444)
    return target


def acknowledge_browser(queue: Path, request_path: Path, browser_receipt: Path) -> dict:
    request = json.loads(request_path.read_bytes())
    evidence = json.loads(browser_receipt.read_bytes())
    if (evidence.get("schema_version") != 1
            or evidence.get("kind") != "lego-physical-render-browser-receipt"
            or evidence.get("status") != "verified"
            or evidence.get("request_sha256") != sha256(request_path)):
        raise ValueError("Browser verification receipt does not match publication request")
    verified_at = datetime.fromisoformat(evidence["verified_at"])
    if verified_at.tzinfo is None:
        raise ValueError("Browser verification time must include a timezone")
    published = Path(request["published_root"])
    qualification = json.loads((published / "qualification.json").read_bytes())
    if (sha256(Path(request["archive_path"])) != request["archive_sha256"]
            or sha256(published / "qualification.json") != request["qualification_sha256"]):
        raise ValueError("Browser verification references changed publication bytes")
    responses = evidence["responses"]
    expected_urls = [
        url for item in request["items"]
        for url in (item["lego_url"], item["data_url"], *(media["url"] for media in item["media"]))
    ]
    by_url = {row["url"]: row for row in responses}
    if len(by_url) != len(responses) or set(by_url) != set(expected_urls):
        raise ValueError("Browser receipt must contain every exact paired metadata/media request")
    for response in responses:
        if response["http_status"] != 200:
            raise ValueError("Browser publication request did not return HTTP 200")
    for item in request["items"]:
        lego = by_url[item["lego_url"]]["body"]
        data = by_url[item["data_url"]]["body"]
        metadata = lego["item"]
        if (lego["release"]["id"] != request["dashboard_release_id"]
                or lego["release"]["artifact_sha256"] != request["archive_sha256"]
                or metadata["example_id"] != item["example_id"]
                or metadata["release_id"] != request["derivative_release_id"]
                or metadata["media_origin"] != "derived_cpu_raster"
                or metadata["physical_pair_render"]["source_release_id"] != request["source_release_id"]
                or metadata["qualification"] != qualification):
            raise ValueError("LEGO browser selected the wrong derivative or qualification")
        if (data["id"] != item["dashboard_item_id"]
                or data["archive_release"] != request["dashboard_release_id"]
                or data["archive_item"] != item["example_id"]
                or data["media_origin"] != "derived_cpu_raster"
                or data["original_recording"] is not False
                or data["paired_camera_images"] is not True
                or data["movie_available"] is not True
                or data["qualification"] != qualification):
            raise ValueError("Data browser did not expose qualified derived paired playback")
        for payload in (lego, data):
            frames = payload["frames"]
            if (len(frames) != item["frame_count"] or not frames
                    or any(frame["wrist"] is not True or frame["external"] is not True for frame in frames)):
                raise ValueError("Browser paired frame population differs from the archive")
        for media in item["media"]:
            response = by_url[media["url"]]
            if response["body_sha256"] != media["sha256"] or response["content_type"] != "image/png":
                raise ValueError("Browser paired image bytes differ from qualified PNGs")
    manifest_path = queue / "ready" / f"{request['source_release_id']}.json"
    manifest, _ = verify_handoff(manifest_path)
    if (sha256(manifest_path) != qualification["source_manifest_sha256"]
            or manifest["example_ids"] != request["example_ids"]
            or qualification["image_qualified"] is not True
            or qualification["sft_qualified"] is not True):
        raise ValueError("Publication no longer matches its immutable physical pair")
    acknowledgment = {
        "release_id": request["source_release_id"], "status": "published",
        "derivative_release_id": request["derivative_release_id"],
        "archive_sha256": request["archive_sha256"], "published_root": request["published_root"],
        "browser_receipt_sha256": sha256(browser_receipt), "verified_at": evidence["verified_at"],
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
    }
    target = queue / "acknowledged" / manifest_path.name
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_bytes())
        for key in ("release_id", "status", "derivative_release_id", "archive_sha256"):
            if existing[key] != acknowledgment[key]:
                raise ValueError("Existing publication acknowledgment conflicts")
        return existing
    atomic_json(target, acknowledgment)
    return acknowledgment


def service_browser_receipts(queue: Path, output: Path, verified_stages: set[Path],
                             stopping: threading.Event | None = None) -> None:
    """Publish qualified pairs and consume evidence while subsequent renders are active."""
    for journal_path in sorted(output.glob("*.json")):
        if stopping is not None and stopping.is_set():
            break
        if (not re.fullmatch("[0-9a-f]{64}", journal_path.stem)
                or not (queue / "ready" / journal_path.name).exists()
                or (queue / "acknowledged" / journal_path.name).exists()):
            continue
        journal = json.loads(journal_path.read_bytes())
        if journal.get("status") not in (
                "image_qualified", "browser_pending", "browser_verification_failed"):
            continue
        evidence = output / "browser_receipts" / journal_path.name
        if (journal.get("publication_request") and (not evidence.exists()
                or sha256(evidence) == journal.get("browser_receipt_sha256"))):
            continue
        stage = Path(journal["derivative_root"])
        if stage not in verified_stages:
            verify_derivative(stage)
            verified_stages.add(stage)
        if not journal.get("publication_request"):
            required = sum((stage / name).stat().st_size for name in (
                "episodes.sqlite", "collection.json", "qualification.json",
                "physical_pair_render.json", "command_0/source/parts_catalog.json",
            ))
            if not DiskBackpressure(output).wait(
                    required, "publication", stopping.is_set if stopping is not None else lambda: False):
                break
            published = publish_derivative(stage, output)
            request = publication_request(published, queue.parent.parent, output)
            journal.update(published_root=str(published), publication_request=str(request),
                           status="browser_pending")
            atomic_json(journal_path, journal)
        if not evidence.exists():
            continue
        digest = sha256(evidence)
        try:
            acknowledge_browser(queue, Path(journal["publication_request"]), evidence)
        except (ValueError, KeyError) as error:
            LOGGER.error("Browser acknowledgment failed %s: %s", evidence, error)
            journal.update(status="browser_verification_failed", reason=str(error))
        else:
            journal["status"] = "published"
        journal["browser_receipt_sha256"] = digest
        atomic_json(journal_path, journal)


def run(queue: Path, output: Path, stopping: threading.Event, *, once: bool = False,
        publish: bool = False) -> None:
    """Sequential raster admission; publication acknowledgment requires host browser evidence."""
    queue = queue.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_relative_to(queue.parent):
        raise ValueError("Raster output must be outside the physical producer root")
    with (queue / ".raster-consumer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified_stages = set()
        next_receipt_check = 0.

        def service_publications() -> None:
            nonlocal next_receipt_check
            if not publish or stopping.is_set() or perf_counter() < next_receipt_check:
                return
            service_browser_receipts(queue, output, verified_stages, stopping)
            next_receipt_check = perf_counter() + 1.

        while not stopping.is_set():
            service_publications()
            admitted = False
            for path in sorted((queue / "ready").glob("*.json")):
                if stopping.is_set():
                    break
                acknowledgment = queue / "acknowledged" / path.name
                if acknowledgment.exists():
                    ack = json.loads(acknowledgment.read_bytes())
                    if (ack["release_id"] != path.stem
                            or ack["status"] not in ("published", "image_rejected")):
                        raise ValueError("Invalid terminal raster acknowledgment")
                    continue
                journal_path = output / f"{path.stem}.json"
                journal = json.loads(journal_path.read_bytes()) if journal_path.exists() else {"attempts": []}
                if journal.get("status") in ("image_qualified", "browser_pending", "browser_verification_failed"):
                    stage = Path(journal["derivative_root"])
                    if stage not in verified_stages:
                        verify_derivative(stage)
                        verified_stages.add(stage)
                    if publish:
                        service_publications()
                    continue
                attempts = journal["attempts"]
                if len(attempts) >= MAX_RETRIES or (attempts and attempts[-1]["retry_after"] > time()):
                    continue
                memory = memory_state()
                if memory["pressure"] or memory["available_bytes"] < MEMORY_ALLOWANCE:
                    atomic_json(output / "resource_status.json", {"status": "paused", "memory": memory})
                    break
                atomic_json(output / "resource_status.json", {"status": "ready", "memory": memory})
                disk = DiskBackpressure(output)
                if not disk.wait(MEMORY_ALLOWANCE, "raster-admission", stopping.is_set):
                    break
                destination = output / ".staging" / path.stem / f"attempt_{len(attempts):03d}"
                if destination.exists():
                    if (destination / "qualification.json").exists():
                        receipt = verify_derivative(destination)
                        journal.update(status="image_qualified", derivative_root=str(destination),
                                       derivative_release_id=receipt["release_id"])
                        atomic_json(journal_path, journal)
                        continue
                    attempts.append({"status": "interrupted", "derivative_root": str(destination),
                                     "retry_after": time() + 60})
                    atomic_json(journal_path, journal)
                    continue
                admitted = True
                started = perf_counter()
                try:
                    receipt = render_pair(path, destination, stopping, service_publications)
                except ImageRejected as error:
                    LOGGER.error("Raster image rejection %s: %s", path, error)
                    (queue / "acknowledged").mkdir(exist_ok=True)
                    atomic_json(acknowledgment, {
                        "release_id": path.stem, "status": "image_rejected",
                        "reason": str(error), "derivative_root": str(destination),
                        "elapsed_seconds": perf_counter() - started,
                    })
                    journal["status"] = "image_rejected"
                except (ValueError, RuntimeError, OSError, mujoco.FatalError) as error:
                    LOGGER.error("Raster processing failure %s: %s", path, error)
                    journal["status"] = "processing_failed"
                    attempts.append({"reason": str(error), "derivative_root": str(destination),
                                     "elapsed_seconds": perf_counter() - started,
                                     "retry_after": time() + 60 * 2**len(attempts)})
                else:
                    journal.update(status="image_qualified", derivative_root=str(destination),
                                   derivative_release_id=receipt["release_id"],
                                   publication_status="awaiting_browser_projection")
                atomic_json(journal_path, journal)
                if once:
                    return
            if once:
                return
            stopping.wait(5 if admitted else 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--publish", action="store_true",
                        help="Publish qualified derivatives; await fixed-loopback host browser receipts before ack")
    args = parser.parse_args()
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.nice(19)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    os.environ["LP_NUM_THREADS"] = "1"
    logging.basicConfig(level=logging.INFO)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    run(args.queue, args.output, stopping, once=args.once, publish=args.publish)


if __name__ == "__main__":
    main()
