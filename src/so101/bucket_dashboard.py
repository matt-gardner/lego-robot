"""Read-only access to the single current dashboard data source."""
import json
import logging
from pathlib import Path
import sqlite3
import threading

from so101.bucket_store import current_image, current_inventory

LOGGER = logging.getLogger(__name__)


def validate_annotation(row, width, height):
    status, box, pixels = (row[key] for key in
                           ("target_bbox_status", "target_bbox_xyxy", "target_visible_pixels"))
    if type(pixels) is not int or pixels < 0:
        raise ValueError("Invalid visible pixel count")
    if status == "visible":
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                or not 0 < pixels <= (box[2] - box[0]) * (box[3] - box[1])):
            raise ValueError("Visible annotation has invalid exclusive pixel bounds/count")
    elif status == "not_visible":
        if box is not None or pixels != 0:
            raise ValueError("Invisible annotation cannot have a fabricated box or positive pixel count")
    else:
        raise ValueError("Captured segmentation must measure visible or not_visible")


class BucketDashboard:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cache = None
        self._cache_error = None
        self._cache_lock = threading.Lock()
        self._cache_ready = threading.Event()
        self._cache_stop = threading.Event()
        self._cache_thread = None

    def start_cache(self, interval: float = 20) -> None:
        def refresh():
            while not self._cache_stop.is_set():
                self.refresh_cache()
                self._cache_stop.wait(interval)
        self._cache_thread = threading.Thread(target=refresh, name="bucket-dashboard-cache", daemon=True)
        self._cache_thread.start()

    def stop_cache(self) -> None:
        self._cache_stop.set()
        if self._cache_thread:
            self._cache_thread.join()

    def refresh_cache(self) -> None:
        try:
            document = self.snapshot()
            body = json.dumps(document).encode()
            with self._cache_lock:
                self._cache = (document, body)
                self._cache_error = None
        except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.Error) as error:
            LOGGER.exception("Cannot refresh current dashboard")
            with self._cache_lock:
                self._cache_error = error
        finally:
            self._cache_ready.set()

    def cached(self):
        self._cache_ready.wait()
        with self._cache_lock:
            if self._cache_error is not None:
                raise self._cache_error
            if self._cache is None:
                raise ValueError("Current dashboard cache is not initialized")
            return self._cache

    def response(self) -> bytes:
        return self.cached()[1] if self._cache_thread else json.dumps(self.snapshot()).encode()

    def snapshot(self) -> dict:
        return current_inventory(self.root)

    def image(self, problem_id: str, index: int) -> tuple[bytes, str]:
        return current_image(self.root, problem_id, index)
