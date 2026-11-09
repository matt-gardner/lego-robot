import json
from pathlib import Path
import pytest

from so101.curriculum_workers import supervise, worker_slots


def test_fanout_reserves_measured_growth_not_just_current_rss():
    assert worker_slots(cpus=8, available_bytes=3000, peak_bytes=1000,
                        resident_bytes=[200, 500], pressure=False) == 1
    assert worker_slots(cpus=8, available_bytes=1200, peak_bytes=1000,
                        resident_bytes=[200, 500], pressure=False) == 0


def test_pressure_prevents_admission_and_affinity_bounds_parallelism():
    assert worker_slots(cpus=8, available_bytes=100000, peak_bytes=1000,
                        resident_bytes=[], pressure=True) == 0
    assert worker_slots(cpus=2, available_bytes=100000, peak_bytes=1000,
                        resident_bytes=[1000], pressure=False) == 1


@pytest.mark.parametrize("pressure,expected_workers,retire", [
    (False, 2, False), (True, 1, False), (False, 2, True),
])
def test_supervisor_fans_out_after_matching_measurement_and_stops_children(
    tmp_path, monkeypatch, pressure, expected_workers, retire
):
    commands, processes, killed = [], {}, []
    monkeypatch.setattr("so101.curriculum_workers.recipe",
                        lambda *args: {"source_sha256": "pinned"})
    monkeypatch.setattr("so101.curriculum_workers.shutil.which", lambda name: "/usr/bin/ionice")
    monkeypatch.setattr("so101.curriculum_workers.os.sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr("so101.curriculum_workers.memory_state",
                        lambda: {"available_bytes": 8 * 1024**2,
                                 "pressure": pressure or (retire and len(commands) >= 2)})
    monkeypatch.setattr("so101.curriculum_workers.process_state", lambda pid: processes.get(pid))
    def kill(pid, sig):
        killed.append(pid)
        if retire:
            processes.pop(pid, None)
    monkeypatch.setattr("so101.curriculum_workers.os.kill", kill)
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 8000 + len(commands)
            commands.append(command)
            processes[self.pid] = ("S", self.pid, 1024**2)
            root = Path(command[command.index("--root") + 1])
            root.mkdir(parents=True)
            (root / "producer.json").write_text(json.dumps({
                "attempts": [{"status": "qualified", "worker_source_sha256": "pinned",
                              "object_count": 8,
                              "peak_rss_kib": 1024}],
            }))
        def poll(self):
            return -15 if retire and self.pid not in processes else None
        def wait(self):
            processes.pop(self.pid, None)
            return 0
    monkeypatch.setattr("so101.curriculum_workers.subprocess.Popen", Child)
    class Stop:
        cycles = 0
        def is_set(self):
            return self.cycles == (6 if retire else 3)
        def wait(self, seconds):
            self.cycles += 1
    root = tmp_path / "supervisor"
    supervise(root, tmp_path / "catalog", tmp_path / "appearance", ["3005"],
              7, 15, 1024, Stop())
    assert len(commands) == expected_workers
    assert all(command[:3] == ["/usr/bin/ionice", "-c", "3"] for command in commands)
    assert all(command[command.index("--ledger") + 1] == str(root / "ledger.sqlite")
               for command in commands)
    assert sorted(killed) == list(range(8000, 8000 + expected_workers))
    state = json.loads((root / "supervisor.json").read_bytes())
    assert state["active_workers"] == 0
    assert state["shutdown_requested"] is True
