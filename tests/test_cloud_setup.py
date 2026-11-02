import subprocess

import pytest

from so101.cloud_requirements import cuda_requirements


def test_cuda_export_changes_only_accelerator_builds_not_versions_or_markers():
    locked = (
        "torch==2.11.0 ; sys_platform != 'linux'\n"
        "torch==2.11.0+cpu ; sys_platform == 'linux'\n"
        "torchvision==0.26.0+cpu ; sys_platform == 'linux'\n"
        "torchcodec==0.11.1+cpu ; sys_platform == 'linux'\n"
        "numpy==2.2.6\n"
    )
    assert cuda_requirements(locked) == locked.replace("+cpu", "")
    with pytest.raises(ValueError, match="Unexpected accelerator"):
        cuda_requirements(locked.replace("2.11.0+cpu", "2.11.0+cu128"))
    with pytest.raises(ValueError, match="Missing locked"):
        cuda_requirements("torch==2.11.0\n")


def test_cloud_setup_dry_run_never_creates_storage(tmp_path):
    storage = tmp_path / "not-mounted"
    result = subprocess.run(
        ["bash", "scripts/setup_cloud_gpu.sh", "--storage", str(storage), "--dry-run"],
        text=True, capture_output=True, check=True,
    )
    assert "without loading a policy" in result.stdout
    assert not storage.exists()


@pytest.mark.parametrize("arguments", [
    [], ["--storage"], ["--storage", "/"], ["--storage", "relative"],
    ["--unknown"],
])
def test_cloud_setup_rejects_invalid_input_before_installing(arguments):
    result = subprocess.run(
        ["bash", "scripts/setup_cloud_gpu.sh", *arguments],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
