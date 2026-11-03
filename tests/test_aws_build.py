import hashlib
import subprocess
import zipfile

import pytest

from so101.aws_build import prepare, wheel_lock


def wheel(root, name, version, tag="cp312-cp312-linux_x86_64"):
    path = root / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA",
                         f"Name: {name}\nVersion: {version}\n")
    return path


def cuda_wheels(root):
    wheel(root, "torch", "2.11.0+cu130")
    wheel(root, "torchvision", "0.26.0+cu130")


def test_wheel_lock_hashes_every_exact_package(tmp_path):
    cuda_wheels(tmp_path)
    path = wheel(tmp_path, "example", "1.2", "py3-none-any")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = wheel_lock(tmp_path)
    assert f"example==1.2 --hash=sha256:{digest}\n" in result
    assert len(result.splitlines()) == 3


@pytest.mark.parametrize("tag", ["cp313-cp313-linux_x86_64", "cp312-cp312-win_amd64",
                                "cp312-cp312-manylinux_2_28_aarch64",
                                "cp313-abi3-linux_x86_64", "cp311-cp311-linux_x86_64",
                                "py3-cp312-linux_x86_64"])
def test_wrong_python_or_platform_is_rejected(tmp_path, tag):
    cuda_wheels(tmp_path)
    wheel(tmp_path, "example", "1.2", tag)
    with pytest.raises(ValueError):
        wheel_lock(tmp_path)


@pytest.mark.parametrize("tag", [
    "cp310-abi3-manylinux_2_17_x86_64",
    "cp311-abi3-manylinux_2_28_x86_64",
    "cp312-abi3-linux_x86_64",
    "cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64",
    "py2.py3-none-any",
])
def test_compatible_abi3_and_portable_wheels_are_accepted(tmp_path, tag):
    cuda_wheels(tmp_path)
    wheel(tmp_path, "example", "1.2", tag)
    assert "example==1.2 --hash=sha256:" in wheel_lock(tmp_path)


def test_missing_cuda_and_duplicate_packages_are_rejected(tmp_path):
    wheel(tmp_path, "torch", "2.11.0+cpu")
    with pytest.raises(ValueError, match="requires torch"):
        wheel_lock(tmp_path)
    wheel(tmp_path, "torch", "2.11.0+cu130")
    with pytest.raises(ValueError, match="Duplicate"):
        wheel_lock(tmp_path)


def test_symlinks_are_rejected(tmp_path):
    cuda_wheels(tmp_path)
    (tmp_path / "alias.whl").symlink_to(next(tmp_path.iterdir()))
    with pytest.raises(ValueError, match="non-symlink"):
        wheel_lock(tmp_path)


def test_prepare_exports_only_immutable_tracked_runtime_inputs(tmp_path):
    repo, wheels, output = tmp_path / "repo", tmp_path / "wheels", tmp_path / "output"
    repo.mkdir()
    wheels.mkdir()
    cuda_wheels(wheels)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src/so101").mkdir(parents=True)
    (repo / "src/so101/module.py").write_text("original = True\n")
    (repo / "scripts").mkdir()
    (repo / "scripts/Dockerfile.gpu-worker").write_text("FROM scratch\n")
    for name in ("uv.lock", "pyproject.toml"):
        (repo / name).write_text("# fixture\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
                   check=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            check=True, text=True, capture_output=True).stdout.strip()
    (repo / "src/so101/module.py").write_text("mutable = True\n")
    (repo / "secret").write_text("not exported\n")
    result = prepare(repo, commit, output, wheels)
    assert result["commit"] == commit
    assert b"original = True" in (output / "source.tar").read_bytes()
    assert b"mutable = True" not in (output / "source.tar").read_bytes()
    assert b"not exported" not in (output / "source.tar").read_bytes()
    assert wheel_lock(output / "wheelhouse") == wheel_lock(wheels)
    with pytest.raises(FileExistsError):
        prepare(repo, commit, output, wheels)
