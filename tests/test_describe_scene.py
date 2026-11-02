import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from copilot.generated.rpc import PermissionDecisionReject

import so101.describe_scene as describe_scene
import so101.robot_chat as robot_chat


def mock_client(monkeypatch, reply):
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.send_and_wait = AsyncMock(return_value=reply)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.create_session = AsyncMock(return_value=session)
    monkeypatch.setattr(describe_scene, "CopilotClient", MagicMock(return_value=client))
    return client, session


def test_describes_fresh_image_without_connecting_motors(monkeypatch):
    jpeg = b"fresh camera jpeg"
    capture = MagicMock(return_value=jpeg)
    monkeypatch.setattr(describe_scene.Camera, "jpeg", capture)
    arm = MagicMock(side_effect=AssertionError("Must not connect to motors"))
    monkeypatch.setattr(robot_chat, "FollowerArm", arm)
    client, session = mock_client(
        monkeypatch, SimpleNamespace(data=SimpleNamespace(content="A green piece."))
    )

    result = asyncio.run(describe_scene.describe_scene(describe_scene.SceneConfig()))

    assert result == "A green piece."
    capture.assert_called_once()
    arm.assert_not_called()
    config = client.create_session.call_args.kwargs
    assert config["available_tools"] == []
    assert config["enable_config_discovery"] is False
    assert config["enable_skills"] is False
    assert config["enable_file_hooks"] is False
    assert config["enable_host_git_operations"] is False
    assert isinstance(
        config["on_permission_request"](None, None), PermissionDecisionReject
    )
    attachment = session.send_and_wait.call_args.kwargs["attachments"][0]
    assert attachment["mimeType"] == "image/jpeg"
    assert base64.b64decode(attachment["data"]) == jpeg


def test_camera_failure_does_not_start_session(monkeypatch):
    monkeypatch.setattr(
        describe_scene.Camera, "jpeg", MagicMock(side_effect=RuntimeError("No camera"))
    )
    client = MagicMock()
    monkeypatch.setattr(describe_scene, "CopilotClient", client)

    with pytest.raises(RuntimeError, match="No camera"):
        asyncio.run(describe_scene.describe_scene(describe_scene.SceneConfig()))

    client.assert_not_called()


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_response_is_an_error(monkeypatch, content):
    monkeypatch.setattr(describe_scene.Camera, "jpeg", lambda self: b"jpeg")
    mock_client(monkeypatch, SimpleNamespace(data=SimpleNamespace(content=content)))
    with pytest.raises(RuntimeError, match="no scene description"):
        asyncio.run(describe_scene.describe_scene(describe_scene.SceneConfig()))


def test_missing_response_is_an_error(monkeypatch):
    monkeypatch.setattr(describe_scene.Camera, "jpeg", lambda self: b"jpeg")
    mock_client(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no scene description"):
        asyncio.run(describe_scene.describe_scene(describe_scene.SceneConfig()))
