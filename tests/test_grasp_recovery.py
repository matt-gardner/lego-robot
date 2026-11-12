import pytest
import torch

from so101.fit_grasp_recovery import recovery_batch, recovery_loss


def sample(valid):
    return {"inputs": ([torch.ones(1, 3, 2, 2)], [torch.ones(1, dtype=torch.bool)],
                       *[torch.ones(1, 2) for _ in range(13)]),
            "actions": torch.ones(1, 3, 6), "valid_steps": valid}


def test_recorded_input_order_and_future_action_padding():
    inputs, valid = recovery_batch([sample(1), sample(3)], 32, "cpu")
    assert len(inputs) == 16
    assert inputs[0][0].shape == (2, 3, 2, 2)
    assert inputs[5].shape == (2, 3, 32)
    assert torch.all(inputs[5][..., :6] == 1)
    assert torch.all(inputs[5][..., 6:] == 0)
    assert valid.tolist() == [[True, False, False], [True, True, True]]


def test_loss_ignores_padded_steps_and_non_robot_dimensions():
    losses = torch.ones(2, 3, 32, requires_grad=True)
    valid = torch.tensor([[True, False, False], [True, True, True]])
    with torch.no_grad():
        losses[0, 1:] = 999
        losses[..., 6:] = 999
    loss = recovery_loss(losses, valid)
    assert loss.item() == 1.
    loss.backward()
    assert torch.all(losses.grad[0, 1:] == 0)
    assert torch.all(losses.grad[..., 6:] == 0)
    assert losses.grad[..., :6].sum().item() == pytest.approx(1.)


@pytest.mark.parametrize("valid", [0, 4])
def test_rejects_empty_or_out_of_range_teacher_targets(valid):
    with pytest.raises(ValueError, match="future-action length"):
        recovery_batch([sample(valid), sample(1)], 32, "cpu")
