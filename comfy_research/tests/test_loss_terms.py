"""Primary loss tensor regression: MultiSlot keeps [B, K, V] while token CE flattens."""

import pytest

torch = pytest.importorskip("torch")

from fastapi import HTTPException  # noqa: E402

from comfy_research.engine.losses.loss_builders import MultiSlotCrossEntropyLoss  # noqa: E402
from comfy_research.engine.trainer.loss_terms import _trainer_primary_loss_tensor  # noqa: E402


def test_multi_slot_primary_loss_accepts_bkv_and_backprops() -> None:
    torch.manual_seed(0)
    pred = torch.randn(4, 3, 7, requires_grad=True)  # [B, K, V]
    targets = torch.randint(0, 7, (4, 3))  # [B, K]
    criterion = MultiSlotCrossEntropyLoss(label_smoothing=0.0)
    loss = _trainer_primary_loss_tensor(
        pred,
        targets,
        trainer_task="token_classification",
        criterion=criterion,
        loss_scale=1.0,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_token_ce_primary_loss_still_flattens_before_criterion() -> None:
    torch.manual_seed(1)
    pred = torch.randn(4, 3, 7)  # [B, T, V]
    targets = torch.randint(0, 7, (4, 3))  # [B, T]
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    loss = _trainer_primary_loss_tensor(
        pred,
        targets,
        trainer_task="token_classification",
        criterion=criterion,
        loss_scale=1.0,
    )
    expected = criterion(pred.reshape(-1, 7), targets.reshape(-1).long())
    assert torch.allclose(loss, expected)


def test_multi_slot_primary_loss_rejects_flat_shapes() -> None:
    pred = torch.randn(12, 7)  # flattened [B*K, V] - the old buggy path
    targets = torch.randint(0, 7, (12,))
    criterion = MultiSlotCrossEntropyLoss()
    with pytest.raises(HTTPException) as exc_info:
        _trainer_primary_loss_tensor(
            pred,
            targets,
            trainer_task="token_classification",
            criterion=criterion,
            loss_scale=1.0,
        )
    assert "MultiSlot" in str(exc_info.value.detail)


def test_multi_slot_primary_loss_rejects_transposed_target() -> None:
    pred = torch.randn(2, 3, 5)  # [B=2, K=3, V=5]
    targets = torch.randint(0, 5, (3, 2))  # transposed [K, B]: same six slots, wrong pairing
    criterion = MultiSlotCrossEntropyLoss()
    with pytest.raises(HTTPException) as exc_info:
        _trainer_primary_loss_tensor(
            pred,
            targets,
            trainer_task="token_classification",
            criterion=criterion,
            loss_scale=1.0,
        )
    assert "must match" in str(exc_info.value.detail)


def test_multi_slot_criterion_rejects_transposed_target() -> None:
    pred = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (3, 2))
    criterion = MultiSlotCrossEntropyLoss()
    with pytest.raises(ValueError, match="must match"):
        criterion(pred, targets)
