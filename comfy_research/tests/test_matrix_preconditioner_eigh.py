"""B-004: preconditioner eigensystem validation and CPU-double fallback."""

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.optimizers.matrix_preconditioner_optimizers import (  # noqa: E402
    _eigh_psd_with_fallback,
)


def test_eigh_rejects_non_finite_input_matrix() -> None:
    bad = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        _eigh_psd_with_fallback(bad, eps=1e-8)


def test_eigh_rejects_non_finite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    real_eigh = torch.linalg.eigh

    def nan_eigh(mat: torch.Tensor, *args: object, **kwargs: object):
        evals, evecs = real_eigh(mat, *args, **kwargs)
        return evals * float("nan"), evecs

    monkeypatch.setattr(torch.linalg, "eigh", nan_eigh)
    with pytest.raises(ValueError, match="non-finite"):
        _eigh_psd_with_fallback(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), eps=1e-8)


def test_eigh_retries_after_transient_non_finite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    real_eigh = torch.linalg.eigh
    calls = 0

    def transient_nan_eigh(mat: torch.Tensor, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        evals, evecs = real_eigh(mat, *args, **kwargs)
        if calls == 1:
            return evals * float("nan"), evecs
        return evals, evecs

    monkeypatch.setattr(torch.linalg, "eigh", transient_nan_eigh)
    evals, evecs = _eigh_psd_with_fallback(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), eps=1e-8)
    assert calls >= 2
    assert torch.isfinite(evals).all()
    assert torch.isfinite(evecs).all()


def test_eigh_cpu_double_fallback_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    real_eigh = torch.linalg.eigh

    def device_eigh_fails(mat: torch.Tensor, *args: object, **kwargs: object):
        if mat.dtype != torch.float64:
            raise RuntimeError("simulated device eigh failure")
        return real_eigh(mat, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", device_eigh_fails)
    mat = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    evals, evecs = _eigh_psd_with_fallback(mat, eps=1e-8)
    assert torch.isfinite(evals).all()
    assert torch.isfinite(evecs).all()
    assert evals.dtype == mat.dtype
    assert evecs.dtype == mat.dtype
    reconstruct = (evecs * evals.unsqueeze(0)) @ evecs.T
    assert torch.allclose(reconstruct, (mat + mat.T) / 2, atol=1e-4)
