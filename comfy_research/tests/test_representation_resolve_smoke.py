"""Runtime representation resolution fallback for rankme/alphaReQ observables.

The lottery draw and manually wired observables can carry a stale default
representation id (``0::output``) that does not exist on most models. The
recorders resolve ids against the real model and fall back to the final hidden
representation; these tests pin that behaviour for the models the draw produces.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from comfy_research.engine.analysis.representation_specs import (
    collect_representation_tensors,
    resolve_representation_for_log,
)
from comfy_research.engine.models.model_builders import (
    _build_gated_mlp,
    _build_mlp,
    _build_moe_mlp,
)
from comfy_research.nodes.definitions.observables.representation_rankme import (
    _rankme_and_eigenvalues,
)


def _assert_resolves_to(model: nn.Module, expected_key: str, *, requested: str = "0::output") -> None:
    reps = collect_representation_tensors(model, torch.randn(16, 11), 2)
    assert requested not in reps, f"{requested} unexpectedly present: {sorted(reps)}"
    tensor, resolved = resolve_representation_for_log(model, reps, requested)
    assert resolved == expected_key
    assert tensor is not None
    rankme, eigenvalues = _rankme_and_eigenvalues(tensor, token_positions_as_samples=False)
    assert math.isfinite(rankme)
    assert eigenvalues and all(math.isfinite(v) for v in eigenvalues)


def test_mlp_default_is_valid_and_exact_match_wins() -> None:
    # Plain MLP is a top-level Sequential, so the "0::output" default is a real
    # module (the first Linear) and must resolve exactly, without a fallback.
    model = _build_mlp(11, 1, 2, 48, "silu")
    reps = collect_representation_tensors(model, torch.randn(16, 11), 2)
    assert "0::output" in reps
    tensor, resolved = resolve_representation_for_log(model, reps, "0::output")
    assert resolved == "0::output"
    assert tensor is not None
    rankme, _ = _rankme_and_eigenvalues(tensor, token_positions_as_samples=False)
    assert math.isfinite(rankme)


def test_gated_mlp_falls_back_to_out_input() -> None:
    _assert_resolves_to(_build_gated_mlp(11, 1, 2, 48, "silu"), expected_key="out::input")


def test_moe_mlp_falls_back_to_last_expert_linear_input() -> None:
    _assert_resolves_to(_build_moe_mlp(11, 1, 2, 32, 2, "silu"), expected_key="experts.1.4::input")


def test_exact_match_wins() -> None:
    model = _build_gated_mlp(11, 1, 2, 48, "silu")
    reps = collect_representation_tensors(model, torch.randn(16, 11), 2)
    tensor, resolved = resolve_representation_for_log(model, reps, "gates.0::output")
    assert resolved == "gates.0::output"
    assert tensor is not None


def test_empty_tensors_resolve_to_none() -> None:
    tensor, resolved = resolve_representation_for_log(nn.Sequential(), {}, "0::output")
    assert tensor is None
    assert resolved is None


def test_kan_falls_back_to_last_captured_output() -> None:
    kan = pytest.importorskip("kan")
    from comfy_research.engine.models.kan_model_build import build_kan_regression

    model = build_kan_regression(11, 1, 2, 5, 3, 3, 0, "silu", fast_training=True)
    model.eval()
    reps = collect_representation_tensors(model, torch.randn(16, 11), 2)
    assert "0::output" not in reps
    tensor, resolved = resolve_representation_for_log(model, reps, "0::output")
    assert resolved == "act_fun.0.base_fun::output"
    assert tensor is not None
    rankme, _ = _rankme_and_eigenvalues(tensor, token_positions_as_samples=False)
    assert math.isfinite(rankme)
