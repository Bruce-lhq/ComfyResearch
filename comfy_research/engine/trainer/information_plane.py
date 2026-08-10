"""Bounded, binned mutual-information estimates for layer trajectories.

This module is intentionally independent of the information-bottleneck
reproduction.  It is a best-effort observable: callers receive an empty point
set when a model has no eligible activations or a measurement cannot be made.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from comfy_research.engine.trainer.model_helpers import _forward_reg


def _entropy_bits(ids: np.ndarray) -> float:
    counts = np.bincount(ids.astype(np.int64, copy=False))
    probs = counts[counts > 0].astype(np.float64)
    if probs.size == 0:
        return float("nan")
    probs /= probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _mutual_information_bits(state_ids: np.ndarray, labels: np.ndarray) -> float:
    states = np.asarray(state_ids, dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim > 1 and raw_labels.shape[1] > 1:
        raw_labels = np.argmax(raw_labels, axis=1)
    raw_labels = raw_labels.reshape(-1)
    if states.size == 0 or states.size != raw_labels.size:
        return float("nan")
    _, label_ids = np.unique(raw_labels, return_inverse=True)
    n_states = int(states.max()) + 1
    n_labels = int(label_ids.max()) + 1
    joint = np.bincount(states * n_labels + label_ids, minlength=n_states * n_labels).reshape(n_states, n_labels)
    p_joint = joint.astype(np.float64) / float(states.size)
    p_state = p_joint.sum(axis=1, keepdims=True)
    p_label = p_joint.sum(axis=0, keepdims=True)
    mask = p_joint > 0
    ratio = p_joint[mask] / (p_state @ p_label)[mask]
    return float(np.sum(p_joint[mask] * np.log2(ratio)))


def _state_ids(quantized: np.ndarray) -> np.ndarray:
    """Use an integer row code when safe; otherwise use the general fallback."""
    values = np.asarray(quantized, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.empty((0,), dtype=np.int64)
    shifted = values - values.min(axis=0, keepdims=True)
    total, powers = 1, []
    for base in (shifted.max(axis=0) + 1).tolist():
        powers.append(total)
        total *= int(base)
        if total - 1 > np.iinfo(np.int64).max:
            break
    if len(powers) == values.shape[1] and total - 1 <= np.iinfo(np.int64).max:
        _, ids = np.unique(shifted @ np.asarray(powers, dtype=np.int64), return_inverse=True)
        return ids
    _, ids = np.unique(values, axis=0, return_inverse=True)
    return ids


def binned_information_pair(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 30,
    strategy: str = "uniform_intervals",
) -> tuple[float, float]:
    """Return ``(I(X;T), I(T;Y))`` using the selected deterministic binning."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    n_bins = max(2, int(bins))
    if strategy == "idnns_equal_points":
        anchors = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
        quantized = np.digitize(np.clip(values, -1.0, 1.0), anchors).astype(np.int16) - 1
        np.clip(quantized, 0, n_bins - 1, out=quantized)
    elif strategy == "uniform_intervals":
        clipped = np.clip(values, -1.0, 1.0)
        quantized = np.floor((clipped + 1.0) * (n_bins / 2.0)).astype(np.int16)
        np.clip(quantized, 0, n_bins - 1, out=quantized)
    elif strategy == "adaptive_minmax":
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            quantized = np.zeros_like(values, dtype=np.int16)
        else:
            quantized = np.floor((values - lo) * (n_bins / (hi - lo))).astype(np.int16)
            np.clip(quantized, 0, n_bins - 1, out=quantized)
    elif strategy == "saxe_fixed_width_0_07":
        quantized = np.floor(values / 0.07).astype(np.int32)
    else:
        raise ValueError(f"unknown information-plane binning strategy: {strategy}")
    ids = _state_ids(quantized)
    if ids.size != values.shape[0]:
        return float("nan"), float("nan")
    return _entropy_bits(ids), _mutual_information_bits(ids, labels)


def _quantize_column(values: np.ndarray, *, bins: int, strategy: str) -> np.ndarray:
    """Bin one feature column with the selected deterministic strategy."""
    v = np.asarray(values, dtype=np.float64)
    n_bins = max(2, int(bins))
    if strategy == "idnns_equal_points":
        anchors = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
        q = np.digitize(np.clip(v, -1.0, 1.0), anchors).astype(np.int16) - 1
        return np.clip(q, 0, n_bins - 1)
    if strategy == "uniform_intervals":
        clipped = np.clip(v, -1.0, 1.0)
        q = np.floor((clipped + 1.0) * (n_bins / 2.0)).astype(np.int16)
        return np.clip(q, 0, n_bins - 1)
    if strategy == "adaptive_minmax":
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo:
            return np.zeros_like(v, dtype=np.int16)
        q = np.floor((v - lo) * (n_bins / (hi - lo))).astype(np.int16)
        return np.clip(q, 0, n_bins - 1)
    if strategy == "saxe_fixed_width_0_07":
        return np.floor(v / 0.07).astype(np.int32)
    raise ValueError(f"unknown information-plane binning strategy: {strategy}")


def _row_codes(rows: np.ndarray) -> np.ndarray:
    """Integer codes for rows of an (n, d) array (unique rows get unique codes)."""
    arr = np.asarray(rows)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    _, codes = np.unique(arr, axis=0, return_inverse=True)
    return codes.astype(np.int64)


def _featurewise_information_pair(
    values: np.ndarray,
    labels: np.ndarray,
    input_codes: np.ndarray,
    *,
    bins: int,
    strategy: str,
    max_features: int,
) -> tuple[float, float]:
    """Return (I(X;T), I(T;Y)) estimated per feature column and averaged.

    Per-column binning avoids the all-rows-unique row-code collapse that made
    both coordinates constant on high-dimensional token outputs (B-007).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return float("nan"), float("nan")
    if not np.isfinite(values).all():
        return float("nan"), float("nan")
    n = values.shape[0]
    label_ids = np.asarray(labels).reshape(-1)
    if label_ids.size != n or input_codes.size != n:
        return float("nan"), float("nan")
    features = values[:, :max_features] if values.shape[1] > max_features else values
    ix_vals: list[float] = []
    iy_vals: list[float] = []
    for col in range(features.shape[1]):
        quantized = _quantize_column(features[:, col], bins=bins, strategy=strategy)
        ix = _mutual_information_bits(quantized, input_codes)
        iy = _mutual_information_bits(quantized, label_ids)
        if math.isfinite(ix) and math.isfinite(iy):
            ix_vals.append(ix)
            iy_vals.append(iy)
    if not ix_vals:
        return float("nan"), float("nan")
    return float(np.mean(ix_vals)), float(np.mean(iy_vals))


def information_plane_for_model(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 30,
    include_output: bool = True,
    binning: str = "uniform_intervals",
    output_mapping: str = "tanh",
    max_features: int = 8,
) -> list[list[float]]:
    """Return one binned information point per captured layer (and optional output).

    The first coordinate is a real ``I(X;T)`` estimate (mutual information
    between the binned input encoding and the representation); the second is
    ``I(T;Y)`` (B-007).  Hooks prefer ``nn`` activation modules; models that
    only use functional activations (RWKV/hyena/transformer) fall back to
    capturing every module output.  High-dimensional activations are measured
    per feature column and aggregated so both coordinates vary with training
    instead of degenerating to constants.  Training mode is restored even if
    the forward pass fails.
    """
    captured: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    activation_types = (nn.Tanh, nn.Sigmoid, nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU)

    def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        if isinstance(output, torch.Tensor) and output.ndim >= 1 and output.shape[0] == x.shape[0]:
            captured.append(output.detach())

    was_training = model.training

    def _run() -> torch.Tensor | None:
        try:
            model.eval()
            with torch.no_grad():
                return _forward_reg(model, x)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

    # Preferred: nn activation modules define the semantic "layers".
    for name, module in model.named_modules():
        if name and isinstance(module, activation_types):
            handles.append(module.register_forward_hook(hook))
    output = _run()
    # B-007: functional-activation models (RWKV/hyena) expose no nn activation
    # modules ? capture every module output instead.
    if not captured:
        handles = []
        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(hook))
        output = _run()

    if include_output and isinstance(output, torch.Tensor) and output.ndim >= 1:
        if output_mapping == "probability":
            if output.ndim >= 2 and int(output.shape[-1]) > 1:
                output = torch.softmax(output, dim=-1)
            else:
                output = torch.sigmoid(output)
        elif output_mapping == "tanh":
            output = torch.tanh(output)
        elif output_mapping == "signed_probability":
            if output.ndim >= 2 and int(output.shape[-1]) > 1:
                output = torch.softmax(output, dim=-1) * 2.0 - 1.0
            else:
                output = torch.tanh(output)
        else:
            raise ValueError(f"unknown information-plane output mapping: {output_mapping}")
        captured.append(output.detach())

    x_np = x.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    points: list[list[float]] = []
    for value in captured:
        arr = value.detach().cpu().numpy()
        per_position = arr.ndim >= 3
        if per_position:
            b, t = int(arr.shape[0]), int(arr.shape[1])
            arr = arr.reshape(b * t, -1)
        elif arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            continue
        n = int(arr.shape[0])
        if n == 0:
            continue
        if per_position:
            if x_np.ndim >= 2 and x_np.shape[0] == b and x_np.shape[1] == t:
                inputs_np = x_np.reshape(n, -1)
            elif x_np.shape[0] == b:
                inputs_np = np.repeat(x_np.reshape(b, -1), t, axis=0)
            else:
                inputs_np = x_np.reshape(n, -1)
            if labels_np.size == n:
                labels_n = labels_np.reshape(-1)
            elif labels_np.size == b:
                labels_n = np.repeat(labels_np.reshape(-1), t)
            else:
                continue
        else:
            inputs_np = x_np
            labels_n = labels_np.reshape(-1)
        if labels_n.size != n:
            continue
        ix, iy = _featurewise_information_pair(
            arr,
            labels_n,
            _row_codes(inputs_np),
            bins=bins,
            strategy=binning,
            max_features=max_features,
        )
        if math.isfinite(ix) and math.isfinite(iy):
            points.append([float(ix), float(iy)])
    return points
