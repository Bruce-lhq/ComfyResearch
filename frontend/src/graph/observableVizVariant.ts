import type { Node } from "@xyflow/react";
import {
  nodeRegistryObservableVizVariant,
  type ObservableVizVariant,
} from "./nodeRegistrySpec";

/** Unified observable viz node (`type: observable_viz`) — visualization is chosen by `vizVariant` in `data`. */
export type { ObservableVizVariant };

const LEGACY_TYPE_TO_VARIANT: Record<string, ObservableVizVariant> = {
  observable_viz_weight_l2: "weight_l2",
  observable_viz_weight_l1: "weight_l1",
  observable_viz_relu_nonlinear: "relu_nonlinear",
  observable_viz_user: "user",
  observable_viz_embedding_trajectory: "embedding_trajectory",
  observable_viz_neuron_trajectory_2d: "neuron_trajectory_2d",
};

/** Dedicated observable metric viz nodes (same ``out_tensor`` / chart data shape as ``observable_viz``). */
const STANDALONE_OBSERVABLE_VIZ_METRIC_TO_VARIANT: Record<string, ObservableVizVariant> = {
  observable_accuracy: "accuracy",
};

/** Load-time migration: legacy per-metric node kinds → `observable_viz` + `vizVariant`. */
export function migrateObservableVizNodeTypes(nodes: Node[]): Node[] {
  const byId = new Map(nodes.map((x) => [x.id, x]));
  return nodes.map((n) => {
    let out = n;
    const legacy = n.type != null ? LEGACY_TYPE_TO_VARIANT[String(n.type)] : undefined;
    if (legacy) {
      const data = { ...(n.data as Record<string, unknown>) };
      if (data.vizVariant == null) data.vizVariant = legacy;
      out = { ...n, type: "observable_viz", data };
    }
    if (out.type === "observable_viz") {
      const d = { ...(out.data as Record<string, unknown>) };
      const pid = d.pairedObservableId;
      const rawV = d.vizVariant;
      const vStr = typeof rawV === "string" ? rawV.trim() : "";
      if (!vStr && typeof pid === "string") {
        const p = byId.get(pid);
        const inferred = p?.type ? nodeRegistryObservableVizVariant(String(p.type)) : undefined;
        if (inferred) {
          d.vizVariant = inferred;
          return { ...out, data: d };
        }
      }
      if (d.vizVariant === "user" && typeof pid === "string") {
        const p = byId.get(pid);
        if (p?.type === "observable_accuracy") {
          d.vizVariant = "accuracy";
          return { ...out, data: d };
        }
      }
    }
    return out;
  });
}

export function getObservableVizVariant(node: Node): ObservableVizVariant | null {
  if (node.type === "observable_viz") {
    const v = (node.data as { vizVariant?: string }).vizVariant;
    if (
      v === "weight_l2" ||
      v === "weight_l1" ||
      v === "capacity" ||
      v === "accuracy" ||
      v === "relu_nonlinear" ||
      v === "kan_reg" ||
      v === "hessian_eigenvalues" ||
      v === "gradient_norm" ||
      v === "activation_stats" ||
      v === "user" ||
      v === "embedding_trajectory" ||
      v === "weight_product_sv" ||
      v === "neuron_trajectory_2d" ||
      v === "information_plane" ||
      v === "layer_spectral_norm" ||
      v === "attention_map"
    ) {
      return v;
    }
    return null;
  }
  const t = node.type != null ? String(node.type) : "";
  return LEGACY_TYPE_TO_VARIANT[t] ?? STANDALONE_OBSERVABLE_VIZ_METRIC_TO_VARIANT[t] ?? null;
}

export function isObservableVizFlowType(t: string | undefined): boolean {
  return t === "observable_viz" || (t != null && LEGACY_TYPE_TO_VARIANT[t] != null);
}

/** Variants whose `out_tensor` re-exports a 1D scalar series (``valueHistory``) for tensor viz / analysis chains. */
const TENSOR_CHAIN_VARIANTS = new Set<ObservableVizVariant>([
  "weight_l2",
  "weight_l1",
  "capacity",
  "accuracy",
  "relu_nonlinear",
  /** Scalar user-style charts (train vs test gap, embedding drift, attention stats, …). */
  "user",
]);

export function observableVizAllowsTensorVizChain(node: Node | undefined): boolean {
  if (!node) return false;
  const v = getObservableVizVariant(node);
  return v != null && TENSOR_CHAIN_VARIANTS.has(v);
}

/** All 1D step-aligned series variants. Their `out_tensor` carries a rank-1 curve
 * (single series via `valueHistory`, or multi-row via `valueHistories`). 2D /
 * trajectory variants (embedding_trajectory, neuron_trajectory_2d, information_plane,
 * attention_map) are excluded — their `out_tensor` is not a 1D series. */
export const TENSOR_SERIES_VARIANTS: ReadonlySet<ObservableVizVariant> = new Set([
  ...TENSOR_CHAIN_VARIANTS,
  "gradient_norm",
  "hessian_eigenvalues",
  "activation_stats",
  "weight_product_sv",
  "kan_reg",
  "layer_spectral_norm",
]);

/** True when the node is an observable viz whose `out_tensor` is a 1D step-aligned series. */
export function observableVizHasTensorSeries(node: Node | undefined): boolean {
  if (!node) return false;
  const v = getObservableVizVariant(node);
  return v != null && TENSOR_SERIES_VARIANTS.has(v);
}
