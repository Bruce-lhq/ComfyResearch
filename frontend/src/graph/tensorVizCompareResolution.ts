import type { Node } from "@xyflow/react";
import { resolveUpstreamTensor, type FlowEdge } from "./resolveUpstreamTensor";
import type { ResolvedObservableVizCompare } from "./observableVizCompareResolution";

/** Tensor visualization node kinds (1D compare applies only when the connected tensor is rank 1). */
export const TENSOR_VIZ_NODE_TYPES = new Set([
  "tensor_viz_0d",
  "tensor_viz_1d",
  "tensor_viz_2d",
  "tensor_viz_general",
]);

export type TensorVizCompareResolution = {
  resolved: ResolvedObservableVizCompare | null;
  /** Human-readable reason when the source can't be compared (non-null iff resolved is null). */
  reason: string | null;
};

/** Resolves a tensor visualization node as an index/value series for Metric compare. */
export function resolveTensorVizCompare(
  nodes: Node[],
  edges: FlowEdge[],
  node: Node,
): TensorVizCompareResolution {
  const type = String(node.type ?? "");
  if (!TENSOR_VIZ_NODE_TYPES.has(type)) {
    return { resolved: null, reason: `${type} isn't a tensor visualization node.` };
  }
  const tensor = resolveUpstreamTensor(nodes, edges, node.id, "tensor");
  if (tensor.kind !== "ok") {
    return { resolved: null, reason: tensor.detail };
  }
  if (tensor.rank !== 1) {
    return { resolved: null, reason: `Connected tensor is rank ${tensor.rank}, not a 1D series.` };
  }
  const hasAlignedTicks =
    Array.isArray(tensor.xTicks) && tensor.xTicks.length === tensor.values.length;
  const points = tensor.values
    .map((y, index) => {
      const x = hasAlignedTicks ? Number(tensor.xTicks?.[index]) : index;
      return { x, xDisplay: String(x), y, rowId: `${node.id}-${index}` };
    })
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  return {
    resolved: {
      nodeId: node.id,
      title: "1D tensor viz",
      plotSeries: points.length >= 2 ? [{ id: node.id, label: "tensor values", color: "var(--cr-chart-2)", points }] : [],
      xKey: hasAlignedTicks ? "step" : "index",
      yAxisLabel: "value",
      canLogX: points.length > 0 && points.every((point) => point.x > 0),
      canLogY: points.length > 0 && points.every((point) => point.y > 0),
      logScaleX: false,
      logScaleY: false,
    },
    reason: null,
  };
}
