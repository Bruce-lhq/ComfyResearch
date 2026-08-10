import type { Node } from "@xyflow/react";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";
import { resolveTrainingVizTrainSeries } from "./trainingVizLossView";
import { resolveCurveSeriesViz, type ResolvedCurveSeriesViz } from "./curveSeriesVizResolution";
import {
  buildFiniteSeries,
  resolveObservableVizCompare,
  type ResolvedObservableVizCompare,
} from "./observableVizCompareResolution";
import { resolveTensorVizCompare } from "./tensorVizCompareResolution";
import type { FlowEdge } from "./resolveUpstreamTensor";
import type { PlotSeries } from "./sweepVizPlot";

export type ResolvedMetricCompareSide =
  | { ok: true; kind: "curve"; value: ResolvedCurveSeriesViz }
  | { ok: true; kind: "series"; value: ResolvedObservableVizCompare }
  | { ok: false; reason: string; nodeTitle: string };

/** null means the side has no `compare` edge (not connected). */
export type MetricCompareSourceResolution = ResolvedMetricCompareSide | null;

const OBSERVABLE_SERIES_TYPES = new Set(["observable_viz", "observable_accuracy"]);

function nodeTitle(node: Node): string {
  const d = (node.data ?? {}) as { title?: unknown; label?: unknown };
  if (typeof d.title === "string" && d.title.trim()) return d.title.trim();
  if (typeof d.label === "string" && d.label.trim()) return d.label.trim();
  return String(node.type ?? "node");
}

/** Training viz loss curves as a step-aligned series (respects loss view and visibility toggles). */
function resolveTrainingVizSeries(node: Node): ResolvedObservableVizCompare | null {
  if (node.type !== "training_visualization") return null;
  const d = (node.data ?? {}) as Partial<TrainingVisualizationNodeData>;
  const steps = Array.isArray(d.stepTicks) ? d.stepTicks : [];
  if (steps.length < 2) return null;
  const plotSeries: PlotSeries[] = [];
  const train = resolveTrainingVizTrainSeries({
    lossHistory: d.lossHistory,
    regLossHistory: d.regLossHistory,
    lossView: d.lossView,
  });
  if (d.showTrainCurve !== false && train.length === steps.length && train.length >= 2) {
    const series = buildFiniteSeries(steps, train, `${node.id}-train`, "train", "var(--cr-chart-1)", "train loss");
    if (series) plotSeries.push(series);
  }
  const test = d.testLossHistory;
  if (
    d.showTestCurve !== false &&
    Array.isArray(test) &&
    test.length === steps.length &&
    test.length >= 2 &&
    (d.lossView ?? "loss") !== "reg"
  ) {
    const series = buildFiniteSeries(steps, test, `${node.id}-test`, "test", "var(--cr-chart-2)", "test loss", "4 3");
    if (series) plotSeries.push(series);
  }
  if (plotSeries.length === 0) return null;
  const allPoints = plotSeries.flatMap((series) => series.points);
  return {
    nodeId: node.id,
    title: "Training viz",
    plotSeries,
    xKey: "step",
    yAxisLabel: (d.yPlotMetric ?? "loss") === "perplexity" ? "perplexity" : "loss",
    canLogX: allPoints.length > 0 && allPoints.every((point) => point.x > 0),
    canLogY: allPoints.length > 0 && allPoints.every((point) => point.y > 0),
    logScaleX: !!d.logScaleX,
    logScaleY: !!d.logScaleY,
  };
}

/**
 * Generic Metric compare source resolution: any wired 1D-series viz node resolves
 * into an overlay series. A connected source that can't produce a comparable 1D
 * series returns `{ ok: false }` with a human-readable reason so the node can warn
 * instead of silently comparing nothing.
 */
export function resolveMetricCompareSource(
  nodes: Node[],
  edges: FlowEdge[],
  compareNodeId: string,
  targetHandle: "left" | "right",
): MetricCompareSourceResolution {
  const edge = edges.find((candidate) => candidate.target === compareNodeId && candidate.targetHandle === targetHandle);
  if (!edge || edge.sourceHandle !== "compare") return null;
  const source = nodes.find((candidate) => candidate.id === edge.source);
  if (!source) return { ok: false, reason: "Source node was not found in the graph.", nodeTitle: edge.source };
  const type = String(source.type ?? "");
  const title = nodeTitle(source);

  if (type === "curve_series_viz") {
    const resolved = resolveCurveSeriesViz(nodes, edges, source.id);
    return resolved
      ? { ok: true, kind: "curve", value: resolved }
      : { ok: false, reason: "Curve series viz has no resolved data yet.", nodeTitle: title };
  }

  if (OBSERVABLE_SERIES_TYPES.has(type)) {
    const resolved = resolveObservableVizCompare(source);
    return resolved
      ? { ok: true, kind: "series", value: resolved }
      : { ok: false, reason: "Observable has no series yet — connect the trainer and run training.", nodeTitle: title };
  }

  if (type === "training_visualization") {
    const resolved = resolveTrainingVizSeries(source);
    return resolved
      ? { ok: true, kind: "series", value: resolved }
      : { ok: false, reason: "Training viz has no loss series yet — run training first.", nodeTitle: title };
  }

  const tensor = resolveTensorVizCompare(nodes, edges, source);
  return tensor.resolved
    ? { ok: true, kind: "series", value: tensor.resolved }
    : { ok: false, reason: tensor.reason ?? "Not a comparable 1D series.", nodeTitle: title };
}
