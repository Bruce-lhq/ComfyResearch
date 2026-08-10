import { normalizeCurveSeriesProgress } from "./curveSeriesPlot";
import type { ResolvedCurveSeriesViz } from "./curveSeriesVizResolution";
import type { ResolvedObservableVizCompare } from "./observableVizCompareResolution";
import { decoratePlotSeries, dualAxisWarranted, type PlotSeries } from "./sweepVizPlot";

export type MetricCompareAxis = "left" | "right";

export type MetricCompareOverlay = {
  series: PlotSeries[];
  xKey: string;
  canLogX: boolean;
  canLogY: boolean;
  /** True when source A and source B are assigned to opposite Y axes. */
  dualAxis: boolean;
  /** Left Y axis label — the y label of whichever source sits on the left axis. */
  yAxisLabel: string;
  /** Right Y axis label — the y label of whichever source sits on the right axis. */
  yAxisLabelRight: string;
};

export type MetricCompareSource = ResolvedCurveSeriesViz | ResolvedObservableVizCompare;

function decorateSourceSeries(
  source: MetricCompareSource,
  slot: "left" | "right",
  normalizeX: boolean,
  yAxis: MetricCompareAxis,
): PlotSeries[] {
  return source.plotSeries.map((series, index) => {
    const normalized = normalizeCurveSeriesProgress(series.points.map((point) => point.x));
    return {
      ...series,
      yAxis,
      id: `${slot}-${series.id}`,
      label: `Source ${slot === "left" ? "A" : "B"} — ${source.title}: ${series.label}`,
      color: slot === "right" ? ["var(--cr-chart-8)", "var(--cr-chart-5)", "var(--cr-chart-4)", "var(--cr-chart-7)"][index % 4]! : series.color,
      strokeDasharray: slot === "right" ? "6 3" : series.strokeDasharray,
      points: normalizeX ? series.points.map((point, pointIndex) => ({ ...point, x: normalized[pointIndex]! })) : series.points,
    };
  });
}

/**
 * Builds the overlay plot. Source A is placed on `axisA` (default left) and
 * source B on `axisB`; when `axisB` is omitted it follows the metric-class
 * heuristic (accuracy-like series go right, everything else left) so loss vs
 * accuracy comparisons get dual axes automatically.
 */
export function buildMetricCompareOverlay(
  left: MetricCompareSource | null,
  right: MetricCompareSource | null,
  axisA: MetricCompareAxis = "left",
  axisB?: MetricCompareAxis,
): MetricCompareOverlay {
  const available = [left, right].filter((source): source is MetricCompareSource => source !== null);
  const isCurve = (source: MetricCompareSource): source is ResolvedCurveSeriesViz => "settings" in source;
  const sameXMode = available.length < 2 || available.every((source) =>
    isCurve(source) && isCurve(available[0]!)
      ? source.settings.plotXMode === available[0]!.settings.plotXMode
      : source.xKey === available[0]!.xKey,
  );
  const normalizeX = !sameXMode;
  const autoAxisB: MetricCompareAxis =
    available.length >= 2 && dualAxisWarranted(available.flatMap((source) => source.plotSeries)) ? "right" : "left";
  const axisBEff = axisB ?? autoAxisB;
  const series = decoratePlotSeries(
    available.flatMap((source) =>
      decorateSourceSeries(
        source,
        source === left ? "left" : "right",
        normalizeX,
        source === left ? axisA : axisBEff,
      ),
    ),
  );
  const leftSource = axisA === "left" ? left : right;
  const rightSource = axisA === "left" ? right : left;
  const yAxisLabel = leftSource ? (isCurve(leftSource) ? leftSource.yAxisLabel : leftSource.yAxisLabel) : "value";
  const yAxisLabelRight = rightSource
    ? isCurve(rightSource)
      ? rightSource.yAxisLabelRight
      : rightSource.yAxisLabel
    : "value";
  const dualAxis =
    axisA !== axisBEff && series.some((s) => s.yAxis === "left") && series.some((s) => s.yAxis === "right");
  return {
    series,
    xKey: normalizeX ? "progress %" : available[0]?.xKey ?? "step",
    canLogX: !normalizeX && available.length > 0 && available.every((source) => source.canLogX),
    canLogY: available.length > 0 && available.every((source) => source.canLogY),
    dualAxis,
    yAxisLabel,
    yAxisLabelRight,
  };
}
