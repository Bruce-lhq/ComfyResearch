import { Handle, Position, useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useMemo } from "react";
import { CurveSeriesPlotBody } from "./CurveSeriesPlotBody";
import { SweepVizLinePlot } from "./SweepVizLinePlot";
import {
  resolveMetricCompareSource,
  type MetricCompareSourceResolution,
} from "../../graph/metricCompareSourceResolution";
import { buildMetricCompareOverlay, type MetricCompareAxis } from "../../graph/metricComparePlot";
import { dualAxisWarranted } from "../../graph/sweepVizPlot";

type MetricCompareNodeData = {
  layout?: "horizontal" | "vertical" | "overlay";
  logScaleX?: boolean;
  logScaleY?: boolean;
  /** Overlay layout: explicit Y axis for source A (undefined = left / auto). */
  overlayAxisA?: MetricCompareAxis;
  /** Overlay layout: explicit Y axis for source B (undefined = auto by metric class). */
  overlayAxisB?: MetricCompareAxis;
};

const defaults = (): Required<Omit<MetricCompareNodeData, "overlayAxisA" | "overlayAxisB">> & MetricCompareNodeData =>
  ({ layout: "horizontal", logScaleX: false, logScaleY: false });

type UnresolvedSide = Extract<MetricCompareSourceResolution, { ok: false }>;

function WarningLine({ slot, side }: { slot: "A" | "B"; side: UnresolvedSide }) {
  return (
    <div className="cr-tviz-chart-warning" role="status" title={side.reason}>
      Source {slot} ({side.nodeTitle}) can't be compared — {side.reason}
    </div>
  );
}

function SourcePanel({ side, slot, chartId }: { side: MetricCompareSourceResolution; slot: "A" | "B"; chartId: string }) {
  if (!side) {
    return <p className="cr-tensor-viz__hint">Connect source {slot}.</p>;
  }
  if (!side.ok) {
    return <WarningLine slot={slot} side={side} />;
  }
  if (side.kind === "curve") {
    return <CurveSeriesPlotBody resolved={side.value} chartId={chartId} />;
  }
  if (!side.value.plotSeries.length) {
    return <p className="cr-sweep-viz__plot-muted">No visible series to compare.</p>;
  }
  return (
    <SweepVizLinePlot
      chartId={chartId}
      series={side.value.plotSeries}
      xKey={side.value.xKey}
      xIsNumeric
      yAxisLabel={side.value.yAxisLabel}
      legendSummary={`${side.value.plotSeries.length} series`}
      logScaleX={side.value.logScaleX && side.value.canLogX}
      logScaleY={side.value.logScaleY && side.value.canLogY}
      showMarkers={false}
    />
  );
}

function AxisSelect({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: MetricCompareAxis;
  onChange: (axis: MetricCompareAxis) => void;
}) {
  return (
    <label className="cr-tviz-metric-select nodrag nopan">
      <span className="cr-tviz-metric-select__lbl">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value === "right" ? "right" : "left")} aria-label={`${label} Y axis`}>
        <option value="left">left</option>
        <option value="right">right</option>
      </select>
    </label>
  );
}

export function MetricCompareNode({ id, selected }: NodeProps) {
  const { setNodes } = useReactFlow();
  const sources = useStore(useCallback((state) => ({
    left: resolveMetricCompareSource(state.nodes as Node[], state.edges as Edge[], id, "left"),
    right: resolveMetricCompareSource(state.nodes as Node[], state.edges as Edge[], id, "right"),
  }), [id]));
  const data = useStore(useCallback((state) => ({ ...defaults(), ...((state.nodes.find((node) => node.id === id)?.data ?? {}) as MetricCompareNodeData) }), [id]));
  const patch = useCallback((next: Partial<MetricCompareNodeData>) => setNodes((nodes) => nodes.map((node) =>
    node.id === id ? { ...node, data: { ...defaults(), ...(node.data as MetricCompareNodeData), ...next } } : node,
  )), [id, setNodes]);

  const leftValue = sources.left?.ok ? sources.left.value : null;
  const rightValue = sources.right?.ok ? sources.right.value : null;
  const autoAxisB: MetricCompareAxis =
    dualAxisWarranted([...(leftValue?.plotSeries ?? []), ...(rightValue?.plotSeries ?? [])]) ? "right" : "left";
  const axisA = data.overlayAxisA ?? "left";
  const axisB = data.overlayAxisB ?? autoAxisB;
  const overlay = useMemo(
    () => buildMetricCompareOverlay(leftValue, rightValue, axisA, axisB),
    [leftValue, rightValue, axisA, axisB],
  );
  const warnings = [
    sources.left && !sources.left.ok ? { slot: "A" as const, side: sources.left } : null,
    sources.right && !sources.right.ok ? { slot: "B" as const, side: sources.right } : null,
  ].filter((x): x is { slot: "A" | "B"; side: UnresolvedSide } => x !== null);
  const missing = [sources.left ? null : "A", sources.right ? null : "B"].filter(Boolean).join(" and ");
  const empty = [
    sources.left?.ok && sources.left.value.plotSeries.length === 0 ? "A" : null,
    sources.right?.ok && sources.right.value.plotSeries.length === 0 ? "B" : null,
  ].filter(Boolean).join(" and ");

  return <div className={`cr-node cr-node--metric-compare${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-analysis)" }}>
    <div className="cr-node__header">Metric compare</div>
    <div className="cr-node__body cr-node__body--compact">
      <div className="cr-tensor-viz__io nodrag nopan" aria-label="Metric compare sockets">
        {(["left", "right"] as const).map((slot, index) => <div className="cr-tviz-socket-row" key={slot}>
          <Handle type="target" position={Position.Left} id={slot} className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket" />
          <span className="cr-tviz-socket-label">source {index === 0 ? "A" : "B"}</span>
        </div>)}
      </div>
      <p className="cr-sweep-viz__hint cr-tensor-viz__hint">Connect any 1D tensor visualization <code>compare</code> output to each source.</p>
      <div className="cr-tviz-chart-controls nodrag nopan">
        {(["horizontal", "vertical", "overlay"] as const).map((layout) => <label className="cr-tviz-check" key={layout}>
          <input type="radio" name={`${id}-layout`} checked={data.layout === layout} onChange={() => patch({ layout })} />{layout === "overlay" ? "same plot" : layout}
        </label>)}
        {data.layout === "overlay" ? <>
          <AxisSelect id={`${id}-axis-a`} label="A axis" value={axisA} onChange={(axis) => patch({ overlayAxisA: axis })} />
          <AxisSelect id={`${id}-axis-b`} label="B axis" value={axisB} onChange={(axis) => patch({ overlayAxisB: axis })} />
          <label className="cr-tviz-check cr-tviz-check--log-x"><input type="checkbox" checked={data.logScaleX} disabled={!overlay.canLogX} onChange={(event) => patch({ logScaleX: event.target.checked })} />log x</label>
          <label className="cr-tviz-check cr-tviz-check--log-y"><input type="checkbox" checked={data.logScaleY} disabled={!overlay.canLogY} onChange={(event) => patch({ logScaleY: event.target.checked })} />log y</label>
        </> : null}
      </div>
      {data.layout === "overlay" ? <div className="cr-sweep-viz__plot-wrap nodrag nopan">
        {overlay.series.length ? <SweepVizLinePlot chartId={`${id}-overlay`} series={overlay.series} xKey={overlay.xKey} xIsNumeric yAxisLabel={overlay.yAxisLabel} yAxisLabelRight={overlay.yAxisLabelRight} legendSummary={`${overlay.series.length} series`} logScaleX={data.logScaleX && overlay.canLogX} logScaleY={data.logScaleY && overlay.canLogY} dualAxis={overlay.dualAxis} showMarkers={false} /> : <p className="cr-sweep-viz__plot-muted">No source series to compare.</p>}
        {warnings.map(({ slot, side }) => <WarningLine key={slot} slot={slot} side={side} />)}
        {missing ? <p className="cr-tensor-viz__hint">Source {missing} is not connected.</p> : null}
        {empty ? <p className="cr-tensor-viz__hint">Source {empty} has no selected series.</p> : null}
      </div> : <div className={`cr-metric-compare__panels cr-metric-compare__panels--${data.layout}`}>
        {(["left", "right"] as const).map((slot, index) => <section className="cr-metric-compare__panel" key={slot}>
          <strong>Source {index === 0 ? "A" : "B"}</strong>
          <SourcePanel side={sources[slot]} slot={index === 0 ? "A" : "B"} chartId={`${id}-${slot}`} />
        </section>)}
      </div>}
    </div>
  </div>;
}
