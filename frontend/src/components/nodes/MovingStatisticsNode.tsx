import { Handle, Position, useReactFlow, useStore, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo } from "react";
import { hydrateResolved } from "../../graph/fetchActivationTensor";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { resolveUpstreamTensor, type FlowEdge, type FlowNodeBare } from "../../graph/resolveUpstreamTensor";
import { ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import {
  defaultMovingStatisticsData,
  type MovingStatisticOp,
  type MovingStatisticsNodeData,
} from "./movingStatisticsDefaults";
import { computeMovingStatistic, movingStatisticsOutputTicks } from "./movingStatisticsMath";
import {
  SL_CHART_H,
  SL_CHART_W,
  SL_PAD_B,
  SL_PAD_L,
  SL_PAD_T,
  SL_TICK_LEN,
  SL_Y_LABEL_X,
  slBuildSeriesPathScalar,
  slComputeBoundsScalar,
  slFormatXTickScalar,
  slFormatYTick,
  slGenerateXTicksScalar,
  slGenerateYTicks,
  slInnerBottom,
  slInnerRight,
  slXToPx,
  slYToPx,
} from "./scalarLineChartShared";

const STATISTIC_OPTIONS: { id: MovingStatisticOp; label: string }[] = [
  { id: "mean", label: "mean" },
  { id: "std", label: "std" },
];

function patchMovingStatisticsData(
  id: string,
  patch: Partial<MovingStatisticsNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const cur = defaultMovingStatisticsData((n.data ?? {}) as Partial<MovingStatisticsNodeData>);
      const next = { ...cur, ...patch };
      if (
        next.statistic === cur.statistic &&
        next.window === cur.window &&
        next.stride === cur.stride &&
        next.minPeriods === cur.minPeriods &&
        next.logScaleX === cur.logScaleX &&
        next.logScaleY === cur.logScaleY &&
        next.lastError === cur.lastError &&
        next.outputTensor === cur.outputTensor
      ) {
        return n;
      }
      return { ...n, data: next };
    }),
  );
}

export function ProcessedCurveChart({
  values,
  stepTicks,
  logScaleX,
  logScaleY,
}: {
  values: number[];
  stepTicks: number[] | null;
  logScaleX: boolean;
  logScaleY: boolean;
}) {
  const steps = useMemo(() => {
    if (stepTicks && stepTicks.length === values.length) return stepTicks;
    return values.map((_, i) => i);
  }, [values, stepTicks]);
  const bounds = useMemo(
    () => slComputeBoundsScalar(steps, values, true, logScaleX, logScaleY, false),
    [steps, values, logScaleX, logScaleY],
  );
  const path = useMemo(
    () => slBuildSeriesPathScalar(steps, values, logScaleX, logScaleY, bounds, false),
    [steps, values, logScaleX, logScaleY, bounds],
  );
  const xTicks = useMemo(
    () => slGenerateXTicksScalar(bounds.minX, bounds.maxX, logScaleX, false),
    [bounds.minX, bounds.maxX, logScaleX],
  );
  const yTicks = useMemo(
    () => slGenerateYTicks(bounds.minY, bounds.maxY, logScaleY),
    [bounds.minY, bounds.maxY, logScaleY],
  );

  if (values.length === 0) return <p className="cr-node__hint">No output values yet.</p>;
  if (values.length === 1) {
    return (
      <p className="cr-node__hint">
        Output has one point: <strong>{slFormatYTick(values[0]!, logScaleY)}</strong>
      </p>
    );
  }

  const innerBottom = slInnerBottom();
  const innerRight = slInnerRight();
  const plotMidX = SL_PAD_L + (SL_CHART_W - SL_PAD_L) / 2;
  const plotMidY = SL_PAD_T + (SL_CHART_H - SL_PAD_T - SL_PAD_B) / 2;
  const nonPositiveCount = logScaleY ? values.filter((v) => v <= 0).length : 0;
  const clipId = `moving-stats-clip-${Math.random().toString(36).slice(2, 9)}`;

  return (
    <div className="cr-tviz-chart-wrap">
      <svg
        className="cr-tviz-chart nodrag nopan"
        viewBox={`0 0 ${SL_CHART_W} ${SL_CHART_H}`}
        width={SL_CHART_W}
        height={SL_CHART_H}
        aria-label="Moving statistics curve"
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={SL_PAD_L}
              y={SL_PAD_T}
              width={SL_CHART_W - SL_PAD_L - (SL_CHART_W - innerRight)}
              height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
            />
          </clipPath>
        </defs>
        <rect
          x={SL_PAD_L}
          y={SL_PAD_T}
          width={SL_CHART_W - SL_PAD_L - (SL_CHART_W - innerRight)}
          height={SL_CHART_H - SL_PAD_T - SL_PAD_B}
          rx={4}
          className="cr-tviz-chart__plot-bg"
        />
        {yTicks.map((yt) => {
          const py = slYToPx(yt, bounds);
          return (
            <line
              key={`gy-${yt}`}
              x1={SL_PAD_L}
              y1={py}
              x2={innerRight}
              y2={py}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        {xTicks.map((xt) => {
          const px = slXToPx(xt, bounds);
          return (
            <line
              key={`gx-${xt}`}
              x1={px}
              y1={SL_PAD_T}
              x2={px}
              y2={innerBottom}
              className="cr-tviz-chart__grid"
            />
          );
        })}
        <line
          x1={SL_PAD_L}
          y1={innerBottom}
          x2={innerRight}
          y2={innerBottom}
          className="cr-tviz-chart__axis-line"
        />
        <line x1={SL_PAD_L} y1={SL_PAD_T} x2={SL_PAD_L} y2={innerBottom} className="cr-tviz-chart__axis-line" />
        {xTicks.map((xt) => {
          const px = slXToPx(xt, bounds);
          return (
            <g key={`xt-${xt}`}>
              <line
                x1={px}
                y1={innerBottom}
                x2={px}
                y2={innerBottom + SL_TICK_LEN}
                className="cr-tviz-chart__tick"
              />
              <text x={px} y={innerBottom + 12} textAnchor="middle" className="cr-tviz-chart__tick-label">
                {slFormatXTickScalar(xt, logScaleX, false)}
              </text>
            </g>
          );
        })}
        {yTicks.map((yt) => {
          const py = slYToPx(yt, bounds);
          return (
            <g key={`yt-${yt}`}>
              <line
                x1={SL_PAD_L - SL_TICK_LEN}
                y1={py}
                x2={SL_PAD_L}
                y2={py}
                className="cr-tviz-chart__tick"
              />
              <text
                x={SL_PAD_L - 6}
                y={py}
                dominantBaseline="middle"
                textAnchor="end"
                className="cr-tviz-chart__tick-label"
              >
                {slFormatYTick(yt, logScaleY)}
              </text>
            </g>
          );
        })}
        {path ? (
          <path
            d={path}
            fill="none"
            className="cr-tviz-chart__line"
            strokeWidth={1.6}
            clipPath={`url(#${clipId})`}
          />
        ) : null}
        <text x={plotMidX} y={SL_CHART_H - 2} textAnchor="middle" className="cr-tviz-chart__axis-title">
          step
        </text>
        <text
          x={SL_Y_LABEL_X}
          y={plotMidY}
          textAnchor="middle"
          dominantBaseline="middle"
          transform={`rotate(-90, ${SL_Y_LABEL_X}, ${plotMidY})`}
          className="cr-tviz-chart__axis-title"
        >
          value
        </text>
      </svg>
      {nonPositiveCount > 0 ? (
        <p className="cr-tviz-hint">
          log y clamps {nonPositiveCount} non-positive point{nonPositiveCount === 1 ? "" : "s"} to a small floor.
        </p>
      ) : null}
    </div>
  );
}

export function MovingStatisticsNode({ id, data, selected }: NodeProps) {
  const d = defaultMovingStatisticsData(data as Partial<MovingStatisticsNodeData>);
  const { setNodes } = useReactFlow();

  const resolved = useStore(
    useCallback(
      (state) =>
        resolveUpstreamTensor(state.nodes as FlowNodeBare[], state.edges as FlowEdge[], id, "tensor"),
      [id],
    ),
  );

  const update = useCallback(
    (patch: Partial<MovingStatisticsNodeData>) => patchMovingStatisticsData(id, patch, setNodes),
    [id, setNodes],
  );

  const compute = useCallback(async () => {
    const hydrated = await hydrateResolved(resolved);
    if (hydrated.kind !== "ok") {
      update({ outputTensor: null, lastError: hydrated.detail });
      return;
    }
    if (hydrated.shape.length !== 1) {
      update({
        outputTensor: null,
        lastError: "Input must be a rank-1 (1D) tensor. Reduce or slice first.",
      });
      return;
    }
    if (hydrated.values.length === 0) {
      update({ outputTensor: null, lastError: "Input series is empty." });
      return;
    }
    const outValues = computeMovingStatistic(
      hydrated.values,
      d.statistic,
      d.window,
      d.stride,
      d.minPeriods,
    );
    const outputTicks = movingStatisticsOutputTicks(
      hydrated.xTicks,
      hydrated.values.length,
      d.window,
      d.stride,
      d.minPeriods,
    );
    const bad = outValues.some((v) => !Number.isFinite(v));
    update({
      outputTensor: bad
        ? null
        : {
            shape: [outValues.length],
            values: outValues,
            ...(outputTicks ? { stepTicks: outputTicks } : {}),
          },
      lastError: bad ? "Moving statistic output contains non-finite values." : null,
    });
  }, [d.statistic, d.window, d.stride, d.minPeriods, resolved, update]);

  useEffect(() => {
    if (resolved.kind !== "none") return;
    if (!d.outputTensor?.values?.length) return;
    update({ outputTensor: null, lastError: resolved.detail });
  }, [resolved, d.outputTensor, update]);

  const outputValues = useMemo(() => d.outputTensor?.values ?? [], [d.outputTensor]);

  return (
    <div
      className={`cr-node cr-node--statistics${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header cr-node__header--trainer">
        <div className="cr-node__header--row cr-node__header--trainer-main">
          <span className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Moving statistics")}</span>
          <button type="button" className="cr-trainer-train-btn nodrag nopan" onClick={() => void compute()}>
            Compute
          </button>
        </div>
      </div>
      <div className="cr-node__body">
        <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
          <div className="cr-tviz-socket-row__left">
            <Handle
              type="target"
              position={Position.Left}
              id="tensor"
              className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
            />
            <span className="cr-tviz-socket-label">tensor</span>
          </div>
          <div className="cr-tviz-socket-row__right cr-tviz-socket-row--dual">
            <div className="cr-tviz-socket-pair">
              <span className="cr-tviz-socket-label">tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
              />
            </div>
          </div>
        </div>
        <DiscreteMultiSelect<MovingStatisticOp>
          label="statistic"
          options={STATISTIC_OPTIONS}
          value={d.statistic}
          singleSelect
          onCommit={(next) => update({ statistic: (Array.isArray(next) ? next[0] : next) ?? "mean" })}
          ariaLabel="Moving statistic"
        />
        <ComfyIntField
          label="window"
          value={d.window}
          min={1}
          max={1024}
          title="Number of samples in each trailing window."
          onCommit={(n) => {
            const w = Math.max(1, Math.floor(n));
            update({ window: w, minPeriods: Math.min(d.minPeriods, w) });
          }}
          ariaLabel="Moving statistics window"
        />
        <ComfyIntField
          label="stride"
          value={d.stride}
          min={1}
          max={1024}
          title="Hop between successive window anchors; 1 keeps every point."
          onCommit={(n) => update({ stride: Math.max(1, Math.floor(n)) })}
          ariaLabel="Moving statistics stride"
        />
        <ComfyIntField
          label="min periods"
          value={d.minPeriods}
          min={1}
          max={d.window}
          title="Minimum samples required per window; smaller values keep more head points."
          onCommit={(n) => update({ minPeriods: Math.max(1, Math.min(d.window, Math.floor(n))) })}
          ariaLabel="Moving statistics minimum periods"
        />
        <div className="cr-tviz-chart-controls cr-tviz-chart-controls--stacked nodrag nopan">
          <div className="cr-tviz-chart-controls__row cr-tviz-chart-controls__row--scales">
            <label className="cr-tviz-check cr-tviz-check--log-x">
              <input
                type="checkbox"
                checked={!!d.logScaleX}
                onChange={(e) => update({ logScaleX: e.target.checked })}
              />
              log x
            </label>
            <label className="cr-tviz-check cr-tviz-check--log-y">
              <input
                type="checkbox"
                checked={!!d.logScaleY}
                onChange={(e) => update({ logScaleY: e.target.checked })}
              />
              log y
            </label>
          </div>
        </div>
        <p className="cr-node__hint">
          Trailing rolling mean or std over a 1D series; stride hops the anchor, min periods trims the head.
        </p>
        <ProcessedCurveChart
          values={outputValues}
          stepTicks={d.outputTensor?.stepTicks ?? null}
          logScaleX={!!d.logScaleX}
          logScaleY={!!d.logScaleY}
        />
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
