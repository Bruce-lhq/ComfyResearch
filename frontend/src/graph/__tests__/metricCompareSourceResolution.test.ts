import { describe, expect, it } from "vitest";
import type { Node } from "@xyflow/react";
import { resolveMetricCompareSource } from "../metricCompareSourceResolution";

const N = (id: string, type: string, data: Record<string, unknown> = {}) =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as Node;

const E = (
  source: string,
  target: string,
  sourceHandle = "compare",
  targetHandle: "left" | "right" = "left",
) => ({ source, target, sourceHandle, targetHandle });

describe("metric compare generic source resolution", () => {
  it("training viz exposes train/test loss series", () => {
    const training = N("tv", "training_visualization", {
      stepTicks: [1, 2, 3],
      lossHistory: [2, 1, 0.5],
      testLossHistory: [2.2, 1.1, 0.6],
      showTrainCurve: true,
      showTestCurve: true,
      lossView: "loss",
    });
    const mc = N("mc", "metric_compare");
    const resolved = resolveMetricCompareSource([training, mc], [E("tv", "mc")], "mc", "left");
    expect(resolved?.ok).toBe(true);
    if (!resolved?.ok) return;
    expect(resolved.kind).toBe("series");
    expect(resolved.value.plotSeries.map((s) => s.label)).toEqual(["train", "test"]);
    expect(resolved.value.xKey).toBe("step");
  });

  it("training viz warns when no loss series exists yet", () => {
    const training = N("tv", "training_visualization", { stepTicks: [1, 2, 3] });
    const mc = N("mc", "metric_compare");
    const resolved = resolveMetricCompareSource([training, mc], [E("tv", "mc")], "mc", "left");
    expect(resolved?.ok).toBe(false);
    if (resolved?.ok) return;
    expect(resolved.reason).toContain("no loss series");
  });

  it("observable accuracy resolves as a step-aligned series", () => {
    const acc = N("obs", "observable_viz", {
      observableName: "Accuracy",
      vizVariant: "accuracy",
      stepTicks: [1, 2, 3],
      valueHistory: [0.1, 0.2, 0.9],
      testValueHistory: [0.05, 0.15, 0.8],
      showSeries: true,
      showTrainCurve: true,
      showTestCurve: true,
    });
    const mc = N("mc", "metric_compare");
    const resolved = resolveMetricCompareSource([acc, mc], [E("obs", "mc", "compare", "right")], "mc", "right");
    expect(resolved?.ok).toBe(true);
    if (!resolved?.ok) return;
    expect(resolved.kind).toBe("series");
    expect(resolved.value.plotSeries.map((s) => s.label)).toEqual(["train", "test"]);
  });

  it("1d tensor viz resolves an upstream rank-1 tensor", () => {
    const tc = N("tc", "tensor_constant", { outputTensor: { shape: [3], values: [1, 2, 3] } });
    const viz = N("viz", "tensor_viz_1d");
    const mc = N("mc", "metric_compare");
    const edges = [E("tc", "viz", "tensor", "tensor"), E("viz", "mc")];
    const resolved = resolveMetricCompareSource([tc, viz, mc], edges, "mc", "left");
    expect(resolved?.ok).toBe(true);
    if (!resolved?.ok) return;
    expect(resolved.value.plotSeries[0]!.points.map((p) => p.y)).toEqual([1, 2, 3]);
  });

  it("rank-2 tensor viz warns instead of comparing", () => {
    const tc = N("tc", "tensor_constant", { outputTensor: { shape: [2, 2], values: [1, 2, 3, 4] } });
    const viz = N("viz", "tensor_viz_general");
    const mc = N("mc", "metric_compare");
    const edges = [E("tc", "viz", "tensor", "tensor"), E("viz", "mc")];
    const resolved = resolveMetricCompareSource([tc, viz, mc], edges, "mc", "left");
    expect(resolved?.ok).toBe(false);
    if (resolved?.ok) return;
    expect(resolved.reason).toContain("rank 2");
  });

  it("unconnected side resolves to null", () => {
    const mc = N("mc", "metric_compare");
    expect(resolveMetricCompareSource([mc], [], "mc", "left")).toBeNull();
    expect(resolveMetricCompareSource([mc], [], "mc", "right")).toBeNull();
  });

  it("unknown viz type warns with a readable reason", () => {
    const table = N("t", "table_viz");
    const mc = N("mc", "metric_compare");
    const resolved = resolveMetricCompareSource([table, mc], [E("t", "mc")], "mc", "left");
    expect(resolved?.ok).toBe(false);
    if (resolved?.ok) return;
    expect(resolved.nodeTitle).toBe("table_viz");
  });
});
