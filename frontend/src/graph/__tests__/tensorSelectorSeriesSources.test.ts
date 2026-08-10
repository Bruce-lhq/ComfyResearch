import { describe, expect, it } from "vitest";
import { isValidCanvasConnection } from "../connectionRules";
import {
  resolveUpstreamTensor,
  tensorChoicesFromSourceHandle,
  type FlowEdge,
} from "../resolveUpstreamTensor";

const N = (id: string, type: string, data: Record<string, unknown> = {}) =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as never;

const check = (srcType: string, sh: string, tgtType: string, th: string, srcData: Record<string, unknown> = {}) =>
  isValidCanvasConnection(
    { source: "s", target: "t", sourceHandle: sh, targetHandle: th } as never,
    [N("s", srcType, srcData), N("t", tgtType)],
    [],
  );

describe("tensor selector accepts every 1D curve viz source", () => {
  it("lists gradient norm / hessian-style variants into the selector", () => {
    for (const variant of [
      "gradient_norm",
      "hessian_eigenvalues",
      "activation_stats",
      "weight_product_sv",
      "kan_reg",
      "layer_spectral_norm",
      "weight_l2",
      "weight_l1",
      "capacity",
      "accuracy",
      "relu_nonlinear",
      "user",
    ]) {
      expect(
        check("observable_viz", "out_tensor", "tensor_selector", "tensor_list", { vizVariant: variant }),
        `variant ${variant}`,
      ).toBe(true);
    }
  });

  it("keeps 2D / trajectory variants out of the selector", () => {
    for (const variant of ["embedding_trajectory", "neuron_trajectory_2d", "information_plane", "attention_map"]) {
      expect(
        check("observable_viz", "out_tensor", "tensor_selector", "tensor_list", { vizVariant: variant }),
        `variant ${variant}`,
      ).toBe(false);
    }
  });

  it("lets tensor viz passthrough feed the selector", () => {
    expect(check("tensor_viz_1d", "out_tensor", "tensor_selector", "tensor_list")).toBe(true);
    expect(check("tensor_viz_general", "out_tensor", "tensor_selector", "tensor_list")).toBe(true);
  });

  it("routes the selector batch output into 2D / general tensor viz only", () => {
    expect(check("tensor_selector", "tensor_list", "tensor_viz_2d", "tensor_list")).toBe(true);
    expect(check("tensor_selector", "tensor_list", "tensor_viz_general", "tensor_list")).toBe(true);
    expect(check("tensor_selector", "tensor_list", "tensor_viz_1d", "tensor_list")).toBe(false);
    expect(check("tensor_selector", "tensor_list", "tensor_viz_2d", "tensor")).toBe(false);
  });

  it("lets every 1D series source feed statistics and curve analysis directly", () => {
    expect(check("observable_viz", "out_tensor", "statistics", "tensor", { vizVariant: "user" })).toBe(true);
    expect(check("observable_viz", "out_tensor", "statistics2", "tensor_1", { vizVariant: "gradient_norm" })).toBe(true);
    expect(check("tensor_viz_1d", "out_tensor", "smoothing_curve", "tensor")).toBe(true);
    expect(check("tensor_viz_1d", "out_tensor", "derivative_curve", "tensor")).toBe(true);
  });
});

describe("observable viz series choices", () => {
  it("lists every row of a multi-series panel by its label", () => {
    const viz = N("viz", "observable_viz", {
      vizVariant: "gradient_norm",
      valueHistories: [[1, 2], [3, 4]],
      seriesLabels: ["global", "weight.bias"],
    });
    expect(tensorChoicesFromSourceHandle([viz], viz, "out_tensor")).toEqual([
      { id: "series::0", label: "global" },
      { id: "series::1", label: "weight.bias" },
    ]);
  });

  it("falls back to variant-aware labels when seriesLabels is absent", () => {
    const hessian = N("viz", "observable_viz", {
      vizVariant: "hessian_eigenvalues",
      valueHistories: [[1, 2], [3, 4], [5, 6]],
    });
    expect(tensorChoicesFromSourceHandle([hessian], hessian, "out_tensor")).toEqual([
      { id: "series::0", label: "eigenvalue 1" },
      { id: "series::1", label: "eigenvalue 2" },
      { id: "series::2", label: "eigenvalue 3" },
    ]);
  });

  it("keeps classic single-series ids for scalar panels", () => {
    const acc = N("viz", "observable_viz", { vizVariant: "accuracy", valueHistory: [0.1, 0.2] });
    expect(tensorChoicesFromSourceHandle([acc], acc, "out_tensor")).toEqual([
      { id: "train_accuracy", label: "train acc" },
      { id: "test_accuracy", label: "test acc" },
    ]);
  });

  it("exposes a single tensor choice for tensor viz passthrough", () => {
    const viz = N("viz", "tensor_viz_1d");
    expect(tensorChoicesFromSourceHandle([viz], viz, "out_tensor")).toEqual([
      { id: "tensor", label: "tensor" },
    ]);
  });
});

describe("selector resolves a picked multi-series row", () => {
  it("returns the selected valueHistories row as a rank-1 series", () => {
    const viz = N("viz", "observable_viz", {
      vizVariant: "gradient_norm",
      valueHistories: [[1, 2, 3], [4, 5, 6]],
      seriesLabels: ["global", "weight.bias"],
      stepTicks: [1, 2, 3],
    });
    const selector = N("sel", "tensor_selector", {
      selectedTensorKeys: ["series::1"],
      selectedTensorKey: "series::1",
    });
    const consumer = N("viz1d", "tensor_viz_1d");
    const edges = [
      { source: "viz", target: "sel", sourceHandle: "out_tensor", targetHandle: "tensor_list" },
      { source: "sel", target: "viz1d", sourceHandle: "selected_tensor", targetHandle: "tensor" },
    ] as FlowEdge[];
    const resolved = resolveUpstreamTensor([viz, selector, consumer], edges, "viz1d", "tensor");
    expect(resolved.kind).toBe("ok");
    if (resolved.kind !== "ok") return;
    expect(resolved.rank).toBe(1);
    expect(resolved.values).toEqual([4, 5, 6]);
    expect(resolved.xTicks).toEqual([1, 2, 3]);
  });

  it("keeps the classic single-series id working for scalar panels", () => {
    const viz = N("viz", "observable_viz", {
      vizVariant: "user",
      observableName: "My drift",
      valueHistory: [7, 8, 9],
    });
    const selector = N("sel", "tensor_selector", {
      selectedTensorKeys: ["user_observable"],
      selectedTensorKey: "user_observable",
    });
    const consumer = N("viz1d", "tensor_viz_1d");
    const edges = [
      { source: "viz", target: "sel", sourceHandle: "out_tensor", targetHandle: "tensor_list" },
      { source: "sel", target: "viz1d", sourceHandle: "selected_tensor", targetHandle: "tensor" },
    ] as FlowEdge[];
    const resolved = resolveUpstreamTensor([viz, selector, consumer], edges, "viz1d", "tensor");
    expect(resolved.kind).toBe("ok");
    if (resolved.kind !== "ok") return;
    expect(resolved.values).toEqual([7, 8, 9]);
    expect(resolved.xTicks).toBeUndefined();
  });

  it("resolves the tensor_list batch output for multi-select", () => {
    const viz = N("viz", "observable_viz", {
      vizVariant: "gradient_norm",
      valueHistories: [[1, 2, 3], [4, 5, 6]],
      seriesLabels: ["global", "weight.bias"],
      stepTicks: [10, 20, 30],
    });
    const selector = N("sel", "tensor_selector", {
      selectedTensorKeys: ["series::0", "series::1"],
      selectedTensorKey: "series::0",
    });
    const consumer = N("viz2d", "tensor_viz_2d");
    const edges = [
      { source: "viz", target: "sel", sourceHandle: "out_tensor", targetHandle: "tensor_list" },
      { source: "sel", target: "viz2d", sourceHandle: "tensor_list", targetHandle: "tensor_list" },
    ] as FlowEdge[];
    const resolved = resolveUpstreamTensor([viz, selector, consumer], edges, "viz2d", "tensor_list");
    expect(resolved.kind).toBe("tensor_list");
    if (resolved.kind !== "tensor_list") return;
    expect(resolved.rows.map((r) => r.label)).toEqual(["global", "weight.bias"]);
    expect(resolved.rows[0]!.values).toEqual([1, 2, 3]);
    expect(resolved.rows[1]!.values).toEqual([4, 5, 6]);
    expect(resolved.rows[0]!.xTicks).toEqual([10, 20, 30]);
  });
});
