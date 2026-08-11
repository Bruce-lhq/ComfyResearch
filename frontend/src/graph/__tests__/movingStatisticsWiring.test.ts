import { describe, expect, it } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import { isValidCanvasConnection } from "../connectionRules";

function probe(srcType: string, tgtType: string, sh: string, th: string): boolean {
  const nodes = [
    { id: "s", type: srcType, position: { x: 0, y: 0 }, data: {} },
    { id: "t", type: tgtType, position: { x: 100, y: 0 }, data: {} },
  ] as unknown as Node[];
  return isValidCanvasConnection(
    { source: "s", target: "t", sourceHandle: sh, targetHandle: th } as never,
    nodes,
    [] as Edge[],
  );
}

describe("moving statistics wiring", () => {
  it("feeds from the tensor selector hub", () => {
    expect(probe("tensor_selector", "moving_statistics", "", "tensor")).toBe(true);
    expect(probe("tensor_selector", "moving_statistics", "tensor_1", "tensor")).toBe(true);
  });

  it("feeds 1D tensor viz and observable viz flow targets", () => {
    expect(probe("moving_statistics", "tensor_viz_1d", "tensor", "tensor")).toBe(true);
    expect(probe("moving_statistics", "tensor_viz_general", "tensor", "tensor")).toBe(true);
    expect(probe("moving_statistics", "tensor_viz_2d", "tensor", "tensor")).toBe(true);
  });

  it("feeds downstream analysis targets like other 1D transforms", () => {
    expect(probe("moving_statistics", "statistics", "tensor", "tensor")).toBe(true);
    expect(probe("moving_statistics", "tensor_slicing", "tensor", "tensor")).toBe(true);
    expect(probe("moving_statistics", "basic_calculator", "tensor", "tensor_1")).toBe(true);
  });

  it("cannot feed metric compare directly (needs a viz compare handle)", () => {
    expect(probe("moving_statistics", "metric_compare", "tensor", "left")).toBe(false);
  });
});
