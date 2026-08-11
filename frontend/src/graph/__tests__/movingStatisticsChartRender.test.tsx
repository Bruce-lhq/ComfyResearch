// @vitest-environment jsdom

import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { ProcessedCurveChart } from "../../components/nodes/MovingStatisticsNode";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function renderChart(values: number[]) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      createElement(ProcessedCurveChart, {
        values,
        stepTicks: null,
        logScaleX: false,
        logScaleY: false,
      }),
    );
  });
  return { container, root };
}

describe("MovingStatistics chart render transitions", () => {
  it("crosses empty -> one-point -> multi-point without a hook-count error", () => {
    const { container, root } = renderChart([]);
    expect(container.textContent).toContain("No output values yet.");

    act(() =>
      root.render(
        createElement(ProcessedCurveChart, {
          values: [4],
          stepTicks: null,
          logScaleX: false,
          logScaleY: false,
        }),
      ),
    );
    expect(container.textContent).toContain("Output has one point");

    act(() =>
      root.render(
        createElement(ProcessedCurveChart, {
          values: [1, 2, 3],
          stepTicks: [10, 20, 40],
          logScaleX: false,
          logScaleY: false,
        }),
      ),
    );
    expect(container.querySelector('svg[aria-label="Moving statistics curve"]')).not.toBeNull();

    act(() =>
      root.render(
        createElement(ProcessedCurveChart, {
          values: [],
          stepTicks: null,
          logScaleX: false,
          logScaleY: false,
        }),
      ),
    );
    expect(container.textContent).toContain("No output values yet.");

    act(() => root.unmount());
    container.remove();
  });
});
