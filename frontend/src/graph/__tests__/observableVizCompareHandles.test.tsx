// @vitest-environment jsdom

import { ReactFlowProvider } from "@xyflow/react";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { NODE_REGISTRY } from "../nodeRegistry";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/** Observable variants that render 1D curves over steps and must expose `compare`. */
const COMPARE_CAPABLE_VARIANTS = [
  "user",
  "capacity",
  "accuracy",
  "gradient_norm",
  "activation_stats",
  "weight_l2",
  "weight_product_sv",
  "hessian_eigenvalues",
  "weight_l1",
  "relu_nonlinear",
  "kan_reg",
  "layer_spectral_norm",
];

/** 2D / trajectory views are not 1D curves, so they intentionally stay out of Metric compare. */
const NON_COMPARE_VARIANTS = ["information_plane", "neuron_trajectory_2d", "embedding_trajectory", "attention_map"];

function renderObservableViz(variant: string) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <ReactFlowProvider>
        {createElement(NODE_REGISTRY.observable_viz.component, {
          id: `viz-${variant}`,
          type: "observable_viz",
          data: { vizVariant: variant },
          selected: false,
          dragging: false,
          zIndex: 0,
          isConnectable: true,
          positionAbsoluteX: 0,
          positionAbsoluteY: 0,
        } as never)}
      </ReactFlowProvider>,
    );
  });
  return { container, root };
}

describe("observable viz compare handles by variant", () => {
  it.each(COMPARE_CAPABLE_VARIANTS)("%s exposes a compare handle", (variant) => {
    const { container, root } = renderObservableViz(variant);
    expect(container.querySelector('[data-handleid="compare"]'), `variant ${variant}`).not.toBeNull();
    act(() => root.unmount());
  });

  it.each(NON_COMPARE_VARIANTS)("%s does not expose a compare handle (not a 1D curve)", (variant) => {
    const { container, root } = renderObservableViz(variant);
    expect(container.querySelector('[data-handleid="compare"]'), `variant ${variant}`).toBeNull();
    act(() => root.unmount());
  });
});
