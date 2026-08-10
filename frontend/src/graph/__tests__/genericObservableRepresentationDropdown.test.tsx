// @vitest-environment jsdom

import { ReactFlowProvider, useStoreApi } from "@xyflow/react";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GenericObservableNode } from "../../components/nodes/GenericObservableNode";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const ENTRIES = [
  { layer_index: 1, io: "input", representation_id: "0::input", module_name: "0", shape: [-1, 10], label: "0 input" },
  { layer_index: 1, io: "output", representation_id: "0::output", module_name: "0", shape: [-1, 64], label: "0 output" },
  { layer_index: 2, io: "input", representation_id: "1::input", module_name: "1", shape: [-1, 64], label: "1 input" },
];

type StoreApi = ReturnType<typeof useStoreApi>;
let storeApi: StoreApi | null = null;

function CaptureStore() {
  storeApi = useStoreApi();
  return null;
}

function renderRankMe() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <ReactFlowProvider>
        <CaptureStore />
        {createElement(GenericObservableNode, {
          id: "rankme",
          type: "observable_representation_rankme",
          data: { representationId: "0::output" },
          selected: false,
        })}
      </ReactFlowProvider>,
    );
  });
  return { container, root };
}

function wireTrainerModel() {
  act(() => {
    storeApi!.setState({
      nodes: [
        { id: "model", type: "mlp_model", position: { x: 0, y: 0 }, data: {} },
        { id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: {} },
        { id: "rankme", type: "observable_representation_rankme", position: { x: 0, y: 0 }, data: { representationId: "0::output" } },
      ],
      edges: [
        { id: "e1", source: "model", target: "trainer", sourceHandle: "model", targetHandle: "model" },
        { id: "e2", source: "rankme", target: "trainer", sourceHandle: "observables", targetHandle: "observables" },
      ],
    });
  });
}

describe("GenericObservableNode representation dropdown wiring", () => {
  beforeEach(() => {
    storeApi = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ entries: ENTRIES, depth: 2, summary: "probe" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stays hidden before the observable is wired to a trainer + model", () => {
    const { container, root } = renderRankMe();
    expect(container.querySelector('[aria-label="Representation ID"]')).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
    act(() => root.unmount());
  });

  it("shows the dropdown as soon as rankme → trainer → model wiring exists (no training needed)", async () => {
    const { container, root } = renderRankMe();
    expect(container.querySelector('[aria-label="Representation ID"]')).toBeNull();

    wireTrainerModel();

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(fetch).toHaveBeenCalledTimes(1);
    const [, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]! as [string, { body: string }];
    expect(JSON.parse(init.body).model_node_id).toBe("model");
    expect(container.querySelector('[aria-label="Representation ID"]')).not.toBeNull();

    act(() => root.unmount());
  });
});
