import { describe, expect, it } from "vitest";
import type { Node } from "@xyflow/react";

import { canNodeTypeConnectToAny } from "../canvasConnectivity";
import { inputSocketsFor, outputSocketsFor } from "../nodeSockets";

const N = (id: string, type: string, data: Record<string, unknown> = {}) =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as unknown as Node;

/** Minimal trainer graph: linear dataset -> MLP, MSE, Adam -> trainer. */
const trainerGraph = () => [
  N("ds", "linear_dataset"),
  N("m", "mlp_model"),
  N("l", "mse_loss"),
  N("o", "adam_optimizer"),
  N("t", "trainer"),
];

describe("nodeSockets", () => {
  it("exposes the trainer observables input and built-in observable outputs", () => {
    expect(inputSocketsFor("trainer")).toContain("observables");
    expect(outputSocketsFor("observable_accuracy")).toContain("observables");
  });

  it("exposes no sockets for language/comment-only nodes", () => {
    expect(outputSocketsFor("url_node")).toHaveLength(0);
  });
});

describe("canNodeTypeConnectToAny", () => {
  it("accepts observables that fit the trainer graph", () => {
    expect(canNodeTypeConnectToAny("observable_accuracy", trainerGraph())).toBe(true);
    expect(canNodeTypeConnectToAny("observable_train_test_gap", trainerGraph())).toBe(true);
  });

  it("accepts an lr schedule feeding the optimizer (candidate -> existing)", () => {
    expect(canNodeTypeConnectToAny("lr_schedule", trainerGraph())).toBe(true);
    expect(canNodeTypeConnectToAny("mup_lr_schedule", trainerGraph())).toBe(true);
  });

  it("accepts a second dataset of the same family", () => {
    expect(canNodeTypeConnectToAny("linear_dataset", trainerGraph())).toBe(true);
  });

  it("accepts atomic layers as a model source for the trainer", () => {
    expect(canNodeTypeConnectToAny("linear_layer", trainerGraph())).toBe(true);
  });

  it("rejects nodes with no matching socket on the trainer graph", () => {
    for (const t of [
      "tensor_constant",
      "comment",
      "table_viz",
      "pca",
      "url_node",
      "hypothesis",
      "activation",
    ]) {
      expect(canNodeTypeConnectToAny(t, trainerGraph())).toBe(false);
    }
  });

  it("accepts tensor chains when the existing model is in input-output mode", () => {
    const canvas = [N("m", "mlp_model", { ioMode: "input-output" })];
    for (const t of ["linear_layer", "layer_norm_layer", "tensor_constant", "reshape"]) {
      expect(canNodeTypeConnectToAny(t, canvas)).toBe(true);
    }
  });

  it("classifies nothing as connectable on an empty canvas", () => {
    expect(canNodeTypeConnectToAny("mlp_model", [])).toBe(false);
  });

  it("accepts optimizers feeding a bare trainer (candidate -> existing)", () => {
    const canvas = [N("t", "trainer")];
    for (const t of ["adam_optimizer", "sgd_optimizer", "adamw_optimizer"]) {
      expect(canNodeTypeConnectToAny(t, canvas)).toBe(true);
    }
    // An lr schedule alone cannot reach the trainer; it needs an optimizer.
    expect(canNodeTypeConnectToAny("lr_schedule", canvas)).toBe(false);
  });

  it("rejects tensor strip nodes when only a dataset is on the canvas", () => {
    const canvas = [N("ds", "linear_dataset")];
    for (const t of ["tensor_splitter", "reshape", "flatten", "softmax", "causal_mask"]) {
      expect(canNodeTypeConnectToAny(t, canvas)).toBe(false);
    }
  });

  it("accepts a tensor selector fed by a dataset (dataset -> tensor_list)", () => {
    const canvas = [N("ds", "linear_dataset")];
    expect(canNodeTypeConnectToAny("tensor_selector", canvas)).toBe(true);
  });
});
