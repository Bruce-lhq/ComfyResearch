/**
 * Canvas socket metadata for the sidebar connectivity classifier.
 *
 * The classifier must know which handles a node type actually exposes; otherwise
 * isValidCanvasConnection's permissive default-true tail would classify almost
 * every type as "connectable". This module is that metadata:
 *
 * - family-level defaults (most nodes in a family share the same sockets);
 * - per-type overrides for special nodes (trainer, crl_trainer, pca/svd, ...);
 * - declarative input ports from the generated node specs are merged in
 *   automatically (they are the single source for W3-C2 / W5-P0 / W7-E1 ports).
 *
 * The map intentionally under-approximates: a type with no entry exposes no
 * sockets and is classified as not connectable. Extend the tables below with
 * verified handle ids as needed (and keep the classifier tests in sync).
 */
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";

const FAMILY_INPUT_SOCKETS: Readonly<Record<string, readonly string[]>> = {
  trainer_runner: ["dataset", "model", "loss", "optimizer", "observables"],
  optimizer_node: ["lr_schedule", "mup_lr_schedule", "optimizer_lr_schedule"],
  canvas_single_tensor_target: ["tensor"],
  canvas_tensor_multi_input: ["tensor", "tensor_1", "tensor_2"],
  observable_user_tensor_viz_display: ["tensor"],
  atomic_layer_model: ["tensor_in"],
  canvas_layer_strip_chain: ["tensor_in"],
  canvas_full_model: ["tensor_in"],
  dataset_mixer: ["dataset_a", "dataset_b"],
  canvas_comment_source: ["comment"],
};

const FAMILY_OUTPUT_SOCKETS: Readonly<Record<string, readonly string[]>> = {
  trainer_runner: ["observable_results", "loss_results", "checkpoint"],
  canvas_trainer_model_source: ["model", "tensor_out"],
  atomic_layer_model: ["tensor", "tensor_out"],
  canvas_layer_strip_chain: ["tensor_out"],
  trainer_primary_loss: ["loss"],
  trainer_loss_socket_aux: ["loss"],
  optimizer_node: ["optimizer"],
  canvas_dataset_source: ["dataset", "train_dataset", "test_dataset"],
  dataset_mixer: ["dataset"],
  trainer_lr_schedule: ["lr_schedule", "mup_lr_schedule"],
  trainer_batch_schedule: ["batch_schedule"],
  observable: ["observables"],
  observable_user_tensor_viz_display: ["out_tensor", "compare"],
  canvas_tensor_source: ["tensor"],
  canvas_comment_source: ["comment"],
  agent_text_context: ["comment"],
  activation_model: ["model"],
};

const TYPE_INPUT_SOCKETS: Readonly<Record<string, readonly string[]>> = {
  trainer: ["batch_schedule"],
  crl_trainer: ["env"],
  regressor: ["tensor"],
  pca: ["tensor"],
  svd: ["tensor"],
  statistics: ["tensor"],
  effective_rank: ["tensor"],
  series_endpoint_gap: ["tensor"],
  dimension_permutator: ["tensor_in"],
  tensor_slicing: ["tensor"],
  elementwise_transform: ["tensor"],
  smoothing_curve: ["tensor"],
  derivative_curve: ["tensor"],
  moving_statistics: ["tensor"],
  observable_viz: ["tensor"],
  tensor_selector: ["tensor_list"],
  model_checkpoint: ["model_checkpoint"],
};

const TYPE_OUTPUT_SOCKETS: Readonly<Record<string, readonly string[]>> = {
  observable_viz: ["out_tensor", "compare"],
  observable_accuracy: ["out_tensor", "compare"],
  observable_user: ["observable"],
  tensor_selector: ["selected_tensor"],
  dimension_permutator: ["tensor_out"],
  tensor_slicing: ["tensor"],
  elementwise_transform: ["tensor"],
  smoothing_curve: ["tensor"],
  derivative_curve: ["tensor"],
  moving_statistics: ["tensor"],
  effective_rank: ["tensor"],
  series_endpoint_gap: ["tensor"],
  basic_calculator: ["tensor"],
  statistics: ["tensor"],
  statistics2: ["tensor"],
  pca: ["transformed_tensor", "principal_components", "explained_variance_ratio"],
  svd: ["u", "s", "v"],
  activation: ["tensor", "tensor_list"],
  model_checkpoint: ["model_checkpoint"],
  training_visualization: ["out_tensor_list", "stream", "compare"],
  curve_series_viz: ["compare"],
  deterministic_diffusion_sampler: ["samples"],
  crl_env_config: ["env"],
  crl_residual_mlp: ["model"],
  random_input_distribution: ["input_distribution"],
  input_sampler: ["sample_tensor"],
  curve_series_table: ["series"],
};

/**
 * Handle-pair plausibility: which source handles may feed each target handle.
 * This covers the legacy cascade channels. Port-declared targets are merged in
 * from the generated specs below. The classifier uses this to reject pairs that
 * only pass through isValidCanvasConnection's permissive default-true tail
 * (e.g. a dataset "dataset" handle feeding a "tensor_in" strip socket) while
 * keeping the rules function as the final arbiter for type-level checks.
 */
const TARGET_HANDLE_SOURCE_HANDLES: Readonly<Record<string, ReadonlySet<string>>> = {
  tensor_in: new Set([
    "tensor", "tensor_out", "tensor_boundary", "selected_tensor", "transformed_tensor",
    "principal_components", "explained_variance_ratio", "u", "s", "v", "out_tensor",
    "out_tensor_list",
  ]),
  tensor: new Set([
    "tensor", "tensor_out", "tensor_boundary", "selected_tensor", "transformed_tensor",
    "principal_components", "explained_variance_ratio", "u", "s", "v", "out_tensor",
    "out_tensor_list", "stream", "sample_tensor", "loss_results", "observable_results",
    "model", "tensor_list",
  ]),
  tensor_list: new Set([
    "tensor_list", "out_tensor_list", "dataset", "train_dataset", "test_dataset",
    "loss_results", "table",
  ]),
  tensor_1: new Set([
    "tensor", "tensor_out", "tensor_boundary", "selected_tensor", "transformed_tensor",
    "principal_components", "explained_variance_ratio", "u", "s", "v", "out_tensor",
    "out_tensor_list", "stream", "sample_tensor", "tensor_list",
  ]),
  tensor_2: new Set([
    "tensor", "tensor_out", "tensor_boundary", "selected_tensor", "transformed_tensor",
    "principal_components", "explained_variance_ratio", "u", "s", "v", "out_tensor",
    "out_tensor_list", "stream", "sample_tensor", "tensor_list",
  ]),
  tensor_return: new Set(["tensor_out", "tensor"]),
  dataset: new Set(["dataset", "train_dataset", "test_dataset"]),
  train_dataset: new Set(["dataset", "train_dataset", "test_dataset"]),
  test_dataset: new Set(["dataset", "train_dataset", "test_dataset"]),
  dataset_a: new Set(["dataset", "train_dataset", "test_dataset"]),
  dataset_b: new Set(["dataset", "train_dataset", "test_dataset"]),
  model: new Set(["model", "tensor", "tensor_out", "model_checkpoint"]),
  loss: new Set(["loss"]),
  optimizer: new Set(["optimizer"]),
  observables: new Set(["observables", "observable"]),
  observable_in: new Set(["observable", "out_tensor", "tensor"]),
  lr_schedule: new Set(["lr_schedule", "mup_lr_schedule"]),
  mup_lr_schedule: new Set(["mup_lr_schedule"]),
  optimizer_lr_schedule: new Set(["lr_schedule", "mup_lr_schedule"]),
  batch_schedule: new Set(["batch_schedule"]),
  env: new Set(["env"]),
  checkpoint: new Set(["model_checkpoint"]),
  model_checkpoint: new Set(["checkpoint", "model_checkpoint"]),
  initialization: new Set(["initialization", "init"]),
  comment: new Set(["comment"]),
  table: new Set(["table"]),
  coords: new Set(["coords"]),
  pred_coords: new Set(["coords"]),
  true_coords: new Set(["coords"]),
  stream: new Set(["stream", "out_tensor_list", "out_tensor"]),
  curves: new Set(["series", "stream"]),
  compare: new Set(["compare", "out_tensor", "out_tensor_list", "stream"]),
  left: new Set(["compare", "out_tensor", "out_tensor_list", "stream"]),
  right: new Set(["compare", "out_tensor", "out_tensor_list", "stream"]),
  from_viz: new Set(["annotator", "out_tensor", "out_tensor_list", "stream"]),
  distribution: new Set(["input_distribution", "distribution", "sample_tensor"]),
  input_distribution: new Set(["distribution"]),
  train_input: new Set(["sample_tensor"]),
  test_input: new Set(["sample_tensor"]),
  sampler_a: new Set(["samples"]),
  sampler_b: new Set(["samples"]),
  generated: new Set(["samples"]),
};

/** Declared-port targets: allowed source handles unioned from the generated specs. */
const PORT_TARGET_HANDLE_SOURCE_HANDLES: Readonly<Record<string, ReadonlySet<string>>> = (() => {
  const acc: Record<string, Set<string>> = {};
  for (const spec of Object.values(GENERATED_NODE_SPECS)) {
    for (const port of spec?.ports?.in ?? []) {
      const set = (acc[port.id] ??= new Set<string>());
      for (const accept of port.accepts ?? []) {
        for (const h of accept.handles ?? []) set.add(h);
      }
    }
  }
  return acc;
})();

/** True when the source handle class can plausibly feed the target handle class
 * (legacy cascade channels or declarative ports). Unknown target handles are
 * treated as not connectable. */
export function isPlausibleHandlePair(sourceHandle: string, targetHandle: string): boolean {
  const allowed =
    TARGET_HANDLE_SOURCE_HANDLES[targetHandle] ?? PORT_TARGET_HANDLE_SOURCE_HANDLES[targetHandle];
  if (!allowed) return false;
  return allowed.has(sourceHandle);
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

/** Input (target) socket ids a node type exposes: type overrides, then declared
 * ports from the generated spec, then family defaults. */
export function inputSocketsFor(nodeType: string): readonly string[] {
  const spec = GENERATED_NODE_SPECS[nodeType];
  const declared = (spec?.ports?.in ?? []).map((p) => p.id);
  const overrides = TYPE_INPUT_SOCKETS[nodeType] ?? [];
  const fromFamily = (spec?.family ?? []).flatMap((f) => FAMILY_INPUT_SOCKETS[f] ?? []);
  return unique([...overrides, ...declared, ...fromFamily]);
}

/** Output (source) socket ids a node type exposes: type overrides, then family
 * defaults. */
export function outputSocketsFor(nodeType: string): readonly string[] {
  const spec = GENERATED_NODE_SPECS[nodeType];
  const overrides = TYPE_OUTPUT_SOCKETS[nodeType] ?? [];
  const fromFamily = (spec?.family ?? []).flatMap((f) => FAMILY_OUTPUT_SOCKETS[f] ?? []);
  // Built-in observables do not carry a `family` key in the generated spec
  // (their capabilities are injected elsewhere); they still expose the
  // `observables` source socket that feeds the trainer.
  const byCategory = spec?.category === "observables" ? ["observables"] : [];
  return unique([...overrides, ...byCategory, ...fromFamily]);
}
