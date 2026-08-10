import { useReactFlow, useStore, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { serializeGraphForApi } from "../../graph/serializeGraphForApi";
import type { RepresentationEntry } from "../../observables/observableAlgebra";
import { ComfyFloatField, ComfyIntField } from "./comfyNumberFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";

/** Pick a sane default among the model's real representation ids. */
function pickRepresentationDefault(entries: RepresentationEntry[]): string | undefined {
  if (entries.length === 0) return undefined;
  const inputEntries = entries.filter((e) => e.io === "input" && e.representation_id);
  const byFeatures = (e: RepresentationEntry): number => {
    const shape = Array.isArray(e.shape) ? e.shape.slice(1) : [];
    return shape.reduce((acc: number, n: number) => acc * (Number.isFinite(n) ? n : 1), 1);
  };
  if (inputEntries.length > 0) {
    const best = [...inputEntries].sort((a, b) => byFeatures(b) - byFeatures(a))[0];
    if (best?.representation_id) return best.representation_id;
  }
  return entries[0]?.representation_id;
}

/** Renderer for NodeDef-channel observables without a custom TSX (Wave 1 ?6). */
export function GenericObservableNode({ id, data, selected, type }: NodeProps) {
  const spec = GENERATED_NODE_SPECS[String(type)];
  const d = { ...(spec?.defaults ?? {}), ...((data ?? {}) as Record<string, unknown>) };
  const { setNodes } = useReactFlow();
  // Subscribe to the flow store so wiring changes (observable → trainer → model)
  // re-run the representation option fetch. useReactFlow().getNodes/getEdges are
  // stable references, so memo deps on them never update after mount.
  const nodes = useStore(useCallback((state) => state.nodes as Node[], []));
  const edges = useStore(useCallback((state) => state.edges as Edge[], []));
  const patch = (key: string, value: unknown) =>
    setNodes((nds) =>
      nds.map((node) => (node.id === id ? { ...node, data: { ...(node.data as object), [key]: value } } : node)),
    );
  const hasRepresentationField = (spec?.fields ?? []).some((f) => f.key === "representationId");

  // Re-fetch representation options whenever the graph wiring changes.
  const graphSignature = useMemo(() => {
    try {
      return JSON.stringify(serializeGraphForApi(nodes, edges));
    } catch {
      return "";
    }
    // The serialized string is the stable dep for the small observable subgraph;
    // `nodes`/`edges` are the reactive flow-store arrays (ObservablePanel parity).
  }, [nodes, edges]);

  const trainerId = useMemo(() => {
    if (!hasRepresentationField) return "";
    const edge = edges.find(
      (e) =>
        e.source === id &&
        (e.targetHandle === "observables" ||
          e.targetHandle === "observable" ||
          e.targetHandle == null ||
          e.targetHandle === ""),
    );
    return edge?.target ?? "";
  }, [edges, graphSignature, hasRepresentationField, id]);

  const modelId = useMemo(() => {
    if (!trainerId) return "";
    const edge = edges.find(
      (e) =>
        e.target === trainerId &&
        (e.targetHandle === "model" || e.targetHandle == null || e.targetHandle === ""),
    );
    return edge?.source ?? "";
  }, [edges, graphSignature, trainerId]);

  const [repEntries, setRepEntries] = useState<RepresentationEntry[]>([]);
  const [repAutoFixNote, setRepAutoFixNote] = useState<{ missing: string; chosen: string } | null>(null);

  useEffect(() => {
    if (!hasRepresentationField || !modelId) {
      setRepEntries([]);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/model_representation_specs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            nodes,
            edges,
            model_node_id: modelId,
          }),
        });
        if (!res.ok) throw new Error(res.statusText);
        const j = (await res.json()) as { entries?: RepresentationEntry[] };
        if (!cancelled) setRepEntries(j.entries ?? []);
      } catch {
        if (!cancelled) setRepEntries([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [edges, graphSignature, hasRepresentationField, modelId, nodes]);

  const currentRepId = String(d.representationId ?? "");
  const repIds = useMemo(() => repEntries.map((e) => e.representation_id), [repEntries]);

  // Auto-correct a stale default (e.g. "0::output") to a real representation.
  useEffect(() => {
    if (!hasRepresentationField || repEntries.length === 0) return;
    if (currentRepId && repIds.includes(currentRepId)) {
      if (repAutoFixNote !== null) setRepAutoFixNote(null);
      return;
    }
    const suggested = pickRepresentationDefault(repEntries);
    if (!suggested || suggested === currentRepId) return;
    patch("representationId", suggested);
    setRepAutoFixNote({ missing: currentRepId, chosen: suggested });
  }, [currentRepId, hasRepresentationField, patch, repAutoFixNote, repEntries, repIds]);

  return (
    <div
      className={`cr-node cr-node--generic-observable${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-observable)" }}
    >
      <ObservableNodeHeader
        id={id}
        graphNodeType={String(type)}
        title={readInstanceTitle(data, spec?.label ?? String(type))}
      />
      <div className="cr-node__body cr-node__body--compact">
        <ObservableSourceStrip />
        {(spec?.fields ?? []).map((f) => {
          if (f.kind === "int") {
            return (
              <ComfyIntField
                key={f.key}
                label={f.label}
                value={Number(d[f.key] ?? f.defaultValue)}
                min={"min" in f && typeof f.min === "number" ? f.min : undefined}
                max={"max" in f && typeof f.max === "number" ? f.max : undefined}
                onCommit={(n) => patch(f.key, n)}
                ariaLabel={f.label}
              />
            );
          }
          if (f.kind === "float") {
            return (
              <ComfyFloatField
                key={f.key}
                label={f.label}
                value={Number(d[f.key] ?? f.defaultValue)}
                min={"min" in f && typeof f.min === "number" ? f.min : undefined}
                max={"max" in f && typeof f.max === "number" ? f.max : undefined}
                onCommit={(n) => patch(f.key, n)}
                ariaLabel={f.label}
              />
            );
          }
          if (f.kind === "boolean") {
            return (
              <label key={f.key} className="cr-observable-hint nodrag nopan">
                <input
                  type="checkbox"
                  className="nodrag nopan"
                  checked={Boolean(d[f.key] ?? f.defaultValue)}
                  onChange={(e) => patch(f.key, e.target.checked)}
                  aria-label={f.label}
                />{" "}
                {f.label}
              </label>
            );
          }
          if (f.kind === "enum" && f.key === "representationId" && repEntries.length > 0) {
            return (
              <div key={f.key} className="cr-node__field">
                <DiscreteMultiSelect
                  label={f.label}
                  options={repEntries.map((e) => ({ id: e.representation_id, label: e.label ?? e.representation_id }))}
                  value={currentRepId}
                  singleSelect
                  onCommit={(v) => {
                    setRepAutoFixNote(null);
                    patch(f.key, v);
                  }}
                  ariaLabel={f.label}
                />
                {repAutoFixNote !== null ? (
                  <p className="cr-observable-hint">
                    {`representationId ${JSON.stringify(repAutoFixNote.missing)} 不存在，已自动选择 ${JSON.stringify(repAutoFixNote.chosen)}（可在下拉框修改）。`}
                  </p>
                ) : null}
              </div>
            );
          }
          // 无 options 的 enum 不渲染(N3b):defaults 推断出的 enum-无-options 字段
          // (如 embedding_evolution/trajectory 的 label)hand TSX 从不渲染;要渲染的
          // 无-options enum 走 custom adapter(weight_l2 先例)。
          if (f.kind === "enum" && "options" in f && f.options.length) {
            return (
              <DiscreteMultiSelect
                key={f.key}
                label={f.label}
                options={f.options.map((o) => ({ id: o, label: o }))}
                value={String(d[f.key] ?? f.defaultValue)}
                singleSelect
                onCommit={(v) => patch(f.key, v)}
                ariaLabel={f.label}
              />
            );
          }
          return null;
        })}
        {spec?.hint ? <p className="cr-observable-hint">{spec.hint}</p> : null}
      </div>
    </div>
  );
}
