/**
 * Sidebar connectivity classification.
 *
 * For every catalog node type we answer one question: given the nodes already on
 * the canvas, can this type form at least one real canvas connection with one of
 * them (either direction)?
 *
 * A pair only counts when the source handle is a socket the source type actually
 * exposes and the target handle is a socket the target type actually exposes
 * (nodeSockets). isValidCanvasConnection then arbitrates the pair — this keeps the
 * rules function as the single source of truth while avoiding its permissive
 * default-true tail. ioMode variants are enumerated for io-sensitive candidates,
 * mirroring the connection golden matrix semantics.
 *
 * KNOWN_UNCONNECTABLE_PAIRS / KNOWN_NOT_RUNNABLE_PAIRS are the curation hooks for
 * the two known gaps:
 *   1. rules say connectable but the real flow is broken;
 *   2. connected graphs still fail at training time because two node types are
 *      not runtime-compatible.
 * Populate them only with verified evidence (e.g. a failed headless /api/train on
 * an otherwise valid graph). Both maps are empty by default.
 */
import type { Connection, Node } from "@xyflow/react";

import { isValidCanvasConnection } from "./connectionRules";
import { inputSocketsFor, isPlausibleHandlePair, outputSocketsFor } from "./nodeSockets";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import { nodeRegistryDefaults } from "./nodeRegistrySpec";

const IO_MODE_VALUES = ["model", "input-output"] as const;

/** ioMode-sensitive families mirroring scripts/generate-connection-golden.ts. */
const IO_SENSITIVE_PREFIX =
  /(_model|_layer|combined_model|reshape|flatten|softmax|causal_mask|tensor_splitter|einsum|elementwise_transform|tensor_slicing)$/;

export function isCanvasIoModeSensitive(nodeType: string): boolean {
  return IO_SENSITIVE_PREFIX.test(nodeType);
}

/**
 * Known pairs where the connection rules say "connectable" but the real flow is
 * broken (problem 1 from the user). candidateType -> set of existing types.
 */
export const KNOWN_UNCONNECTABLE_PAIRS: ReadonlyMap<string, ReadonlySet<string>> = new Map();

/**
 * Known pairs that connect fine but fail at training time (problem 2). Populate
 * after verification with headless /api/train runs.
 */
export const KNOWN_NOT_RUNNABLE_PAIRS: ReadonlyMap<string, ReadonlySet<string>> = new Map();

const pairCache = new Map<string, boolean>();

function typePairKey(candidateType: string, existingType: string, existingIoMode: string): string {
  return `${candidateType} > ${existingType} [${existingIoMode}]`;
}

function isBlockedPair(candidateType: string, existingType: string): boolean {
  const unconnectable = KNOWN_UNCONNECTABLE_PAIRS.get(candidateType);
  if (unconnectable?.has(existingType)) return true;
  const notRunnable = KNOWN_NOT_RUNNABLE_PAIRS.get(candidateType);
  return notRunnable?.has(existingType) ?? false;
}

function dummyCandidateNode(nodeType: string, ioMode?: string): Node {
  const data: Record<string, unknown> = { ...(nodeRegistryDefaults(nodeType) ?? {}) };
  if (ioMode != null) data.ioMode = ioMode;
  return { id: "__candidate", type: nodeType, position: { x: 0, y: 0 }, data } as unknown as Node;
}

/** Whether a single candidate type can form a valid canvas connection with one
 * existing canvas node, in either direction, under any of its ioMode variants. */
function canTypePairConnect(candidateType: string, existing: Node): boolean {
  const existingType = String(existing.type ?? "");
  const existingIoMode = readNodeCanvasIoMode(
    (existing.data ?? {}) as Record<string, unknown> | undefined,
  );
  const key = typePairKey(candidateType, existingType, existingIoMode);
  const cached = pairCache.get(key);
  if (cached !== undefined) return cached;

  const candidateModes: ReadonlyArray<string | undefined> = isCanvasIoModeSensitive(candidateType)
    ? IO_MODE_VALUES
    : [undefined];

  let connectable = false;
  for (const candMode of candidateModes) {
    const candidate = dummyCandidateNode(candidateType, candMode);
    const nodes = [existing, candidate] as unknown as Node[];
    const candidateInputs = inputSocketsFor(candidateType);
    const candidateOutputs = outputSocketsFor(candidateType);
    const existingInputs = inputSocketsFor(existingType);
    const existingOutputs = outputSocketsFor(existingType);

    // candidate -> existing
    for (const sh of candidateOutputs) {
      for (const th of existingInputs) {
        const probe = {
          source: candidate.id,
          target: existing.id,
          sourceHandle: sh,
          targetHandle: th,
        } as Connection;
        if (isPlausibleHandlePair(sh, th) && isValidCanvasConnection(probe, nodes, [])) {
          connectable = true;
          break;
        }
      }
      if (connectable) break;
    }
    if (connectable) break;

    // existing -> candidate
    for (const sh of existingOutputs) {
      for (const th of candidateInputs) {
        const probe = {
          source: existing.id,
          target: candidate.id,
          sourceHandle: sh,
          targetHandle: th,
        } as Connection;
        if (isPlausibleHandlePair(sh, th) && isValidCanvasConnection(probe, nodes, [])) {
          connectable = true;
          break;
        }
      }
      if (connectable) break;
    }
    if (connectable) break;
  }

  const result = connectable && !isBlockedPair(candidateType, existingType);
  pairCache.set(key, result);
  return result;
}

/** True when the candidate type can connect to at least one node currently on
 * the canvas. An empty canvas therefore classifies every type as not
 * connectable; the UI treats that state as "every type can start a graph". */
export function canNodeTypeConnectToAny(candidateType: string, canvasNodes: Node[]): boolean {
  for (const existing of canvasNodes) {
    if (canTypePairConnect(candidateType, existing)) return true;
  }
  return false;
}
