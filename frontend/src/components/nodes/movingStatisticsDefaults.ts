import type { CollectedActivationTensor } from "./activationDefaults";

export type MovingStatisticOp = "mean" | "std";

export type MovingStatisticsNodeData = {
  statistic: MovingStatisticOp;
  window: number;
  stride: number;
  minPeriods: number;
  logScaleX: boolean;
  logScaleY: boolean;
  outputTensor: CollectedActivationTensor | null;
  lastError: string | null;
};

function clampInt(raw: unknown, fallback: number, min: number, max: number): number {
  const n = Math.floor(Number(raw));
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

export function defaultMovingStatisticsData(
  partial?: Partial<MovingStatisticsNodeData>,
): MovingStatisticsNodeData {
  const window = clampInt(partial?.window, 10, 1, 1024);
  return {
    statistic: partial?.statistic === "std" ? "std" : "mean",
    window,
    stride: clampInt(partial?.stride, 1, 1, 1024),
    minPeriods: clampInt(partial?.minPeriods, 1, 1, window),
    logScaleX: partial?.logScaleX ?? false,
    logScaleY: partial?.logScaleY ?? false,
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}

/** Number of output points for a trailing rolling window anchored at each output index. */
export function movingStatisticsOutputLength(
  n: number,
  window: number,
  stride: number,
  minPeriods: number,
): number {
  if (!(n > 0) || !(window >= 1) || !(stride >= 1) || !(minPeriods >= 1)) return 0;
  let count = 0;
  for (let i = 0; i < n; i += stride) {
    const available = Math.min(i + 1, window);
    if (available >= minPeriods) count += 1;
  }
  return count;
}
