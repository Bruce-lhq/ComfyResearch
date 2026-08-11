import type { MovingStatisticOp } from "./movingStatisticsDefaults";

/**
 * Trailing rolling statistic: output point i aggregates the samples in
 * [max(0, i - window + 1), i]. `stride` hops the anchor; `minPeriods` drops
 * head windows that do not have enough samples. std uses the population
 * formula (divide by N), so a single-sample window yields 0 instead of NaN.
 */
export function computeMovingStatistic(
  values: number[],
  statistic: MovingStatisticOp,
  window: number,
  stride: number,
  minPeriods: number,
): number[] {
  const w = Math.max(1, Math.floor(window));
  const out: number[] = [];
  for (const i of movingStatisticAnchors(values.length, window, stride, minPeriods)) {
    const start = Math.max(0, i - w + 1);
    const count = i - start + 1;
    let sum = 0;
    for (let j = start; j <= i; j += 1) sum += values[j]!;
    if (statistic === "mean") {
      out.push(sum / count);
      continue;
    }
    const mean = sum / count;
    let acc = 0;
    for (let j = start; j <= i; j += 1) {
      const diff = values[j]! - mean;
      acc += diff * diff;
    }
    out.push(Math.sqrt(acc / count));
  }
  return out;
}

/** Input indices kept as output anchors for a trailing rolling window. */
export function movingStatisticAnchors(
  n: number,
  window: number,
  stride: number,
  minPeriods: number,
): number[] {
  const w = Math.max(1, Math.floor(window));
  const s = Math.max(1, Math.floor(stride));
  const mp = Math.max(1, Math.min(w, Math.floor(minPeriods)));
  const out: number[] = [];
  for (let i = 0; i < n; i += s) {
    const start = Math.max(0, i - w + 1);
    const count = i - start + 1;
    if (count < mp) continue;
    out.push(i);
  }
  return out;
}

/** Output x coordinates: map each kept anchor through the aligned upstream ticks. */
export function movingStatisticsOutputTicks(
  stepTicks: number[] | null | undefined,
  inputLength: number,
  window: number,
  stride: number,
  minPeriods: number,
): number[] | undefined {
  if (!stepTicks || stepTicks.length !== inputLength) return undefined;
  const ticks = movingStatisticAnchors(inputLength, window, stride, minPeriods).map((i) => stepTicks[i]!);
  return ticks.every((t) => Number.isFinite(t)) ? ticks : undefined;
}
