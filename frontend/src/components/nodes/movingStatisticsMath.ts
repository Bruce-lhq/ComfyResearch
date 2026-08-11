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
  const s = Math.max(1, Math.floor(stride));
  const mp = Math.max(1, Math.min(w, Math.floor(minPeriods)));
  const out: number[] = [];
  for (let i = 0; i < values.length; i += s) {
    const start = Math.max(0, i - w + 1);
    const count = i - start + 1;
    if (count < mp) continue;
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
