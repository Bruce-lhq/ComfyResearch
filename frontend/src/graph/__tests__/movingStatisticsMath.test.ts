import { describe, expect, it } from "vitest";
import {
  computeMovingStatistic,
  movingStatisticAnchors,
  movingStatisticsOutputTicks,
} from "../../components/nodes/movingStatisticsMath";
import { movingStatisticsOutputLength } from "../../components/nodes/movingStatisticsDefaults";

describe("computeMovingStatistic", () => {
  it("computes a trailing moving mean with min periods 1", () => {
    expect(computeMovingStatistic([1, 2, 3, 4, 5], "mean", 3, 1, 1)).toEqual([1, 1.5, 2, 3, 4]);
  });

  it("hops the anchor by stride", () => {
    expect(computeMovingStatistic([1, 2, 3, 4, 5], "mean", 3, 2, 1)).toEqual([1, 2, 4]);
  });

  it("drops head windows that do not reach min periods", () => {
    expect(computeMovingStatistic([1, 2, 3, 4, 5], "mean", 3, 1, 3)).toEqual([2, 3, 4]);
  });

  it("computes population standard deviation per window", () => {
    expect(computeMovingStatistic([1, 3, 1, 3], "std", 2, 1, 1)).toEqual([0, 1, 1, 1]);
  });
});

describe("movingStatisticsOutputLength", () => {
  it("keeps the input length at stride 1 with min periods 1", () => {
    expect(movingStatisticsOutputLength(10, 3, 1, 1)).toBe(10);
  });

  it("shrinks when full windows are required", () => {
    expect(movingStatisticsOutputLength(10, 3, 1, 3)).toBe(8);
  });

  it("downsamples with stride", () => {
    expect(movingStatisticsOutputLength(10, 3, 2, 1)).toBe(5);
  });

  it("counts anchors whose available samples meet min periods", () => {
    expect(movingStatisticsOutputLength(10, 4, 3, 4)).toBe(3);
  });
});

describe("movingStatisticAnchors", () => {
  it("keeps every anchor at stride 1 with min periods 1", () => {
    expect(movingStatisticAnchors(5, 3, 1, 1)).toEqual([0, 1, 2, 3, 4]);
  });

  it("hops by stride and drops head anchors below min periods", () => {
    expect(movingStatisticAnchors(5, 3, 2, 2)).toEqual([2, 4]);
  });

  it("matches movingStatisticsOutputLength", () => {
    const cases: [number, number, number, number][] = [
      [10, 3, 1, 1],
      [10, 3, 1, 3],
      [10, 3, 2, 1],
      [10, 4, 3, 4],
    ];
    for (const [n, w, s, mp] of cases) {
      expect(movingStatisticAnchors(n, w, s, mp).length).toBe(movingStatisticsOutputLength(n, w, s, mp));
    }
  });
});

describe("movingStatisticsOutputTicks", () => {
  it("maps non-uniform upstream ticks through the kept anchors", () => {
    expect(movingStatisticsOutputTicks([10, 20, 40, 80, 160], 5, 3, 2, 2)).toEqual([40, 160]);
  });

  it("returns undefined when ticks are missing or misaligned", () => {
    expect(movingStatisticsOutputTicks([10, 20, 40], 5, 3, 1, 1)).toBeUndefined();
    expect(movingStatisticsOutputTicks(null, 5, 3, 1, 1)).toBeUndefined();
  });
});
