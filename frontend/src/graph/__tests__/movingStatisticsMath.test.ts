import { describe, expect, it } from "vitest";
import { computeMovingStatistic } from "../../components/nodes/movingStatisticsMath";
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
