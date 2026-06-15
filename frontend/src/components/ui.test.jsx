import { describe, expect, it } from "vitest";
import { isPathLikeValue } from "./ui.jsx";

describe("isPathLikeValue", () => {
  it("detects long local paths", () => {
    expect(isPathLikeValue("E:\\myproject\\LocalTune\\models\\Qwen\\Qwen3___6-27B")).toBe(true);
    expect(isPathLikeValue("./outputs/localtune/bnb4/final")).toBe(true);
  });

  it("does not treat short labels or versions as paths", () => {
    expect(isPathLikeValue("0.4.0")).toBe(false);
    expect(isPathLikeValue("configs")).toBe(false);
  });
});
