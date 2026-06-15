import { describe, expect, it } from "vitest";
import { ApiError, apiErrorMessage, shortPath } from "./api.js";
import { copy } from "./config/appConfig.jsx";

describe("apiErrorMessage", () => {
  it("prefers localized stable error codes over generic messages", () => {
    const error = new ApiError("No compatible load method found for this model format: nvfp4", "MODEL_BRANCH_UNSUPPORTED", 400);

    expect(apiErrorMessage(error, copy.zh)).toBe("当前模型格式没有兼容的微调加载方式。请确认模型是 Transformers 兼容格式，并在模型管理查看模型格式，在系统环境查看训练支持范围。");
    expect(apiErrorMessage(error, copy.en)).toBe("No compatible fine-tuning load method is configured for this model format. Confirm the model is Transformers-compatible, review the model format in Model Management, and check training support in System Environment.");
  });

  it("falls back to backend detail when no translation exists", () => {
    const error = new ApiError("Backend detail", "SOME_NEW_CODE", 400);

    expect(apiErrorMessage(error, copy.en)).toBe("Backend detail");
  });
});

describe("shortPath", () => {
  it("keeps compact paths readable", () => {
    expect(shortPath("outputs/final")).toBe("outputs/final");
  });

  it("truncates long paths from the left", () => {
    const path = "E:/very/long/path/that/keeps/going/for/a/model/output/final_adapter";

    expect(shortPath(path)).toMatch(/^\.\.\./);
    expect(shortPath(path)).toContain("final_adapter");
  });
});
