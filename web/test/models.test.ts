import { describe, expect, it } from "vitest";
import { STORAGE_KEY, loadStoredModel, optionLabel, parseModels, pickModel, saveStoredModel } from "../src/models";

const LIST = {
  default: "google_genai:g",
  models: [
    { id: "google_genai:g", provider: "google_genai", available: true },
    { id: "groq:fast", provider: "groq", available: true },
    { id: "anthropic:c", provider: "anthropic", available: false, reason: "ANTHROPIC_API_KEY is not set" },
  ],
};

describe("parseModels", () => {
  it("accepts the /models contract", () => {
    expect(parseModels(LIST)).toEqual(LIST);
    expect(parseModels({ default: null, models: [] })).toEqual({ default: null, models: [] });
  });
  it("rejects anything else", () => {
    expect(parseModels(null)).toBeNull();
    expect(parseModels({ default: 3, models: [] })).toBeNull();
    expect(parseModels({ default: null, models: [{ id: "x" }] })).toBeNull();
    expect(parseModels({ default: null, models: "no" })).toBeNull();
  });
});

describe("pickModel", () => {
  it("keeps the stored id while it is still available", () => {
    expect(pickModel(LIST, "groq:fast")).toBe("groq:fast");
  });
  it("falls back to the default when the stored id is gone, unavailable or absent", () => {
    expect(pickModel(LIST, "groq:removed")).toBe("google_genai:g");
    expect(pickModel(LIST, "anthropic:c")).toBe("google_genai:g");
    expect(pickModel(LIST, null)).toBe("google_genai:g");
  });
  it("uses the first available model if the default is unavailable, null if none", () => {
    const l = { default: "anthropic:c", models: LIST.models };
    expect(pickModel(l, null)).toBe("google_genai:g");
    expect(pickModel({ default: null, models: [] }, "x")).toBeNull();
  });
});

describe("optionLabel", () => {
  it("shows the id, and marks models without a key", () => {
    expect(optionLabel(LIST.models[1])).toBe("groq:fast");
    expect(optionLabel(LIST.models[2])).toBe("anthropic:c (no key)");
  });
});

describe("stored choice", () => {
  const mem = () => {
    const m = new Map<string, string>();
    return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) } as Storage;
  };
  it("round-trips through storage", () => {
    const s = mem();
    saveStoredModel("groq:fast", s);
    expect(s.getItem(STORAGE_KEY)).toBe("groq:fast");
    expect(loadStoredModel(s)).toBe("groq:fast");
  });
  it("never throws when storage is blocked", () => {
    const bad = {
      getItem: () => { throw new Error("blocked"); },
      setItem: () => { throw new Error("blocked"); },
    } as unknown as Storage;
    expect(() => saveStoredModel("x", bad)).not.toThrow();
    expect(loadStoredModel(bad)).toBeNull();
  });
});
