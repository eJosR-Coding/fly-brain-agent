import { describe, expect, it } from "vitest";
import { deriveStatus, legendView, parseHealth, versionLine } from "../src/status";

const ON = { status: "ok", neurons: 139248, chat: true, model: "google_genai:gemini-3.5-flash" };
const OFF = { status: "ok", neurons: 139248, chat: false, model: null };

describe("parseHealth", () => {
  it("accepts the /health contract", () => {
    expect(parseHealth(ON)).toEqual(ON);
  });
  it("passes the simulation flag through when the backend sends it", () => {
    expect(parseHealth({ ...ON, simulation: true })).toEqual({ ...ON, simulation: true });
    expect(parseHealth({ ...ON, simulation: false })?.simulation).toBe(false);
    expect(parseHealth({ ...ON, simulation: "yes" })).toBeNull();
  });
  it("rejects anything else", () => {
    expect(parseHealth(null)).toBeNull();
    expect(parseHealth({ status: "ok", neurons: 3 })).toBeNull();
    expect(parseHealth({ ...ON, chat: "yes" })).toBeNull();
    expect(parseHealth({ ...ON, model: 7 })).toBeNull();
  });
});

describe("deriveStatus", () => {
  it("reports a configured agent and an idle stream", () => {
    const v = deriveStatus(ON, false);
    expect(v.rows).toEqual([
      { key: "LangGraph", value: "OK" },
      { key: "MCP server (FastMCP)", value: "OK" },
      { key: "stream", value: "idle" },
    ]);
    expect(v.model).toBe("model: google_genai:gemini-3.5-flash");
  });
  it("shows the stream as live while a turn is running", () => {
    expect(deriveStatus(ON, true).rows[2]).toEqual({ key: "stream", value: "live" });
  });
  it("is honest when chat is not configured", () => {
    const v = deriveStatus(OFF, false);
    expect(v.rows[0]).toEqual({ key: "LangGraph", value: "off" });
    expect(v.rows[1]).toEqual({ key: "MCP server (FastMCP)", value: "off" });
    expect(v.model).toBe("model: none");
  });
  it("says unknown when the backend did not answer", () => {
    const v = deriveStatus(null, false);
    expect(v.rows[0].value).toBe("unknown");
    expect(v.rows[1].value).toBe("unknown");
    expect(v.model).toBe("model: unknown");
  });
  it("chat on without a model name", () => {
    expect(deriveStatus({ ...ON, model: null }, false).model).toBe("model: unknown");
  });
});

describe("legendView", () => {
  it("no selection", () => {
    expect(legendView(null)).toEqual({ text: "no selection", active: false });
  });
  it("a selection with a thousands separator", () => {
    expect(legendView(2926)).toEqual({ text: "selected · 2,926", active: true });
  });
  it("zero matches is not an active selection", () => {
    expect(legendView(0)).toEqual({ text: "no matches", active: false });
  });
  it("a message wins over the count", () => {
    expect(legendView(null, "search failed: x")).toEqual({ text: "search failed: x", active: false });
  });
});

describe("versionLine", () => {
  it("shows the dataset and the neuron count", () => {
    expect(versionLine(139248)).toBe("FlyWire v783 · 139,248 neurons");
  });
  it("before the count is known", () => {
    expect(versionLine(null)).toBe("FlyWire v783 · loading");
  });
});


describe("deriveStatus with a picked model", () => {
  it("shows the selected model instead of the server default", () => {
    expect(deriveStatus(ON, false, "groq:fast").model).toBe("model: groq:fast");
  });
  it("is unchanged without a selection or without chat", () => {
    expect(deriveStatus(ON, false, null).model).toBe("model: google_genai:gemini-3.5-flash");
    expect(deriveStatus(OFF, false, "groq:fast").model).toBe("model: none");
  });
});
