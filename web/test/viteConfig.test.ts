import { describe, expect, it } from "vitest";
import config from "../vite.config";

describe("vite proxy", () => {
  it.each(["server", "preview"] as const)("forwards /api to the API on %s, WebSocket upgrades included (the live session is /api/ws/live)", (where) => {
    const proxy = config[where]?.proxy as Record<string, { target: string; ws?: boolean; rewrite?: (p: string) => string }>;
    expect(proxy["/api"].target).toBe("http://127.0.0.1:8000");
    expect(proxy["/api"].ws).toBe(true);
    expect(proxy["/api"].rewrite?.("/api/ws/live")).toBe("/ws/live");
  });
});
