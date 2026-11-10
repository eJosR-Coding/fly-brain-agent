/** Pure view logic for the status block, legend and subtitle. */

export interface Health {
  status: string;
  neurons: number;
  chat: boolean;
  model: string | null;
}

export interface StatusRow {
  key: string;
  value: string;
}

export interface StatusView {
  rows: StatusRow[];
  model: string;
}

export const DATASET = "FlyWire v783";

/** Validate the /health payload; null if it does not match the contract. */
export function parseHealth(d: unknown): Health | null {
  if (typeof d !== "object" || d === null) return null;
  const h = d as Record<string, unknown>;
  if (typeof h.status !== "string" || typeof h.neurons !== "number") return null;
  if (typeof h.chat !== "boolean") return null;
  if (h.model !== null && typeof h.model !== "string") return null;
  return { status: h.status, neurons: h.neurons, chat: h.chat, model: h.model };
}

/**
 * Rows for the status block. The agent (LangGraph) and the in-process MCP server exist only when a chat
 * model is configured, so both follow `health.chat`. `streaming` is true while an SSE turn is running.
 */
export function deriveStatus(health: Health | null, streaming: boolean): StatusView {
  const agent = health === null ? "unknown" : health.chat ? "OK" : "off";
  const model = health === null ? "unknown" : !health.chat ? "none" : (health.model ?? "unknown");
  return {
    rows: [
      { key: "LangGraph", value: agent },
      { key: "MCP server (FastMCP)", value: agent },
      { key: "stream", value: streaming ? "live" : "idle" },
    ],
    model: `model: ${model}`,
  };
}

export function legendView(total: number | null, message?: string): { text: string; active: boolean } {
  if (message) return { text: message, active: false };
  if (total === null) return { text: "no selection", active: false };
  if (total === 0) return { text: "no matches", active: false };
  return { text: `selected · ${total.toLocaleString("en-US")}`, active: true };
}

export function versionLine(count: number | null): string {
  return count === null ? `${DATASET} · loading` : `${DATASET} · ${count.toLocaleString("en-US")} neurons`;
}
