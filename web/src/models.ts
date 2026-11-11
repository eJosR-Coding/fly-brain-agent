/** Pure logic for the model picker: the /models contract, the choice rule and its persistence. */

export interface ModelInfo {
  id: string;
  provider: string;
  available: boolean;
  reason?: string;
}

export interface ModelList {
  default: string | null;
  models: ModelInfo[];
}

export const STORAGE_KEY = "flybrain.model";

/** Validate the /models payload; null if it does not match the contract. */
export function parseModels(d: unknown): ModelList | null {
  if (typeof d !== "object" || d === null) return null;
  const o = d as Record<string, unknown>;
  if (o.default !== null && typeof o.default !== "string") return null;
  if (!Array.isArray(o.models)) return null;
  const models: ModelInfo[] = [];
  for (const m of o.models) {
    if (typeof m !== "object" || m === null) return null;
    const r = m as Record<string, unknown>;
    if (typeof r.id !== "string" || typeof r.provider !== "string" || typeof r.available !== "boolean") return null;
    models.push({
      id: r.id,
      provider: r.provider,
      available: r.available,
      ...(typeof r.reason === "string" ? { reason: r.reason } : {}),
    });
  }
  return { default: o.default as string | null, models };
}

/** The stored id while it is still available, else the server default, else the first available model. */
export function pickModel(list: ModelList, stored: string | null): string | null {
  const ok = (id: string | null) => id !== null && list.models.some((m) => m.id === id && m.available);
  if (ok(stored)) return stored;
  if (ok(list.default)) return list.default;
  return list.models.find((m) => m.available)?.id ?? null;
}

export function optionLabel(m: ModelInfo): string {
  return m.available ? m.id : `${m.id} (no key)`;
}

export function loadStoredModel(storage: Storage = localStorage): string | null {
  try {
    return storage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveStoredModel(id: string, storage: Storage = localStorage): void {
  try {
    storage.setItem(STORAGE_KEY, id);
  } catch {
    /* private mode or blocked storage: the choice just is not remembered */
  }
}
