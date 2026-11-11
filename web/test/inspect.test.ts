import { describe, expect, it } from "vitest";
import { cardView, isClick, parseNeuronCard, pickNearest, pickThreshold, placeCard } from "../src/inspect";

const raw = {
  row: 60, root_id: "720575940624566549", cell_type: "PAM05", cell_class: "DAN", super_class: "central",
  side: "left", top_nt: "dopamine", position_um: [301.25, 120.5, 88], region: "central", partners: true,
  inputs: [{ row: 5, cell_type: "KCg-m", super_class: "central", synapses: 31 }],
  outputs: [{ row: 9, cell_type: null, super_class: "central", synapses: 12 }],
  synapses_in: 120, synapses_out: 1234,
};

describe("pickNearest", () => {
  it("takes the hit closest to the ray", () => {
    expect(pickNearest([
      { index: 4, distanceToRay: 0.9, distance: 10 },
      { index: 7, distanceToRay: 0.2, distance: 30 },
    ])).toBe(7);
  });
  it("breaks ties by camera distance, then by index", () => {
    expect(pickNearest([
      { index: 4, distanceToRay: 0.5, distance: 30 },
      { index: 7, distanceToRay: 0.5, distance: 10 },
    ])).toBe(7);
    expect(pickNearest([
      { index: 9, distanceToRay: 0.5, distance: 10 },
      { index: 3, distanceToRay: 0.5, distance: 10 },
    ])).toBe(3);
  });
  it("no hits, no pick", () => expect(pickNearest([])).toBeNull());
});

describe("isClick", () => {
  it("a tiny wobble is a click, a real drag is not", () => {
    expect(isClick({ x: 100, y: 100 }, { x: 102, y: 99 })).toBe(true);
    expect(isClick({ x: 100, y: 100 }, { x: 110, y: 100 })).toBe(false);
    expect(isClick({ x: 100, y: 100 }, { x: 103, y: 104 }, 6)).toBe(true);
  });
});

describe("pickThreshold", () => {
  it("grows with camera distance and shrinks with canvas height", () => {
    const near = pickThreshold(8, 45, 100, 900);
    expect(pickThreshold(8, 45, 200, 900)).toBeCloseTo(near * 2);
    expect(pickThreshold(8, 45, 100, 1800)).toBeCloseTo(near / 2);
  });
  it("is the world size of `px` pixels at that distance", () => {
    // 900 px tall at fov 90 and distance 450: the view is 900 world units tall, so 1 px = 1 unit
    expect(pickThreshold(8, 90, 450, 900)).toBeCloseTo(8);
  });
});

describe("parseNeuronCard", () => {
  it("accepts the endpoint payload", () => {
    const c = parseNeuronCard(raw, 139255);
    expect(typeof c).toBe("object");
    expect((c as { rootId: string }).rootId).toBe("720575940624566549");
  });
  it("rejects bad shapes and out-of-range rows", () => {
    expect(parseNeuronCard(null, null)).toMatch(/object/);
    expect(parseNeuronCard({ ...raw, row: -1 }, null)).toMatch(/row/);
    expect(parseNeuronCard({ ...raw, row: 10 }, 10)).toMatch(/row/);
    expect(parseNeuronCard({ ...raw, position_um: [1, 2] }, null)).toMatch(/position/);
    expect(parseNeuronCard({ ...raw, root_id: 5 }, null)).toMatch(/root_id/);
  });
  it("tolerates missing partners (no connectome) and drops malformed partner rows", () => {
    const c = parseNeuronCard({ ...raw, partners: false, inputs: undefined, outputs: undefined }, null);
    expect((c as { partners: unknown }).partners).toBeNull();
    const d = parseNeuronCard({ ...raw, inputs: [{ row: "x" }, raw.inputs[0]] }, null);
    expect((d as { partners: { inputs: unknown[] } }).partners.inputs).toHaveLength(1);
  });
});

describe("cardView", () => {
  const card = parseNeuronCard(raw, null);
  if (typeof card === "string") throw new Error(card);
  it("titles the neuron as type, region class, side, transmitter", () => {
    expect(cardView(card).title).toBe("PAM05 · central · left · dopamine");
  });
  it("skips missing parts and names unannotated neurons", () => {
    const bare = { ...card, cellType: null, topNt: null };
    expect(cardView(bare).title).toBe("unannotated · central · left");
  });
  it("formats the detail rows", () => {
    const rows = Object.fromEntries(cardView(card).rows.map((r) => [r.label, r.value]));
    expect(rows["root id"]).toBe("720575940624566549");
    expect(rows["class"]).toBe("DAN");
    expect(rows["position"]).toBe("301.3, 120.5, 88.0 µm");
    expect(rows["synapses"]).toBe("120 in · 1,234 out");
  });
  it("lists partners with synapse counts", () => {
    const v = cardView(card);
    expect(v.inputs[0]).toEqual({ row: 5, text: "KCg-m · central", syn: "31 syn" });
    expect(v.outputs[0].text).toBe("unannotated · central");
    expect(v.partnersNote).toBeNull();
  });
  it("says so when partners are unavailable", () => {
    const v = cardView({ ...card, partners: null });
    expect(v.inputs).toEqual([]);
    expect(v.partnersNote).toMatch(/connectome/);
  });
});

describe("placeCard", () => {
  const bounds = { w: 1000, h: 800 };
  const size = { w: 280, h: 300 };
  it("sits to the right of the anchor when there is room", () => {
    expect(placeCard({ x: 200, y: 300 }, size, bounds, { gap: 16, margin: 12 })).toEqual({ left: 216, top: 150 });
  });
  it("flips to the left near the right edge", () => {
    expect(placeCard({ x: 900, y: 300 }, size, bounds, { gap: 16, margin: 12 }).left).toBe(900 - 16 - 280);
  });
  it("stays inside the bounds vertically", () => {
    expect(placeCard({ x: 200, y: 10 }, size, bounds, { gap: 16, margin: 12 }).top).toBe(12);
    expect(placeCard({ x: 200, y: 790 }, size, bounds, { gap: 16, margin: 12 }).top).toBe(800 - 12 - 300);
  });
});
