// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { addGroup, addSubgroup, renderKey } from "../src/controls";

describe("addGroup", () => {
  it("makes a labelled role=group and returns the container for its chips", () => {
    const host = document.createElement("div");
    const items = addGroup(host, "run", "RUN");
    const g = host.querySelector<HTMLElement>('[role="group"]')!;
    expect(g.getAttribute("aria-label")).toBe("RUN");
    expect(g.dataset.group).toBe("run");
    expect(g.querySelector(".ctl-label")?.textContent).toBe("RUN");
    expect(g.contains(items)).toBe(true);
    const b = document.createElement("button");
    items.appendChild(b);
    expect(g.querySelectorAll("button")).toHaveLength(1);
  });

  it("nests a tight sub-group inside a group", () => {
    const host = document.createElement("div");
    const view = addGroup(host, "view", "VIEW");
    const cam = addSubgroup(view, "camera");
    expect(view.querySelector('[role="group"][aria-label="camera"]')).toBe(cam);
  });
});

describe("renderKey", () => {
  it("renders a title row then one row per category with its count", () => {
    const el = document.createElement("div");
    renderKey(el, "REGIONS", [{ name: "central", count: 32383, hex: "#fff" }]);
    expect(el.querySelector(".ctl-label")?.textContent).toBe("REGIONS");
    const row = el.querySelector(".key-row")!;
    expect(row.querySelector("span")?.textContent).toBe("central");
    expect(row.querySelector("b")?.textContent).toBe("32,383");
  });
});
