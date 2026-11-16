import { describe, expect, it } from "vitest";
import { shouldOpenOnBoot, shouldOpenOnChange } from "../src/sceneOpen";

describe("shouldOpenOnBoot", () => {
  it("opens the brain when the remembered layout has it large", () => {
    expect(shouldOpenOnBoot({ main: "brain", insetHidden: false })).toBe(true);
    expect(shouldOpenOnBoot({ main: "brain", insetHidden: true })).toBe(true);
  });
  it("leaves the body layout on the preview", () => {
    expect(shouldOpenOnBoot({ main: "body", insetHidden: false })).toBe(false);
  });
});

describe("shouldOpenOnChange", () => {
  it("opens when the brain just became the main pane", () => {
    expect(shouldOpenOnChange("body", { main: "brain", insetHidden: false })).toBe(true);
  });
  it("does not retry on hide/show of the inset while the brain stays main", () => {
    expect(shouldOpenOnChange("brain", { main: "brain", insetHidden: true })).toBe(false);
    expect(shouldOpenOnChange("brain", { main: "brain", insetHidden: false })).toBe(false);
  });
  it("does nothing when the body becomes main or stays main", () => {
    expect(shouldOpenOnChange("brain", { main: "body", insetHidden: false })).toBe(false);
    expect(shouldOpenOnChange("body", { main: "body", insetHidden: true })).toBe(false);
  });
});
