import { describe, expect, it } from "vitest";
import { headAt, type WalkClip } from "../src/bodyClip";

const clip = (head?: number[]): WalkClip => ({ nFrames: 3, sampleRateHz: 200, head: head ? Float64Array.from(head) : undefined } as unknown as WalkClip);

describe("headAt", () => {
  it("reads the head yaw of the frame at t", () => {
    expect(headAt(clip([0, 5, 9]), 0.005)).toBe(5);
  });
  it("is null without a head track or with a track of the wrong length", () => {
    expect(headAt(clip(), 0)).toBeNull();
    expect(headAt(clip([1, 2]), 0)).toBeNull();
  });
});
