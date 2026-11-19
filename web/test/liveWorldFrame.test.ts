import { describe, expect, it } from "vitest";
import { worldPose } from "../src/bodyClip";
import { objectToViewer, parseLiveChunk, parseLiveHeader } from "../src/liveFrames";
import { chunk, header } from "./liveFixtures";

const apply = (m: number[], p: number[]) => [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);

describe("one frame for the fly, the cube and the objects", () => {
  const h = parseLiveHeader({ ...header(), thorax_origin_um: [496, 0, 1300] });
  it("a cube 3 mm ahead of the thorax is drawn 3 mm ahead of the drawn thorax", () => {
    const { chunk: c } = parseLiveChunk(chunk(2, { thorax: { pos_um: [[0, 0, 1022], [0, 0, 1022]], yaw_deg: [0, 0], quat: [[1, 0, 0, 0], [1, 0, 0, 0]] } }), h, null);
    const wp = worldPose(c.clip, 0)!;
    const thorax = apply(wp.matrix, c.clip.thoraxOrigin); // viewer [X = y, Y, Z = x]
    const cube = objectToViewer([3, 0, 0.5]);
    expect(thorax[0]).toBeCloseTo(0, 6);
    expect(thorax[2]).toBeCloseTo(0, 6);
    expect(cube[2] - thorax[2]).toBeCloseTo(3, 6);
    expect(cube[0] - thorax[0]).toBeCloseTo(0, 6);
  });
  it("the full 3D orientation is applied (ceiling: dorsal axis points down)", () => {
    const { chunk: c } = parseLiveChunk(chunk(2, { thorax: { pos_um: [[0, 0, 1022], [0, 0, 49000]], yaw_deg: [0, 0], quat: [[0, 1, 0, 0], [0, 1, 0, 0]] } }), h, null);
    const wp = worldPose(c.clip, 0)!;
    expect(wp.matrix[5]).toBeCloseTo(-1, 6); // viewer Y -> -Y
  });
});
