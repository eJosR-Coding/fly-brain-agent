/** Fixtures of the live-session tests: the server's header and chunk messages (see flybrain.live). */
export const Q = 10000;
/** 2 bodies; neutral at rest; thorax origin at the mesh origin */
export const header = () => ({
  format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa", "lf_tibia"], legs: ["LF"],
  sample_rate_hz: 200, neutral: { pos_um: [[0, 0, 0], [0, 0, 0]], quat: [[Q, 0, 0, 0], [Q, 0, 0, 0]] },
});
/** n frames; body 0 slides +100 um per frame along fly x (viewer Z); thorax walks 1000 um per frame along x */
export function chunk(n = 10, extra: Record<string, unknown> = {}) {
  const frames = Array.from({ length: n }, (_, i) => [[i * 100, 0, 0, Q, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]]);
  return {
    type: "chunk", seq: 0, t_ms: 50, chunk_ms: 50, escape: false, frames,
    thorax: { pos_um: Array.from({ length: n }, (_, i) => [i * 1000, 0, 800]), yaw_deg: Array.from({ length: n }, () => 0) },
    ...extra,
  };
}

