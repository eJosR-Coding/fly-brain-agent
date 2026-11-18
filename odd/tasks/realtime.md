# Real time: brain on the GPU, faster physics, then a live session

## Objective
Bring the brain-body loop close to real time on the user's laptop (RTX 3050 Ti, 4 GB VRAM, 16 CPU threads) WITHOUT
changing validated results and WITHOUT stressing the GPU. Requested 2026-11-18 ("migremos y agrandemos paso de tiempo pero
que no queme mi gpu ... mi vram no es mucha").

## Evidence (spike, 2026-11-18)
- CuPy LIF (2 RawModule kernels, trials batched as columns, float32, counter-based Poisson RNG) matched the CPU over
  10 trials: sugar MN9 63.5 vs 61.2 Hz; sugar+bitter 1.9 vs 1.9; active neurons 345 vs 342; per-neuron r = 0.998.
- Speed per simulated second: CPU 4.6 s/trial; GPU 0.17 s (1 trial), 0.55 s (3), 1.59 s (10). 50 ms chunks: 9 ms (1 trial),
  28 ms (3). VRAM 0.35-0.65 GB. Needs `cupy-cuda13x[ctk]` (~2 GB venv); driver 620.70 (CUDA 13.5).
- Physics: flygym/MuJoCo on CPU ~8-9 s per simulated second at dt = 1e-4 s. MJX (GPU) only pays off with many parallel
  bodies and JAX preallocates VRAM: physics stays on the CPU.

## Safety rules (hard)
- GPU is optional (extra `gpu`); automatic CPU fallback on any GPU error or when free VRAM is below a safe margin.
- Never preallocate the whole card; check free VRAM before loading; cap live-mode trials (1-3); idle when no run.
- A health readout (VRAM used/free, temperature, utilization via nvidia-smi) so the user can see the load.
- Results stay statistically equivalent to the CPU (tests); not bit-exact (documented).

## Tasks
- [x] G1 GPU brain backend in the project (gpu_simulation.py, FLYBRAIN_SIM_BACKEND=auto|cpu|gpu, VRAM guard, spike
      buffer overflow check, health readout, statistical equivalence tests gated on a GPU).
      Progress G1: src/flybrain/gpu_simulation.py (GPUSimulation(Simulation), two CuPy kernels, VRAM guard,
      overflow raises), sim_service backend selection + CPU fallback, /health/gpu, `gpu` extra. Real brain, sugar 100 Hz,
      1 s: 1 trial CPU 4.5 s vs GPU 0.46 s; 3 trials 13.1 s vs 1.2 s (~10x); MN9 CPU 59-60 Hz vs GPU 63-64 Hz.
      Peak 71 C under load, 234 MiB; resting 52 C. Follow-up: thermal cutoff (refuse new GPU runs above ~80 C).
- [x] G2 physics timestep: measure dt 1e-4 (reference) vs 2e-4 / 3e-4 / 5e-4 (stability, stepping frequency, tripod,
      speed, joint-angle error); adopt the largest that matches; report the real-time factor.
      Progress G2: default dt 2e-4 s (2.0x faster, real-time factor 8.4 -> 4.3 s per simulated s on 1 CPU thread). 3e-4 passes the
      band too but does not divide the 200 Hz frame; 5e-4 fails (speed +23 %, bouncing). export_flygym_walk.py pinned to 1e-4;
      fly-walk.json regenerates byte-identical. Gated test in tests/test_body_runner.py.
- [ ] G3 live session over WebSocket: brain and body advance together in 50 ms chunks, closed loop, streamed to the viewer.

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.
