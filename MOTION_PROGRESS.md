# Motion FrameGen: progress log

The engineering log of the [Motion engine](MOTION.md), newest first. Every Motion experiment adds an entry with its numbers.

Motion has its own version number, separate from the FramePortal app it ships in. It keeps the `-beta` tag, and Apple's engine stays the app's default, until Motion is visually clean enough for **v1.0**.

| Motion | Shipped in FramePortal | Summary |
|---|---|---|
| **v0.4-beta** | v1.10.0 | Frame-border handling, learned static-HUD mask |
| v0.3-beta | (branch only) | Dense anchored fields splatted to the midpoint, context-ring matching, temporal seeding |
| v0.2-beta | v1.9.4 | 5×5 match window, guarded vector median, confidence fallback |
| v0.1-beta | v1.8.0 | First engine: VT 16×16 motion, per-pixel selection, YCbCr warp |

How the numbers are measured:
- **Live:** a 3-minute 1080p session on a MacBook Air M2 with FrameGen at 60 FPS, analyzed from the session's perf trace.
  - Generated frame time: p50 / p95.
  - Missed 60 Hz slots: refreshes with no new frame.
  - Arrival → screen: from the network to the glass, p50 / p95.
  - Late frames: generated frames that missed their slot (the real frame was shown instead).
  - Memory slope: footprint minus the Instant Replay buffer.
- **Real footage (`motion-bench`):** a 22-second 1080p30 HDR gameplay recording (GTA VI). Frames i and i+2 make a midpoint, which is scored against the real frame i+1.
  - Y-PSNR covers the whole picture.
  - Edge PSNR covers strong-gradient pixels only: lines, poles, signs and silhouettes.
  - The gap is twice the live 30 → 60 step, so this is a harsh test.
- **Synthetic (an in-app bench, not in this repo):** a camera pan, an occluding object and a static HUD, scored against the exact truth.

## Current standing

| | Apple FrameGen | Motion v0.2-beta | **Motion v0.4-beta** |
|---|---|---|---|
| Real footage Y-PSNR | 28.17 dB | 27.85 dB | **28.66 dB** |
| Real footage edge PSNR | **26.61 dB** | 24.81 dB | 26.06 dB |
| Steadiness (all pixels) | 24.72 dB | 24.48 dB | **25.23 dB** |
| Generated frame p50 / p95 (offline bench) | 14.5 / 15.3 ms | ~11 / 15 ms | 12.0 / 13.9 ms |
| Synthetic HUD pixels wrong | 2.58 % | 7.6 % | **0.00 %** |
| Live generated frame p50 / p95 | 22.4 / 26.0 ms | 9.0 / 11.7 ms (11.3 / 21.1 with 60 FPS recording) | not measured yet |

**Verdict:** Motion v0.4 now beats Apple on overall quality and steadiness, and nearly matches it on thin edges. Apple stays the default until v0.4 has been checked live: translucent layers over fast road (a car's shadow, the minimap interior) and repeated objects still show artifacts.

---

## Motion v0.4-beta (app v1.10.0): frame borders and static HUD (2026-11-23)

Changes:
- **Frame borders:**
  - A motion vector whose endpoint leaves the frame is scored and warped from the in-frame side only; the edge row is never clamp-sampled.
  - Near the border, the match window skips taps outside the frame.
- **Static-HUD mask:**
  - Each guide pixel keeps two long-term means (about 1 s): brightness and edge strength.
  - An edge that persists in both means and in both current frames counts as static HUD.
  - A per-cell mask switches on and off with hysteresis.
  - Masked cells keep the zero vector, and no splat may cross into or out of them.
  - The mask learns only while the camera moves, so a still cutscene background is not mistaken for HUD.

Results, GTA reference clip (300 midpoints): Y-PSNR 28.66 dB, edge PSNR 26.06 dB, 12.0 / 13.9 ms. That is about 0.6 to 0.9 ms more than v0.3. Synthetic HUD pixels wrong: 13.1 % (v0.3) → 0.00 %.

Live clip review (1080p60 recording of a v0.2 session, real frames fed back in):
- **Bottom edge under the car's shadow:** large blotches → a few grey 8-px blocks.
- **Minimap:** the icons and the distance text stay intact, where v0.2 smeared them.

Finding: GTA's minimap is not static. The map scrolls 1-3 px a frame, so only its border and text are locked.

Open:
- Translucent or shadow layers over fast road: no single vector fits both layers.
- Repeated objects (roof lights, railings): a ghost copy in some frames.
- A thin lower sign pole.
- The live cost while recording.

## Motion v0.3-beta (branch only): dense anchored fields (2026-11-23)

The per-pixel winner-take-all choice among VideoToolbox block vectors was the weak point. v0.3 adds dense fields:
- **Dense fields:** A→B and B→A fields are built on a 4 px grid. Candidates come from the VideoToolbox blocks, then propagation and sub-pixel steps refine them.
- **Matching:** a 3×3 window 2 px apart, plus an 8-tap context ring 14 px out, so dark or flat areas are decided by their surroundings.
- **Temporal seeding:** the previous pair's fields are extra candidates, which cuts frame-to-frame jitter.
- **Splatting:** the fields are splatted to the midpoint. Forward-backward-consistent splats rank first, so thin foreground objects (poles, sign edges) keep their own motion.

| Step | Y-PSNR | Edge PSNR | p50 |
|---|---|---|---|
| v0.2 | 27.85 | 24.81 | ~11 ms |
| + splatted anchored fields, consistency rule | 28.29 | 25.69 | |
| + context ring | 28.52 | 25.87 | |
| + temporal seed | 28.64 | 26.04 | 15-18 ms |
| + 4 px field grid (same quality, search 7 → under 4 ms) | **28.67** | **26.03** | **11.1 ms** |

Measured neutral or worse, and removed:
- census cost, extra local candidates, splat pull, a zero-vector rule;
- a stronger smoothness prior (λ 0.002: Y 28.49);
- regional cost smoothing, a guide tone curve.

Neural Engine hybrid (not kept): a small quarter-res refinement net runs in 2.2 / 3.3 ms on the ANE, but it needs a trained model and does not fit on top of v0.4's budget yet.

## Motion v0.2-beta (app v1.9.4): cleaner selection on real footage (2026-11-23)

Changes:
- The match window grew from a 5-tap cross to a 5×5 grid 2 px apart (18×18 full-res pixels).
- A guarded 3×3 vector median runs before the warp.
- Low-confidence pixels fade to a 50/50 blend.
- New real-footage benchmark (now `motion-bench`).

Real footage, 300 midpoints (Motion alone):

| Variant | Y-PSNR | Edge PSNR | p50 |
|---|---|---|---|
| v1.9.3 baseline (5-tap cross, no median) | 27.61 dB | 24.19 dB | 7.9 ms |
| + 3×3 median | 27.67 | 24.25 | 9.3 |
| + 3×3 window (2 px) | 27.71 | 24.52 | 9.1 |
| + 5×5 window (2 px) | 27.77 | 24.70 | 9.1 |
| + 5×5 window + confidence fallback (**shipped**) | **27.85** | **24.81** | 9.7 |
| + 2 propagation passes | 27.80 | 24.86 | 15.3 |
| VT 4×4 blocks, multi-pass | 26.79 | 25.12 | 25.7 (VT alone 15 ms) |
| VT 4×4 blocks, single pass | 26.55 | 24.27 | 23.2 |
| 5×5 neighbouring blocks as candidates | 27.53 | 24.01 | 11.0 |
| Own GPU block matcher (16×16 full-res) as extra candidates | 27.24 | 24.23 | 10.2 (removed) |
| Stronger prior toward the block field (λ 0.002 to 0.004) | 27.14 to 27.59 | 24.34 to 24.49 | 9.2 |
| Apple FrameGen (reference) | 28.17 | 26.61 | 14.8 |
| 50/50 blend (reference) | 26.17 | 24.13 | |

What we learned:
- The VideoToolbox 16×16 field is very noisy in flat areas such as the sky.
- Adding candidates made results *worse*. The weak point is the per-pixel winner-take-all choice, not missing candidates: with more choices, a wrong vector matches a small window by accident more often.
- 4×4 blocks give the sharpest edges, but VideoToolbox needs 15 ms for them, which is too slow for 60 FPS.

Visual notes from the fast road sign at midpoint 100:
- **Baseline:** the arrow and "ONLY" were torn apart, the pole was missing, power lines were doubled, and the minimap "N" vanished.
- **v1.9.4:** the arrow and "ONL" are readable again, the pole is partly back, and the truck edges are cleaner.
- **Still visible:** tearing on the sign edges and fragments at the pole base.

Memory: an offline run of the Motion engine over the whole clip stayed flat at 69 MB.

## Live A/B on PS5, Motion v0.1-beta (app v1.9.3, 2026-11-23)

These are 3-minute 1080p60 live sessions, same day, same network conditions class:

| | Apple FrameGen | Motion v0.1-beta |
|---|---|---|
| Generated frame time p50 / p95 | 22.4 / 26.0 ms | 9.0 / 11.7 ms |
| Late frames (real shown instead) | 922 | 1 |
| Missed 60 Hz slots | 9.8 % | 1.8 % |
| Arrival → screen p50 / p95 | 76.4 / 96.5 ms | 66.7 / 84.3 ms |
| Decoded → render start p50 | 43.1 ms | 26.6 ms |
| Commit → glass p50 | 16.2 ms | 32.3 ms |
| Memory slope after warm-up | +2.6 MB/min | +13.4 MB/min (footprint minus replay, includes a live 1080p → 1440p output switch) |

Player feedback: Motion feels noticeably smoother and snappier, but it has too many jitters and artifacts around edges and motion. Apple stays the default.

Open questions from this run:
- Commit → glass is two refreshes with Motion.
- The memory slope needs a long run with no output-size change.

## Motion v0.1-beta (app v1.8.0): baseline (2026-11-23)

- **Pipeline:** half-res guides, then VT 16×16 multi-pass in both directions, then per-pixel selection among 19 candidates (5-tap cross SAD, zero-vector bias, occlusion side choice), then a full-res YCbCr warp.

Synthetic bench, 1080p, 60 runs:

| | Y-PSNR | Object PSNR | HUD pixels wrong | p50 / p95 |
|---|---|---|---|---|
| Motion | 27.0 dB | 19.8 dB | 2.61 % | 6.3 / 18.2 ms |
| Apple FrameGen | 19.6 dB | 26.1 dB | 2.58 % (with static preserve) | 15.3 / 16.4 ms |
| 50/50 blend | 24.4 dB | 14.1 dB | 0.00 % | |
