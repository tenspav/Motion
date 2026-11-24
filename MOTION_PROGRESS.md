# Motion FrameGen: progress log

The engineering log of the [Motion engine](MOTION.md), newest first. Every Motion experiment adds an entry with its numbers.

Motion has its own version number, separate from the FramePortal app it ships in. It keeps the `-beta` tag, and Apple's engine stays the app's default, until Motion is visually clean enough for **v1.0**.

| Motion | Shipped in FramePortal | Summary |
|---|---|---|
| **v0.6-beta** | v1.14.0 | Pass 3 on by default and 1.3 ms (p95 3.1 ms) cheaper than v0.5's; pair-local subtitle guard replaces the overlay lock |
| v0.5-beta | v1.12.0 | Faster pipeline (fused upsample + warp, no CPU waits), occlusion only when proven, soft upsample and per-pixel overlay lock |
| v0.4-beta | v1.10.0 | Frame-border handling, learned static-HUD mask |
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

| | Apple FrameGen | Motion v0.2-beta | Motion v0.4-beta | Motion v0.5-beta (P3=7) | **Motion v0.6-beta** |
|---|---|---|---|---|---|
| Real footage Y-PSNR | 28.17 dB | 27.85 dB | 28.66 dB | 29.41 dB | **29.39 dB** |
| Real footage edge PSNR | 26.61 dB | 24.81 dB | 26.06 dB | 26.67 dB | **26.68 dB** |
| Steadiness (all pixels) | 24.72 dB | 24.48 dB | 25.23 dB | 25.82 dB | **25.81 dB** |
| Generated frame p50 / p95 (offline bench) | 14.5 / 15.3 ms | ~11 / 15 ms | 12.0 / 13.9 ms | 14.5 / 18.5 ms (P3 off: 10.9 / 12.7) | **13.3 / 15.4 ms** |
| Synthetic HUD pixels wrong | 2.58 % | 7.6 % | **0.00 %** | **0.00 %** | **0.00 %** |
| Live generated frame p50 / p95 | 22.4 / 26.0 ms | 9.0 / 11.7 ms (11.3 / 21.1 with 60 FPS recording) | not measured yet | not measured yet | not measured yet |

**Verdict:** Motion v0.6 ships the Pass 3 picture by default (soft upsample, subtitle guard, frame-edge fade) at the v0.5 P3=7 quality (−0.02 dB Y, +0.01 dB edge) for 1.3 ms less p50 and 3.1 ms less p95. It still costs +2.4 ms p50 / +2.7 ms p95 over v0.5's default (Pass 3 off): the target was +1.0 / +1.5 ms and is missed. The subtitle and the trunk are clean; a faint outline and a small dark square can remain next to the first letter of a subtitle line over moving shadow, where v0.5's overlay lock was cleaner. Apple stays the default until a live A/B.

---

## Motion v0.6-beta: experimental 120 FPS (4x mode) (2026-11-24)

Three made frames per real pair (t = 0.25, 0.5, 0.75) for 120 Hz displays, off by default (the app settings, shown only at 120 Hz). Pair-level work runs once; splat, select, grid fix and the warp run per t from time-specialised kernels.

**60 FPS path unchanged:** with the 4x code in, the GTA bench gives the same dB on every one of the 300 midpoints, and the first 120 dumped midpoints are byte-identical PNGs.

GTA VI clip, 4x bench, 150 pairs (frames i and i+4, scored against the real i+1..i+3; four times the live 30 → 120 step):

| t | Motion Y / edge | Plain blend (1 − t)·A + t·B, Y / edge |
|---|---|---|
| 0.25 | 27.16 / 24.63 dB | 25.38 / 22.80 dB |
| 0.50 | 26.55 / 24.07 dB | 24.69 / 22.59 dB |
| 0.75 | 27.34 / 24.67 dB | 25.42 / 22.76 dB |

Whole pair (three made frames, 1080p, MacBook Air M2, five runs): p50 20.5-22.1 ms, p95 27.3-29.6 ms; GPU p50 / p95 about 17 / 25 ms. The pair must finish within the 33 ms real-frame period, so the cost guard falls back to 60 FPS when the median over 24 pairs passes 25 ms or 6 pairs pass 30 ms. A first try with 20 / 25 ms tripped within 17-23 pairs on this machine; at 25 / 30 ms the replayed guard does not trip on the GTA clip. 4K would cost about four times as much and is expected to fall back on an M2.

Visual (20-55-31 @01:45, pair 72): the three frames step evenly between A and B, the subtitle stays whole in all three; the small ghost left of the first letter (see v0.6 below) shows at t = 0.25 / 0.5 / 0.75 too.

Not checked: live presentation on a 120 Hz display (no session run), recording with 4x on (the recording stays 60 FPS by design).

---

## Motion v0.6-beta (v1.14.0): Pass 3 by default, subtitle guard (2026-11-24)

Bench: the GTA VI clip, 300 midpoints, three interleaved rounds against the v0.5 binary (MacBook Air M2, background load from the network client and a browser). dB are identical in every round; the times are the means of the three rounds.

| Build | Y / edge / steady (dB) | Frame p50 / p95 (ms) | GPU p50 / p95 (ms) |
|---|---|---|---|
| v0.5 default (P3 off) | 28.83 / 26.21 / 25.44 | 10.89 / 12.68 | 7.38 / 9.17 |
| v0.5 P3=7 | 29.41 / 26.67 / 25.82 | 14.51 / 18.47 | 10.96 / 14.79 |
| **v0.6 default** | **29.39 / 26.68 / 25.81** | **13.25 / 15.36** | **9.63 / 11.60** |
| v0.6, subtitle guard off (`MFG_GLYPH=0`) | 29.40 / 26.70 / 25.82 | 12.44 / 14.39 | 8.79 / 10.70 |
| v0.6, shadow retry on (`MFG_SHADOW=1`) | 29.39 / 26.69 / 25.81 | 13.32 / 15.36 | 9.69 / 11.62 |

Gates: Y ≥ 29.36 and edge ≥ 26.62 (the v0.5 P3=7 values minus 0.05) pass. Synthetic HUD pixels wrong: 0.00 % (Y 26.1 dB, as v0.5 P3=7). Cost: missed (target +1.0 / +1.5 ms over the v0.5 default, measured +2.4 / +2.7).

### What changed

- **Soft upsample in registers.** v0.5 gathered up to 10 merged vectors per pixel in a dynamically indexed array; the spilled registers made the warp stage about 4 ms slower than the hard pick. v0.6 streams the 3×3 cells into at most three entries (a cell joins an entry within `max(2 px, 0.3 |v|)` on the same side), scores them at the pixel and warps the three strongest. Tried and dropped: a fixed 9-slot array (as slow), one thread per 2×2 pixels sharing the gather (slower: less parallelism). Cost scale 0.12 (was 0.06) and unchanged-pixel rule 0.004 (was 0.012): +0.04 dB. Two entries instead of three: −0.03 dB, 0.1 ms faster, more 8-px blocks on shadowed road.
- **Pair-local subtitle guard (`MFG_GLYPH`, replaces the overlay lock).** Per pair, in the lower band of the picture (70-98 % of the height): bright strokes next to dark edges are seeds. A seed is steady (same place, same shape in A and B, not explained by the local world motion), an onset (in B only) or an offset (in A only). Seeds must group (three 8-px bins of 7 along the row) to count; the mask grows over unchanged pixels and bright strokes near a seed bin. Steady pixels take the plain blend, offset pixels take A; a warp never samples a glyph pixel of a frame that holds it (anti-copy: that side is dropped; both sides: the blend). No stacked duplicates in the checked frames. The world vector is the VideoToolbox vector that best explains a 4×4 patch (zero excluded): with the zero vector allowed, every steady subtitle was "explained" and the guard found nothing; with any vector allowed, road grain made false seeds (−0.5 dB). Rows that cannot reach a glyph run a copy of the warp kernel compiled without the guard; the band rows run the guarded kernel only within 48 px of a glyph row of this pair (indirect dispatch). Cost: about 0.85 ms p50 (seeds 0.29 ms, mask 0.13 ms, warp +0.4 ms).
- **Shadow-over-road zero-mean retry (`MFG_SHADOW`, replaces `MFG_CMATCH`).** A cell whose winner and best-supported neighbour vector differ by more than 1 px and whose windows differ in mean brightness retries both with a zero-mean match; the alternative is admitted only if it is better by a margin, correlates (≥ 0.75) and passes a forward-backward check. Admitted cells keep both endpoints and are not stamped as 8×8 blocks. **Left off:** neutral on the GTA bench (±0.01 dB), no visible change in the palm-shadow crops (it admits almost no cell at these thresholds; looser thresholds were also neutral), +0.1 ms.
- **Frame-edge fade (`MFG_EDGE`)**, unchanged from v0.5, on.
- **`MFG_SOFTK`:** the coordinator's run of `MFG_P3=7 MFG_SOFTK=0` on the v0.5 binary gave the exact P3=7 numbers. In this run the switch does reach the kernel on the fused path (60 midpoints, v0.5 binary: 30.01 → 29.86 dB), so the identical result most likely means the variable never got into the bench's environment. In v0.6 the register-only soft path always uses the 3×3 tent; `MFG_SOFTK` applies only to the v0.5 loop (`MFG_SOFTV2=0`).

### Visual checks (rows a / live / v0.5 default / v0.5 P3=7 / v0.6 / b)

| Clip | Verdict |
|---|---|
| 20-55-31 @01:45, frames 288 / 292 / 296: palm shadows, subtitle, minimap, trunk | Minimap and trunk as clean as P3=7. Subtitle whole, no duplicate line; a faint outline and a 4-px dark square can stay left of the first letter ("I") over the moving shadow, where P3=7 is clean. Palm shadows over road: soft, a few weak cell edges left (P3=7 has slightly fewer). |
| 20-55-31 @01:15 | Same as P3=7 (35.34 vs 35.38 dB, edge 31.17 vs 31.03). |
| 18-11-45 @00:20 and @01:25 | Same as P3=7 (35.44 / 35.52 vs 35.45 / 35.50 dB). Wall texture clean. |
| 17-32-49 @00:40 (roof, marquee) | Same as P3=7 (26.94 vs 26.95 dB). |

### Still open

- Cost: +2.4 ms p50 over the v0.5 default. The soft upsample itself is about 1.0 ms, the subtitle guard 0.85 ms, the rest is the frame-edge fade and the larger kernel.
- The first letter of a subtitle line over moving shadow (see above).

---

## Motion v0.5-beta (v1.12.0): faster pipeline, proven occlusion (2026-11-23)

### Pass 1: same output, less time

Each step kept the output (GTA reference clip 28.66 / 26.06 / 25.23 dB; the per-frame values match to 0.001 dB). GPU stage times on an M2, 1080p:

| Step | Before | After |
|---|---|---|
| Anchored search: distinct-vector list in threadgroup memory, 8×8 groups, options as pipeline constants | 2.16 ms | 1.33 ms |
| Upsample: nearest cell from the four already read, duplicate vectors scored once, early exit | 1.57 ms | |
| Fused upsample + warp (`mfg_upwarp`, `MFG_FUSE=0` for the staged path) | 1.57 + 0.75 ms | about 1.97 ms |
| Propagation: unrolled 8-neighbour de-duplication | 0.70 ms | 0.63 ms |
| CPU: no waits (guides start both motion searches, buffers made while they run), one-pass field statistics | 0.29 ms post | 0.26 ms, overlapped |
| Whole pair, GPU | 8.1 ms | 6.8 ms |
| Generated frame p50 / p95 (offline bench, quiet machine) | 11.8 / 14.4 ms | 10.4 / 12-13 ms |

The 9-9.5 ms target was not reached. Finding: taking the nearest cell's vector only (no per-pixel choice among four) is faster (0.58 ms warp) and even scores higher (28.75 / 26.17), but its output differs, so it was left out of Pass 1.

### Pass 2: quality

GTA reference clip, 300 midpoints:

| Variant (`MFG_P2`) | Y-PSNR | Edge PSNR | Steadiness |
|---|---|---|---|
| v0.4 (`MFG_V04=1`) | 28.66 | 26.06 | 25.23 |
| proven occlusion only (1) | 28.78 | 26.19 | 25.35 |
| + interpolated fade cost (3) | 28.80 | 26.21 | 25.34 |
| + bilinear vector candidate (7) | 28.84 | 26.23 | 25.39 |
| **default: occlusion + bilinear + sub-pixel + 1-ring pull (85)** | **28.83** | **26.21** | **25.44** |
| 87 (85 + interpolated fade cost) | 28.86 | 26.25 | 25.44 |

Synthetic bench (default): HUD pixels wrong 0.00 %. The interpolated fade cost (bit 2) blurs the synthetic object (−1.7 dB, HUD 0.10 % wrong), so it stays off. The unique-match margin (bit 8) measured worse (28.49 / 25.95) and added blocks on the roof lights, so it stays off. The 1-ring pull costs 0.10 ms. The edge PSNR target of 26.6 was not reached.

Visual checks, real frames fed back in, v0.4 and v0.5 crops side by side with the recorded frame:
- **20-55-31 @01:45 (priority):** the dark block above the minimap is nearly gone (a faint smudge), the block left of the subtitle is mostly gone, and dark dropout blocks along the bottom edge are fewer. The 8-px stair steps on fast palm shadows remain, and so do the grey blocks on the right-edge trunk. Two frames later both versions are badly broken (subtitle letters missing, text fragments copied below it). **Not clean: reduced, not gone.**
- **20-55-31 @01:15:** no regression.
- **18-11-45 @00:20:** left-edge blocks slightly reduced; road patches about the same.
- **18-11-45 @01:25:** blocks along the minimap's bottom edge slightly reduced, still visible.
- **17-32-49 @00:40 (roof lights):** the ghost of the fourth light is reduced. In the frame with the largest motion, dark fragments between the lights remain, slightly fewer.

Open:
- Fast shadows over road: 8-px stair steps and blocks.
- Moving subtitles over moving background: broken text in some frames.
- Contrast-adjusted matching and a per-pixel HUD lock: not attempted.
- The live cost while recording: not measured.

### Pass 3: soft upsample, overlay lock, frame-edge fade

Design: see "v0.5 Pass 3" in [MOTION.md](MOTION.md). Each change sits behind a bit of `MFG_P3`. **Shipped off by default (`MFG_P3=0`)**: the cost misses the 60 FPS deadline at p95 on an M2 (table below).

GTA reference clip, 300 midpoints:

| Variant (`MFG_P3`) | Y-PSNR | Edge PSNR | Steadiness |
|---|---|---|---|
| Pass 2 (0) | 28.83 | 26.21 | 25.44 |
| frame-edge fade + soft upsample (5) | 29.45 | 26.74 | 25.87 |
| **default: + overlay lock (7)** | **29.41** | **26.67** | **25.82** |
| soft upsample on the 2×2 cells only (`MFG_SOFTK=0`) | about 29.27 | 26.56 | 25.69 |
| + contrast-adjusted match (15) | neutral to −0.02 | | |
| + HUD cells' moving vector (39) | +0.09 | | |

Synthetic bench (60 runs): Y-PSNR 25.6 → 26.1 dB, object 18.7 → 19.4 dB, HUD pixels wrong 0.00 %. While the HUD mask is still learning (30 runs) the soft upsample first raised HUD errors to 32 %; the unchanged-pixel rule (`MFG_SOFTH`) brought them to 7.7 % (Pass 2: 12.1 %).

Root causes found:
- **Stair steps:** a hard per-pixel pick among only the 2×2 surrounding cells, plus a per-cell fade cost. Fixed by the soft upsample.
- **Subtitle breakup:** the warp sampled locked text (often with the other side outside the frame at the bottom edge, where the border branch fell back to B). The semi-transparent first letter differs slightly between frames, so it needed the weak-lock rule.
- **Edge and trunk blocks:** the border branch. Border pixels had no confidence fade, so a bad one-sided sample was shown at full strength. The fallback cost was not the cause.
- **Nearest-cell upsample** (the 0.58 ms option from Pass 1): not adopted. It only speeds up the hard path, which the soft upsample replaces wherever cells disagree.

Cost, GPU stage times on an M2, 1080p (machine busy, load 5-8, so the absolute numbers are high; the differences hold across runs):

| Stage | Pass 2 | Pass 3 |
|---|---|---|
| Overlay lock | | 0.55 ms |
| Select (+ lock dilation) | 1.38 ms | 1.65 ms |
| Upsample + warp | 2.41 ms | 5.47-5.70 ms |
| Generated frame p50 (synthetic bench) | 6.3-7.5 ms | 11.3-11.6 ms |

#### Pass 3 cost per bit (merge check)

GTA reference clip, 300 midpoints, interleaved runs on the same (busy) M2, made frame p50 / p95 and GPU p50 / p95:

| `MFG_P3` | Y / edge / steady (dB) | Frame ms | GPU ms |
|---|---|---|---|
| 0 (Pass 2, shipped default) | 28.83 / 26.21 / 25.44 | 11.0 / 12.8 | 7.3 / 8.7 |
| 1 (soft upsample) | 29.42 / 26.70 / 25.92 | 13.6 / 16.3 | 9.8 / 12.3 |
| 3 (+ overlay lock) | 29.41 / 26.65 / 25.91 | 14.7 / 18.3 | 11.0 / 14.2 |
| 7 (+ frame-edge fade) | 29.41 / 26.67 / 25.82 | 15.7 / 19.0 | 11.7 / 15.0 |

The soft upsample carries the whole score gain; the lock and the edge fade are what clear the subtitle and edge blocks visually. 19 ms p95 misses the 16.7 ms deadline at 60 FPS, so Pass 3 is off by default in v0.5-beta. Next (v0.6): make the soft upsample and the lock cheaper, then turn Pass 3 on.

Visual checks, real frames fed back in, rows A, live recording, v0.4, v0.5 Pass 2, v0.5 Pass 3, B:
- **20-55-31 @01:45, frames 288 / 292 / 296 (priority):** the bottom-edge blocks, the stair steps on the palm shadows, the grey blocks on the right-edge trunk and the minimap-edge blocks are gone; the road shadows are soft and continuous. The subtitle is whole in all three frames (the first letter "I" included) and no text is copied below it. Left: a faint smear on the trunk in 292 and soft ghosting on the fastest shadows. **Clean in the checked frames.**
- **20-55-31 @01:15:** no regression; the subtitle and minimap are clean, the small specks on the dirt from v0.4 / Pass 2 are gone.
- **18-11-45 @00:20:** no regression; the block patches inside the dark car body are gone.
- **18-11-45 @01:25:** the blocks along the minimap's bottom edge and right of it are gone; a soft grey smear remains under the minimap, where the frame changes most.
- **17-32-49 @00:40 (roof lights):** the dark fragments between the lights are gone and the outlines are cleaner than v0.4 and Pass 2. A thin halo stays at the fourth light.

Open:
- The cost: the soft upsample takes about 3 ms more on the GPU. `MFG_SOFTK=0` saves about 2 ms for −0.15 dB.
- The live cost (playing and while recording at 60 FPS): not measured.
- Soft smears remain where the frame changes most (no vector explains it): a blend, not a block.

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
