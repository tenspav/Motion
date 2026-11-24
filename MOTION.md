# Motion FrameGen

Motion is a real-time frame-generation engine for Apple silicon. It turns a 30 FPS game stream into 60 FPS by building one new frame between every two real frames. It is an alternative to Apple's built-in interpolator (`VTLowLatencyFrameInterpolation`), and it is currently in **beta**. It was built for low-latency PS5 Remote Play streaming and runs in the FramePortal app.

**Current version: Motion v0.5-beta.** Motion is versioned separately from the app. It becomes v1.0, and the default engine, once it is visually as clean as Apple's.

Engineering log and benchmark history: [MOTION_PROGRESS.md](MOTION_PROGRESS.md).

## In plain English

Two frames arrive 33 ms apart. Motion works out how each part of the picture moved between them, then draws the picture halfway along that movement. The result is shown 16.7 ms after the first real frame, so the display gets a new image every refresh.

The hard parts:
- **Things that don't move** (HUD, minimap, button prompts) must stay perfectly still.
- **Things that appear or disappear** behind a moving object (occlusion) have no match in one of the two frames.
- **Thin fast objects** (poles, signs, power lines) are smaller than a motion block and easy to tear.
- **Flat areas** (sky, walls) match any motion, so a naive search picks random vectors that flicker.

## Architecture

```
real frame A ─┐                                      ┌─ made frame (t = 0.5)
real frame B ─┤                                      │
              ▼                                      │
   mfg_guide (Metal)        half-res 8-bit luma guides (A and B)
              │
              ▼
   VTMotionEstimationSession  16×16 blocks on the guide (32×32 full-res),
   (hardware, VideoToolbox)   multi-pass, both directions (A→B and B→A), run concurrently
              │
              ▼
   mfg_select (Metal)       per guide pixel: pick the best displacement among the zero vector and
                            the vectors of the 3×3 neighbouring blocks in both fields
              │
   mfg_refine (optional)    propagation passes: try neighbours' chosen vectors and ±1 px
              │
   mfg_smooth (Metal)       guarded 3×3 vector median (kills pixel-to-pixel flicker)
              │
              ▼
   mfg_warp_y / mfg_warp_c  full-res warp of the decoded YCbCr planes (luma + chroma),
   (Metal)                  10-bit in HDR, blend of both sides or one side on occlusion
```

The decoded frames are never converted to RGB. The warp reads the decoder's own bi-planar YCbCr buffers (8-bit, or 10-bit `x420` in HDR) and writes the made frame in the same format. The renderer then treats a made frame exactly like a real one. Only the guides used for motion search are 8-bit half-res luma.

### Since v0.3: dense anchored fields, borders and static HUD

The block-vector selection above was v0.2's weak point: one winner per pixel among noisy 16×16 VideoToolbox vectors. Since v0.3, Motion builds dense fields first, then selects.

1. **Anchored fields (A→B and B→A), on a 4 px grid of the half-res guide.**
   - Every cell tests the VideoToolbox block vectors around it and the previous pair's vectors (temporal seed).
   - Propagation steps (4 px, then 1 px) and sub-pixel steps refine the result.
   - The match uses a 3×3 window 2 px apart plus an 8-tap context ring 14 px out, so flat or dark areas are decided by their surroundings instead of by look-alike texture far away.
2. **Splat.** Each field is forward-splatted to the midpoint.
   - Forward-backward-consistent vectors rank first, then the one furthest from the camera motion, so thin foreground objects keep their own motion.
   - The splatted vectors join the midpoint selection as extra candidates. The grid is brought back to guide pixels by a per-pixel choice among the four surrounding cells.
3. **Frame borders.**
   - A vector whose endpoint leaves the frame is scored and warped from the in-frame side only.
   - The match window skips taps outside the frame, so the edge row is never clamp-sampled.
4. **Static-HUD mask.**
   - Long-term means (about 1 s) of the guide and of its edge strength find edges that stay put while the camera moves.
   - A per-cell hysteresis mask keeps those cells on the zero vector, and no splat crosses into or out of them.

### Since v0.5: a faster pipeline, proven occlusion only

- **Fused upsample and warp** (`mfg_upwarp`): one thread per guide pixel picks its vector from the four surrounding grid cells and writes its 2×2 luma block and its CbCr texel, so the per-pixel field never goes through memory. The staged path (`mfg_upsample`, `mfg_warp_y`, `mfg_warp_c`) stays for the debug views, `MFG_REFINE` and `MFG_SMOOTH`.
- **Anchored search:** the tap spacing, ring and propagation options are pipeline constants; the distinct-vector list lives in threadgroup memory and the kernel runs in 8×8 groups.
- **No CPU waits:** the guides' completion starts both motion searches; the output buffer and the plane textures are made while they run, and the pair continues on whichever side finishes last. Still one pair in flight.
- **Occlusion only when proven:** a pixel with no good two-sided match used to take the vector nearest the camera motion and sample one frame only. On fast road this pasted unrelated content (dark 8-px blocks). Now a pixel is one-sided only when one end of the background vector clearly lands on the foreground (foreground match under 0.04, the other end 0.02 worse). Otherwise it keeps its cheapest two-sided match and fades toward the 50/50 blend as before.
- **Upsample:** the bilinear blend of the four cells' vectors is one more per-pixel candidate (a smooth field keeps its gradient instead of stepping between cells), and a pixel of a one-sided cell takes a two-sided vector when one matches well there.
- **Sub-pixel fit:** on the last propagation pass, a parabola through the ±0.5 px costs refines confident winners; it is kept only if its own cost is lower.
- **1-ring pull:** a midpoint cell whose match cost is over 0.08 tries the vectors its 8 neighbours chose (static-HUD cells excluded).

### v0.5 Pass 3: soft upsample, overlay lock, edge fade

- **Why the stair steps:** every full-res pixel picked one vector among the 2×2 surrounding 8-px cells, with that cell's fade cost. On fast shadows over road two vectors match about equally well, so neighbouring pixels flipped between them along the cell grid: an 8-px staircase. A good vector also could not reach past its own cell.
- **Soft upsample (`P3_SOFT`):** each pixel gathers the vectors of the 3×3 cells around it (a tent 1.5 cells wide), merges vectors closer than 2 px, scores each at the pixel and weights it by `exp(-(c - c_min) / 0.06)`. A clear winner stays a hard edge; a near tie becomes a smooth cross-fade (overlapped block motion compensation). The fade toward the 50/50 blend is mixed the same way, so it has no 8-px steps either. A pixel that did not change and wins on the zero vector keeps it alone (static HUD before the mask has learned it).
- **Overlay lock (`P3_LOCK`, `mfg_lock` + `mfg_lockdil`):** per full-res pixel, text over a moving background is found without the learned HUD mask: the pixel is nearly equal in A and B (under 0.02), most of its 3×3 neighbours too, and its 4×4 patch holds a bright stroke (over 0.45) next to a dark shadow (under 0.2). Locked pixels take the plain blend of A and B and are never warped. A moving pixel whose vector would sample a locked pixel drops that side (both locked: the blend; the other side outside the frame: the edge row). Weak locks (under 0.15) count only within 3 guide px of a strong one, which keeps semi-transparent strokes whole.
- **Frame edge (`P3_EDGE`):** border pixels (one end of the vector outside the frame) now fade toward the blend by their match cost like every other pixel, and the two sides are weighted by their distance to the edge instead of switching hard. This removed the grey and dark blocks on the frame edges and the trunk.
- **Tried, left off:** a contrast-adjusted 5-tap match for ambiguous cells (`MFG_CMATCH`, neutral to slightly worse) and offering a static-HUD cell's moving vector to its pixels (`MFG_HUDPX`, +0.09 dB for +1.4 ms).

Per-stage GPU time on an M2 (1080p, v0.4 before the fused path):

| Stage | Time |
|---|---|
| Anchor + splat | about 4.0 ms |
| Select | 1.3 ms |
| Upsample | 1.3-1.7 ms |
| Warp | 0.75 ms |
| Guide | 0.5 ms |
| VideoToolbox motion (CPU wait) | 2.35 ms |

### Since v0.6: Pass 3 by default, subtitle guard

- **Soft upsample in registers:** the 3×3 cells around a pixel stream into at most three merged entries (vectors within `max(2 px, 0.3 |v|)` on the same side), each scored at the pixel and weighted `exp(-(c - c_min) / 0.12)`; the three strongest are warped. Same picture as the v0.5 loop within 0.03 dB, about 3 ms cheaper at p95 (no spilled registers).
- **Pair-local subtitle guard (`P3_GLYPH`, `mfg_gseed` + `mfg_gfinal`, replaces the overlay lock):** in the lower band of the picture, bright strokes next to dark edges that the local world motion (the VideoToolbox vector that best explains the patch, zero excluded) does not explain are glyph seeds: steady (in A and B), onset (B only) or offset (A only). Grouped seeds make a mask, grown over unchanged pixels and bright strokes nearby. Steady glyph pixels take the plain blend, offset pixels take A, and no warp samples a glyph pixel of a frame that holds it (that side is dropped). The band rows run the guarded warp kernel only near a glyph row of this pair; all other rows run a copy compiled without it.
- **Shadow retry (`P3_SHADOW`, off):** a zero-mean match for cells whose two best vectors disagree under a brightness change (shadow over road). Admitted cells keep both endpoints. Neutral on every bench clip, so it stays off.

### Experimental: 120 FPS (4x mode)

On a 120 Hz display Motion can make three frames per real pair instead of one: t = 0.25, 0.5 and 0.75 (30 → 120 FPS). In FramePortal it is an experimental setting, off by default, shown only when Motion is on and the display reaches 120 Hz; streams up to 4K. The API is `interpolate4(prev:cur:)`, which returns the three frames in order.

- **One pair's work, three frames:** guides, both motion searches, the anchored fields, the subtitle mask and the static-HUD evidence run once per pair. Splat, select, grid fix and the fused warp run once per t, from kernels specialised with the time (`kT`): A is sampled at m + t·d and B at m − (1 − t)·d, the A field splats t of its way and the B field 1 − t, and A and B blend (1 − t) : t. The extra passes do not update the per-pair state (`kPairWork`). With t = 0.5 the kernels compile to the 60 FPS code: the 60 FPS output is bit-identical with or without the 4x passes built.
- **Presentation:** the three made frames, then the real one, each on its own refresh (`PacedCadence` at 120, stride 1; `PresentQueue` 6 deep for the four-frame bursts).
- **Cost guard:** the whole pair is timed. A median over 25 ms (24 pairs) or 6 pairs over 30 ms falls back to Motion's 60 FPS midpoint, logged ("Motion 120 FPS stepped down to 60 FPS").
- The 60 FPS recording keeps the midpoint only.

### Shader stages

| Stage | Resolution | What it does |
|---|---|---|
| `mfg_guide` | half | Full-res Y plane → 8-bit half-res luma guide (2×2 box filter from one bilinear tap). The guide of the previous frame is reused for the next pair. |
| VT motion | 16×16 blocks on the guide | Apple's hardware motion estimation in both directions: `motion(of: B, comparedTo: A)` and the reverse. |
| `mfg_select` | half | For each pixel `m`, it tests the candidate displacements `d` (zero, vB of the 3×3 blocks, −vA of the 3×3 blocks, de-duplicated at a quarter pixel). The cost is the SAD between A at `m + d/2` and B at `m − d/2`, over a 5×5 grid of taps 2 px apart (18×18 full-res pixels, centre-weighted), plus a small prior toward the local block field. The cheapest candidate wins. |
| `mfg_refine` | half | Optional (off by default). Each pixel also tries the vectors chosen 8, 4, 2 and 1 px away, plus ±1 px around its own. |
| `mfg_smooth` | half | 3×3 vector median: the vector closest to all nine neighbours. It is accepted only if it matches this pixel nearly as well as the pixel's own vector, so real motion edges and the static HUD keep their vectors. Occlusion pixels are never touched. |
| `mfg_warp_y` | full | Samples A at `p + d/2` and B at `p − d/2` (bilinear) and averages them, or uses one side only on occlusion. Low-confidence pixels fade toward a plain 50/50 blend: a soft ghost instead of a tear. |
| `mfg_warp_c` | half (chroma) | The same for the CbCr plane. |

### Static HUD locking

- **Zero-vector bias:** the zero vector gets a small cost bonus, so a pixel that is identical in both frames stays still even when a moving vector matches almost as well.
- **HUD-edge rule:** if no vector matches well but the pixel itself did not change between A and B, it keeps the zero vector. This covers HUD text over a moving background, where the match window reaches the moving world.
- **Guarded median:** smoothing never replaces a pixel's vector with one that matches worse, so the median cannot drag the HUD along with the scene.

### Occlusion

When no candidate explains both frames, the pixel is background that is visible from only one side.
1. Motion takes the candidate closest to the camera motion as the background vector `g`. The camera motion is the median of the whole vector field.
2. It takes the cheapest clearly different candidate as the foreground vector `f`.
3. It checks which end of `g` lands on something that moved with `f`, and samples only the other frame.

If more than 30% of the picture is occluded or unmatched (a scene cut, for example), the pair is skipped and the real frame is shown instead.

### Why it is faster than Apple's interpolator

- Apple's `VTLowLatencyFrameInterpolation` is a learned model that produces a full-quality image. It needs its own 8-bit copy of each frame and takes about 15 ms offline and 22 ms p50 live on an M2.
- Motion does a hardware motion search (about 2.4 ms), a few lightweight Metal passes (about 5 ms) and one warp. It reads the decoded frames in place, with no conversion copy.

The live median is **9.0 ms vs 22.4 ms (about 2.5× faster)**. At 60 FPS the generated frame must be ready within about 16 ms. Apple misses that deadline more often (922 late frames in 3 minutes vs 1 for Motion), and every late frame is a missed refresh.

## Hardware comparison (MacBook Air M2, 1080p)

| | Apple FrameGen | Motion FrameGen |
|---|---|---|
| Method | Learned interpolation model (VideoToolbox) | Block motion estimation, then Metal vector selection and warp |
| Media / video engine | HEVC decode | HEVC decode + motion estimation (VideoToolbox) |
| Neural Engine | Not documented by Apple | Not used |
| GPU (Metal) | Static-HUD preserve pass, about 0.25 ms | Guide, select, smooth and warp passes, about 5 to 7 ms |
| CPU | Frame copy into Apple's 8-bit layout | Camera-motion median on a sampled field (well under 1 ms) |
| Generated frame time, live p50 / p95 | 22.4 / 26.0 ms | 9.0 / 11.7 ms |
| Late frames in a 3-minute run | 922 | 1 |
| Missed 60 Hz slots | 9.8 % | 1.8 % |
| Arrival → on screen, p50 / p95 | 76 / 96 ms | 67 / 84 ms |
| Overall quality (real footage, Y-PSNR) | 28.2 dB | 27.9 dB |
| Thin edges (edge PSNR) | **26.6 dB** | 24.8 dB |
| Visual trade-off | Clean edges, softer motion, steady HUD | Smoother and snappier, but tears on fast thin objects (signs, poles, power lines) |

Live numbers are from 3-minute 1080p sessions on a MacBook Air M2. Quality numbers come from `motion-bench` on a real 1080p30 HDR gameplay recording (see [MOTION_PROGRESS.md](MOTION_PROGRESS.md)).

## Tuning switches

These are environment variables for A/B benchmarks. The defaults are what ships.

| Variable | Default | Meaning |
|---|---|---|
| `MFG_BLOCK` | 16 | VT block size on the guide: 16 or 4. 4×4 is sharper but costs 15 ms. |
| `MFG_MULTIPASS` | 1 | VT multi-pass motion search. |
| `MFG_SAD9` | 2 | Match window: 0 = 5-tap cross, 1 = 3×3, 2 = 5×5 (2 px spacing). |
| `MFG_LAMBDA` | 0.0004 | Prior toward the local block field. |
| `MFG_SMOOTH` | 0 | Guarded 3×3 vector median (no measurable effect since v0.3). |
| `MFG_SPLAT` | 1 | Anchored fields splatted to the midpoint (v0.3). |
| `MFG_GRID` | 4 | Anchored field grid in guide pixels (1-4). |
| `MFG_TSEED` | 1 | Previous pair's fields as candidates. |
| `MFG_VT2` | 1 | Both VideoToolbox directions; 0 reuses one, negated (about 1 ms faster, 0.2 dB lower). |
| `MFG_PRING` | 1 | Context ring in propagation. |
| `MFG_BORDER` | 1 | Frame-border handling (v0.4). |
| `MFG_HUD` | 1 | Static-HUD mask (v0.4, about 0.3 ms). |
| `MFG_REFINE` | 0 | Propagation passes. |
| `MFG_FB0`, `MFG_FB1` | 0.03, 0.08 | Match-cost range over which a pixel fades to a 50/50 blend. |
| `MFG_FUSE` | 1 | Fused upsample + warp (v0.5); 0 = the staged path, same output. |
| `MFG_V04` | 0 | 1 = none of the v0.5 quality changes (the v0.4 output, for A/B). |
| `MFG_P2` | 85 | v0.5 quality changes as a bit mask (1 + 4 + 16 + 64). The per-feature switches below override single bits. |
| `MFG_OCC` | 1 | Bit 1: one-sided (occlusion) sampling only when proven. `MFG_OCCE` 0.04 / `MFG_OCCM` 0.02 set the proof. |
| `MFG_UPZ` | 0 | Bit 2: fade cost interpolated between cells (`MFG_UPZM` 1 / 2 / 3). Off: blurs object edges on the synthetic bench. |
| `MFG_UPSIDE` | 1 | Bit 4: bilinear vector candidate in the upsample, one-sided cells reconsidered per pixel. |
| `MFG_MARGIN` | 0 | Bit 8: unique-match margin for splat overrides. Off: measured worse (see MOTION_PROGRESS v0.5). |
| `MFG_SUBPIX` | 1 | Bit 16: parabolic sub-pixel fit (`MFG_SUBC` 0.04: winners under this cost only). |
| `MFG_HOLE` | 1 | Bit 64: 1-ring pull for midpoint cells over `MFG_HOLEC` (0.08). |
| `MFG_P3` | 7 | Pass 3 changes as a bit mask (on by default since v0.6: about +2.4 ms p50 / +2.7 ms p95 over the Pass 2 picture). The per-feature switches below override single bits. |
| `MFG_SOFT` | 1 | Bit 1: soft upsample. `MFG_SOFTS` 0.12 (cost scale), `MFG_SOFTM` 2 / `MFG_SOFTMR` 0.3 (merge radius: px / share of the vector length), `MFG_SOFTTOP` 3 (entries warped, 2 or 3), `MFG_SOFTKMIN` 0 (weight under which an entry is dropped), `MFG_SOFTH` 0.004 (unchanged-pixel rule). `MFG_SOFTV2=0` runs the v0.5 loop (`MFG_SOFTK` 1: 3×3 tent, 0 = 2×2 bilinear), about 2 ms slower. |
| `MFG_GLYPH` | 1 | Bit 2: pair-local subtitle guard (v0.6; replaces v0.5's `MFG_LOCK`). `MFG_GBAND` 0.70,0.98 (rows, share of the height), `MFG_GSTROKE` 0.035, `MFG_GBRIGHT` 0.45, `MFG_GCENTER` 0.012, `MFG_GSHAPE` 0.025, `MFG_GWORLD` 0.02, `MFG_GMOVE` 2, `MFG_GBINS` 3, `MFG_GONSET` 1, `MFG_GGROW` 1, `MFG_GFINE` 0 (anti-copy per luma px). |
| `MFG_EDGE` | 1 | Bit 4: frame-edge fade and distance weighting. |
| `MFG_SHADOW` | 0 | Bit 8: shadow-over-road zero-mean retry (v0.6; replaces v0.5's `MFG_CMATCH`). `MFG_SHCORR` 0.75, `MFG_SHEZ` 0.012, `MFG_SHMARGIN` 0.004, `MFG_SHMD` 0.015, `MFG_SHTEX` 0.012, `MFG_SHFB` 1. Neutral on the benches. |
| `MFG_HUDPX` | 0 | Bit 32: static-HUD cells offer their moving vector per pixel. |
| `MFG_P3DBG` | 0 | Debug views of Pass 3 (1 = glyph mask, 2 = soft-path pixels, 4 = anti-copy rejections). |
| `MFG_TG` | | Threadgroup shape per kernel for sweeps, e.g. `mfg_anchor=8x8,mfg_upwarp=32x4`. |
| `MFG_VERBOSE` | 0 | 1 = print shader compile errors. |
| `FG_DEBUG_VIEW` | 0 | 1 = occlusion map, 2 = horizontal vector, written into the luma plane. |

## Benchmarks

```bash
# Real recording (any 1080p30 or lower video): frames i and i+2 make a midpoint, scored against the real frame i+1
swift run -c release motion-bench clip.mp4 --count 300 [--no-apple] [--start S]
```

The 4x (120 FPS) and recording benches live in the FramePortal app and are not in this repo.

The engine is a single file: [`Sources/Motion/MotionFrameGen.swift`](Sources/Motion/MotionFrameGen.swift).
