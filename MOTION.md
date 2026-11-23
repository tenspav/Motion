# Motion FrameGen

Motion is a real-time frame-generation engine for Apple silicon. It turns a 30 FPS game stream into 60 FPS by building one new frame between every two real frames. It is an alternative to Apple's built-in interpolator (`VTLowLatencyFrameInterpolation`), and it is currently in **beta**. It was built for low-latency PS5 Remote Play streaming and runs in the FramePortal app.

**Current version: Motion v0.4-beta.** Motion is versioned separately from the app. It becomes v1.0, and the default engine, once it is visually as clean as Apple's.

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

Per-stage GPU time on an M2 (1080p):

| Stage | Time |
|---|---|
| Anchor + splat | about 4.0 ms |
| Select | 1.3 ms |
| Upsample | 1.3-1.7 ms |
| Warp | 0.75 ms |
| Guide | 0.5 ms |
| VideoToolbox motion (CPU wait) | 2.35 ms |

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
| `FG_DEBUG_VIEW` | 0 | 1 = occlusion map, 2 = horizontal vector, written into the luma plane. |

## Benchmarks

```bash
# Real recording (any 1080p30 or lower video): frames i and i+2 make a midpoint, scored against the real frame i+1
swift run -c release motion-bench clip.mp4 --count 300 [--no-apple] [--start S]
```

The engine is a single file: [`Sources/Motion/MotionFrameGen.swift`](Sources/Motion/MotionFrameGen.swift).
