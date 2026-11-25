<p align="center">
  <img src="assets/icon.svg" width="160" alt="Motion icon">
</p>

<h1 align="center">Motion</h1>

<p align="center">
  <b>Real-time 30 → 60 FPS frame generation for Apple silicon, fast enough for live play.</b><br>
  One Swift file: VideoToolbox motion search + Metal shaders, on the decoded YCbCr frames as they are.
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-v0.8--beta-yellow">
  <img alt="platform" src="https://img.shields.io/badge/platform-macOS%2026%20%7C%20iOS%2026-lightgrey">
  <img alt="engine" src="https://img.shields.io/badge/engine-Metal%20%2B%20VideoToolbox-orange">
  <img alt="neural engine" src="https://img.shields.io/badge/Neural%20Engine-0%25-blueviolet">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## Motion vs Apple (MacBook Air M2)

Apple ships its own interpolator, `VTLowLatencyFrameInterpolation`. On a 1080p30 HDR gameplay recording (GTA VI):

| | Motion v0.8-beta | Apple |
|---|---|---|
| ⚡ Time to make one frame (p50 / p95) | **12.7 / 15.4 ms**\* | 14.8 / 16.7 ms |
| 🖼️ Overall picture quality (Y-PSNR) | **29.43 dB** | 28.17 dB |
| 📏 Thin edges: poles, signs, outlines | **26.72 dB** | 26.61 dB |
| 🎞️ Steadiness, frame to frame | **25.81 dB** | 24.72 dB |
| 🧠 Neural Engine | **0 %** | — |
| 🎨 Formats | 8-bit and 10-bit HDR YCbCr, no RGB round trip | — |

The test is harsh on purpose: frames 1 and 3 rebuild frame 2, twice the motion of a live 30 → 60 step (a plain 50/50 blend scores 26.17 / 24.13 dB).
\* Fade guard on, mean of 4 interleaved rounds of 300 frames; v0.7-beta measured 13.3 / 16.2 ms in the same rounds. Apple's numbers are from an earlier run on the same clip.

| Live, 3 minutes at 60 FPS | Missed 60 Hz frames | Generated frames that arrived late |
|---|---|---|
| Motion | **1.8 %** | **1** |
| Apple | 9.8 % | 922 |

(Live numbers from an earlier Motion version, v0.1-beta.)

## What's new in v0.8-beta

> **Fade and exposure guard:** while the motion search runs, Motion compares the two frames' brightness in 16×9 zones (about 0.55 ms on the CPU, off the critical path). On a fade it makes a clean cross-fade, or redoes the search on brightness-matched guides. Pairs without a fade give exactly the same output as before.

| Scene | v0.7-beta | v0.8-beta |
|---|---|---|
| 🚇 Tunnel exit (fade from grey, rising exposure) | 36.16 dB | **40.22 dB** (+4.1) |
| 🗺️ Closing a full-screen map | 36.83 dB | **41.93 dB** (+5.1) |
| 📍 Static map, dark-speck pixels per frame | up to 12,060 | **0** (12 of 12 pairs) |

> **Fewer GPU drains, same pixels:** clears, the subtitle pre-test and the first search scores now overlap the hardware motion search; HUD reads are skipped when nothing is masked; two kernels are reordered. Output is bit-identical to v0.7, **0.6 to 1.1 ms faster** at p50.

| Where the time goes (GPU, 1080p, M2) | ms |
|---|---|
| Anchored search + 3 propagation steps | 1.35 + 3 × 0.65 |
| Per-pixel selection (`mfg_select`) | 1.44 |
| Subtitle guard | 0.30 |
| Final grid + grid fix | 0.14 |
| Fused upsample + warp (`mfg_upwarp`) | 4.4 to 4.7 |
| Waiting on the hardware motion search | ~2.3 |

## Milestones

| Version | Highlights |
|---|---|
| **v0.8-beta** | Fade / exposure guard, fewer GPU drains, 120 FPS phase split (first frame 14.3 vs 24.8 ms p50, off by default) |
| v0.7-beta | Never-blocking `trySubmit`, **clean repeat at hard cuts** (no ghost of two scenes), side-proof **subtitle fix**, canonical 10-bit `x420` |
| v0.6-beta | Soft upsample, frame-edge fade and subtitle guard on by default; experimental 120 FPS (3 frames per pair, ~25 ms at 1080p) |
| v0.5-beta | Faster pipeline, proven occlusion |
| v0.4-beta | Clean frame borders, static HUD locked in place |

**Still in progress:** halos around fast cars, slow exposure ramps with motion, see-through layers over fast motion (a car's shadow, a translucent minimap), rows of identical objects, very thin fast poles.

## How it works

1. **Find the motion.** The hardware motion estimator measures how blocks of the picture moved between the two frames, in both directions.
2. **Refine it per pixel.** Metal shaders build dense motion fields, check them forwards and backwards, and keep objects on their own motion. Static HUD elements are learned over time and locked in place.
3. **Draw the middle frame.** Both real frames are warped halfway and blended. Areas only visible in one frame use that frame. Uncertain pixels fade to a soft blend instead of tearing.

The full shader-by-shader design is in [MOTION.md](MOTION.md). Every experiment, with its numbers, is in [MOTION_PROGRESS.md](MOTION_PROGRESS.md).

## Requirements

- Apple silicon Mac with macOS 26, or iPhone / iPad with iOS 26
- Swift 6 toolchain (Xcode 26)
- Frames as bi-planar YCbCr `CVPixelBuffer`s (`420v`, `420f`, `x420`), IOSurface-backed and Metal-compatible. Video decoders give you these.

## Use it

Add the package (`https://github.com/tenspav/Motion`), or copy [`Sources/Motion/MotionFrameGen.swift`](Sources/Motion/MotionFrameGen.swift) into your project.

```swift
import Motion

guard let motion = MotionInterpolator(width: 1920, height: 1080) else { return }   // nil: no Metal device
try motion.start()

// For each new decoded frame `cur`, with the previous one in `prev`:
let admission = motion.trySubmit(prev: prev, cur: cur) { made, error in
    // made: the frame halfway between prev and cur, in the same pixel format.
    // At a hard scene cut it is a clean copy of prev (see motion.lastOutcome).
    // Show it halfway between the two real frames' times.
}
if admission == .busy {
    // The previous pair is still running: show the real frame only and try again with the next one.
}

motion.end()
```

`interpolate(prev:cur:completion:)` does the same but reports busy as an error. The completion runs on a Metal callback queue. Keep `prev` and `cur` alive until it fires.

## Benchmark it yourself

```bash
swift run -c release motion-bench your-clip.mp4 --count 300
```

This works with any 30 FPS recording up to 1080p. Add `--no-apple` to time Motion only. Tuning switches for A/B tests are environment variables, listed in [MOTION.md](MOTION.md#tuning-switches).

## Datasets & Credits

Motion is hand-designed: classic motion estimation plus Metal shaders. Its quality and speed numbers come from recordings of PS5 gameplay streams captured for testing (the "GTA VI" clip and a few other game recordings) and from a synthetic test scene generated in code. None of this footage is included in this repository, and no third-party dataset was used for the numbers above. Game footage belongs to its publishers.

The motion estimation itself comes from Apple's VideoToolbox motion estimator; everything else is Metal and Swift in this repository.

This section is kept up to date whenever footage used for benchmarks is added or changed.

## License

MIT. See [LICENSE](LICENSE).
