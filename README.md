# Motion

Motion is a real-time frame-generation engine for Apple silicon. It turns a 30 FPS video or game stream into smooth 60 FPS by drawing a new frame between every two real ones, fast enough for live play.

It runs on the GPU (Metal) and the video hardware's motion estimator (VideoToolbox). It works on the decoded YCbCr frames as they are, 8-bit or 10-bit HDR, with no RGB conversion and no Neural Engine. It is one Swift file with no dependencies outside Apple's frameworks.

**Current version: v0.6-beta.** It was built for low-latency PS5 Remote Play streaming in the FramePortal app.

## How it compares

Apple ships its own interpolator, `VTLowLatencyFrameInterpolation`. On a 1080p30 HDR gameplay recording (GTA VI), MacBook Air M2:

| | Motion v0.6-beta | Apple |
|---|---|---|
| Time to make one frame (median / 95th percentile) | **13.3 / 15.4 ms** | 14.8 / 16.7 ms |
| Overall picture quality (Y-PSNR) | **29.39 dB** | 28.17 dB |
| Thin edges: poles, signs, outlines (edge PSNR) | **26.68 dB** | 26.61 dB |
| Steadiness, frame to frame (higher = less shimmer) | **25.81 dB** | 24.72 dB |

The test is harsh on purpose: frames 1 and 3 are used to rebuild frame 2, twice the motion of a live 30 → 60 step. A plain 50/50 blend scores 26.17 / 24.13 dB.

Live, at 60 FPS in a game stream, Motion's lower cost means far fewer frames miss the display's 16.7 ms deadline:

| Live, 3 minutes | Missed 60 Hz frames | Generated frames that arrived late |
|---|---|---|
| Motion | 1.8 % | 1 |
| Apple | 9.8 % | 922 |

These live numbers are from an earlier Motion version (v0.1-beta).

**What it does well:** camera pans, HUD text and icons that stay still, screen edges, thin power lines and car outlines.
**New in v0.6:** the soft upsample, frame-edge fade and a pair-local subtitle guard are on by default and cheaper than v0.5's optional cleanup. Fast shadows no longer break into blocks and moving subtitles stay whole. Set `MFG_P3=0` for the faster v0.5 output (about 2.4 ms less per frame).

**Experimental 120 FPS:** on a 120 Hz display, `interpolate4(prev:cur:)` makes three frames per real pair (t = 0.25, 0.5, 0.75), about 21 ms per pair at 1080p on an M2. The 60 FPS output is unchanged.

**Still in progress:**
- see-through layers over fast motion (a car's shadow on a moving road, a translucent minimap);
- rows of identical objects;
- very thin, fast poles.

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
motion.interpolate(prev: prev, cur: cur) { made, error in
    // made: the frame halfway between prev and cur, in the same pixel format.
    // nil with no error: the pair was unreliable (a scene cut, for example). Show the real frame only.
    // Show `made` 16.7 ms before `cur`.
}

motion.end()
```

The completion runs on a Metal callback queue. Keep `prev` and `cur` alive until it fires.

## Benchmark it yourself

```bash
swift run -c release motion-bench your-clip.mp4 --count 300
```

This works with any 30 FPS recording up to 1080p. Add `--no-apple` to time Motion only. Tuning switches for A/B tests are environment variables, listed in [MOTION.md](MOTION.md#tuning-switches).

## License

MIT. See [LICENSE](LICENSE).
