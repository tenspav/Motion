import Foundation
import AVFoundation
import CoreVideo
import QuartzCore
import VideoToolbox
import Motion

// motion-bench: Motion vs Apple's VTLowLatencyFrameInterpolation on a real 30 FPS recording.
//
// Skip-one test: frames i and i+2 make a midpoint, which is scored against the real frame i+1. The gap is twice a
// live 30 -> 60 step, so it is a harsh test with the same failure modes.
//   Y-PSNR      whole picture (luma, video range)
//   edge PSNR   strong-gradient pixels only: thin lines, poles, signs, silhouettes
//   steady      PSNR of the frame-to-frame change of the error (low = shimmer / wobble)
//
// usage: motion-bench CLIP.mp4 [--count N] [--start S] [--no-apple]

let args = Array(CommandLine.arguments.dropFirst())
func opt(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }

guard #available(macOS 26.0, *) else { print("needs macOS 26"); exit(1) }
guard let path = args.first, !path.hasPrefix("--") else {
    print("usage: motion-bench CLIP.mp4 [--count N] [--start S] [--no-apple]")
    exit(1)
}
let count = Int(opt("--count") ?? "300") ?? 300
let start = Double(opt("--start") ?? "0") ?? 0
let useApple = !args.contains("--no-apple")

/// Blocks until a completion-style call returns.
func wait<T>(_ body: (@escaping @Sendable (T?, Error?) -> Void) -> Void) -> T? {
    let sem = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var out: T?
    body { v, _ in out = v; sem.signal() }
    sem.wait()
    return out
}

func percentile(_ v: [Double], _ p: Double) -> Double {
    guard !v.isEmpty else { return 0 }
    let s = v.sorted()
    return s[min(s.count - 1, Int(Double(s.count - 1) * p / 100 + 0.5))]
}

/// Luma plane as 0...1 floats (8- or 10-bit, video range).
func luma(_ b: CVPixelBuffer) -> [Float] {
    CVPixelBufferLockBaseAddress(b, .readOnly); defer { CVPixelBufferUnlockBaseAddress(b, .readOnly) }
    let w = CVPixelBufferGetWidth(b), h = CVPixelBufferGetHeight(b)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(b, 0), base = CVPixelBufferGetBaseAddressOfPlane(b, 0)!
    let f = CVPixelBufferGetPixelFormatType(b)
    let ten = f == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || f == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    let full = f == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || f == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    var out = [Float](repeating: 0, count: w * h)
    for y in 0..<h {
        let row = base.advanced(by: y * stride)
        for x in 0..<w {
            var v = ten ? Float(row.assumingMemoryBound(to: UInt16.self)[x] >> 6) / 1023
                        : Float(row.assumingMemoryBound(to: UInt8.self)[x]) / 255
            if full { v = 16.0 / 255 + v * 219 / 255 }
            out[y * w + x] = v
        }
    }
    return out
}

/// Strong luma gradient in the true frame.
func edges(_ y: [Float], _ w: Int, _ h: Int) -> [Bool] {
    var e = [Bool](repeating: false, count: y.count)
    for r in 1..<(h - 1) {
        for c in 1..<(w - 1) {
            let k = r * w + c
            let gx = y[k + 1] - y[k - 1], gy = y[k + w] - y[k - w]
            e[k] = gx * gx + gy * gy > 0.06 * 0.06
        }
    }
    return e
}

struct Score {
    var se = 0.0, n = 0, ese = 0.0, en = 0
    var prevErr: [Float]?, jse = 0.0, jn = 0
    mutating func add(_ got: [Float], _ truth: [Float], _ edge: [Bool]) {
        var err = [Float](repeating: 0, count: truth.count)
        for k in 0..<truth.count {
            let d = got[k] - truth[k]
            err[k] = d
            se += Double(d * d); n += 1
            if edge[k] { ese += Double(d * d); en += 1 }
            if let p = prevErr { let j = Double(d - p[k]); jse += j * j; jn += 1 }
        }
        prevErr = err
    }
    static func db(_ e: Double, _ n: Int) -> Double { 10 * log10(1 / max(e / Double(max(n, 1)), 1e-12)) }
    var line: String { String(format: "%6.2f dB  %6.2f dB  %6.2f dB", Self.db(se, n), Self.db(ese, en), Self.db(jse, jn)) }
}

/// Apple's low-latency interpolator (8-bit input in its own layout; frames are copied into it).
@available(macOS 26.0, *)
final class AppleInterpolator: @unchecked Sendable {
    private let processor = VTFrameProcessor()
    private let config: VTLowLatencyFrameInterpolationConfiguration
    private var srcPool: CVPixelBufferPool?, dstPool: CVPixelBufferPool?
    private var transfer: VTPixelTransferSession?

    init?(width: Int, height: Int) {
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported,
              let c = VTLowLatencyFrameInterpolationConfiguration(frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1)
        else { return nil }
        config = c
        do { try processor.startSession(configuration: c) } catch { return nil }
        CVPixelBufferPoolCreate(nil, nil, c.sourcePixelBufferAttributes as CFDictionary, &srcPool)
        CVPixelBufferPoolCreate(nil, nil, c.destinationPixelBufferAttributes as CFDictionary, &dstPool)
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
    }

    private func copy(_ b: CVPixelBuffer) -> CVPixelBuffer? {
        guard let srcPool, let transfer else { return nil }
        var o: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, srcPool, &o)
        guard let o, VTPixelTransferSessionTransferImage(transfer, from: b, to: o) == noErr else { return nil }
        return o
    }

    /// Converts both frames first (as a live pipeline would), then times only the interpolation.
    func midpoint(_ a: CVPixelBuffer, _ b: CVPixelBuffer, pts: Int64) -> (CVPixelBuffer?, Double) {
        guard let ca = copy(a), let cb = copy(b), let dstPool else { return (nil, 0) }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, dstPool, &out)
        let t = { (us: Int64) in CMTime(value: us, timescale: 1_000_000) }
        guard let out,
              let p = VTFrameProcessorFrame(buffer: ca, presentationTimeStamp: t(pts)),
              let c = VTFrameProcessorFrame(buffer: cb, presentationTimeStamp: t(pts + 66_666)),
              let d = VTFrameProcessorFrame(buffer: out, presentationTimeStamp: t(pts + 33_333)),
              let params = VTLowLatencyFrameInterpolationParameters(sourceFrame: c, previousFrame: p, interpolationPhase: [0.5], destinationFrames: [d])
        else { return (nil, 0) }
        let t0 = CACurrentMediaTime()
        let r: CVPixelBuffer? = wait { (done: @escaping @Sendable (CVPixelBuffer?, Error?) -> Void) in
            processor.process(parameters: params) { _, err in done(err == nil ? out : nil, err) }
        }
        return (r, (CACurrentMediaTime() - t0) * 1000)
    }

    func end() { processor.endSession() }
}

// MARK: - run

let asset = AVURLAsset(url: URL(fileURLWithPath: path))
guard let track = try await asset.loadTracks(withMediaType: .video).first, let reader = try? AVAssetReader(asset: asset) else {
    print("cannot open \(path)"); exit(1)
}
reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600), duration: .positiveInfinity)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    kCVPixelBufferMetalCompatibilityKey as String: true])
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else { print("reader failed"); exit(1) }
func nextFrame() -> CVPixelBuffer? { output.copyNextSampleBuffer().flatMap { CMSampleBufferGetImageBuffer($0) } }

guard let first = nextFrame() else { print("no frames"); exit(1) }
let W = CVPixelBufferGetWidth(first), H = CVPixelBufferGetHeight(first)
guard let motion = MotionInterpolator(width: W, height: H) else { print("Motion unavailable on this device"); exit(1) }
try motion.start()
let apple = useApple ? AppleInterpolator(width: W, height: H) : nil
if useApple && apple == nil { print("Apple interpolator unavailable (it supports up to 1080p); running Motion only") }

var window: [CVPixelBuffer] = [first]
while window.count < 3, let f = nextFrame() { window.append(f) }
var sBlend = Score(), sMotion = Score(), sApple = Score()
var msMotion: [Double] = [], msApple: [Double] = []
var i = 0
while window.count == 3 && i < count {
    let (a, t, b) = (window[0], window[1], window[2])
    let truth = luma(t), la = luma(a), lb = luma(b)
    let edge = edges(truth, W, H)
    let blend = zip(la, lb).map { 0.5 * ($0 + $1) }
    sBlend.add(blend, truth, edge)

    let t0 = CACurrentMediaTime()
    let made: CVPixelBuffer? = wait { (done: @escaping @Sendable (CVPixelBuffer?, Error?) -> Void) in
        motion.interpolate(prev: a, cur: b, completion: done)
    }
    msMotion.append((CACurrentMediaTime() - t0) * 1000)
    sMotion.add(made.map(luma) ?? blend, truth, edge)       // a skipped pair counts as the plain blend

    if let apple {
        let (ao, ms) = apple.midpoint(a, b, pts: Int64(i) * 66_666)
        if let ao { msApple.append(ms); sApple.add(luma(ao), truth, edge) }
    }
    window.removeFirst()
    if let f = nextFrame() { window.append(f) }
    i += 1
}
motion.end(); apple?.end()

print("=== \(URL(fileURLWithPath: path).lastPathComponent): \(i) skip-one midpoints, \(W)x\(H), Motion \(MotionInterpolator.engineVersion) ===")
print("               Y-PSNR    edge PSNR   steady      p50 ms   p95 ms")
func row(_ name: String, _ s: Score, _ ms: [Double]) {
    let t = ms.isEmpty ? "      -        -" : String(format: "%7.2f  %7.2f", percentile(ms, 50), percentile(ms, 95))
    print(name.padding(toLength: 13, withPad: " ", startingAt: 0) + s.line + "  " + t)
}
row("50/50 blend", sBlend, [])
row("Motion", sMotion, msMotion)
if apple != nil { row("Apple", sApple, msApple) }
