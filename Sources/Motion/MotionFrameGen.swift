import Foundation
import CoreVideo
import VideoToolbox
import Metal
import QuartzCore

/// Motion: real-time 30 -> 60 FPS frame generation on Apple silicon (an alternative to VTLowLatencyFrameInterpolation):
///   half-res 8-bit luma guides (Metal) -> VTMotionEstimationSession 16×16 multi-pass, both directions (concurrent)
///   -> per-pixel candidate selection at guide resolution (19 candidates: 3×3 blocks of each field + zero, 5-tap SAD
///      on the guides, zero-vector bias = static HUD stays put) -> full-res warp of the decoded YCbCr planes
///      (10-bit x420 kept, no RGB round trip): both endpoints blended where they agree, one side only where they don't
///      (occlusion: no double images). A pair with too many unreliable pixels is skipped (real frame only).
/// Vector convention (measured 2026-11-23, synthetic pan): motion(of: B, comparedTo: A) = v in guide pixels with
/// B(p) ≈ A(p + v); a static HUD box reads 0.
@available(macOS 26.0, iOS 26.0, *)
public final class MotionInterpolator: @unchecked Sendable {
    /// Frame size in pixels (this file is self-contained: no dependency outside Apple's frameworks).
    public struct Size: Equatable, Sendable, CustomStringConvertible {
        public var width: Int, height: Int
        public init(_ width: Int, _ height: Int) { self.width = width; self.height = height }
        public var description: String { "\(width)x\(height)" }
    }

    /// Engine revision (bench header, A/B notes).
    public static let engineVersion = "0.5-beta"
    let size: Size
    let guide: Size
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let psoGuide, psoNeg, psoSelect, psoAnchor, psoAnchorProp, psoSplat, psoUpsample, psoSmooth, psoRefine, psoWarpY, psoWarpC, psoEma, psoUpwarp, psoGridFix, psoLock, psoLockDil: MTLComputePipelineState
    private var cache: CVMetalTextureCache?
    private let session: VTMotionEstimationSession
    private var guidePool: CVPixelBufferPool?
    private var outPool: CVPixelBufferPool?
    private var outFormat: OSType = 0
    private let midTex: MTLTexture
    /// Vector field after the 3×3 median clean-up (what the warp reads).
    private let smoothTex: MTLTexture
    /// Dense anchored fields A -> B, B -> A (+ propagation scratch) and their midpoint splat buffers.
    private let fTmp: MTLTexture, fTmpB: MTLTexture, gridMid: MTLTexture, gridFix: MTLTexture
    /// Per guide px: static overlay (text that is the same in both frames over a moving picture), P3_LOCK.
    private let lockTex: MTLTexture, lockDil: MTLTexture, lockBits: MTLTexture
    /// P3_HUDPX: per midpoint cell, the moving vector a static-HUD cell gave up (flag in .w).
    private let gridAlt: MTLTexture
    /// Two sets of anchored fields, alternating per pair: the previous pair's set seeds this pair's search.
    private let fAs: [MTLTexture], fBs: [MTLTexture]
    private var par = 0, hasPrev = false
    private var fA: MTLTexture { fAs[par] }
    private var fB: MTLTexture { fBs[par] }
    private let gridSize: Size
    private let splatA: MTLBuffer, splatB: MTLBuffer
    /// Static HUD state per selection cell (float2: evidence level, mask on), two buffers alternating per pair.
    private let hudS: [MTLBuffer]
    /// Long-term means per guide px (float2): the guide, and its gradient magnitude. A static edge keeps its gradient
    /// in the mean of the guide; moving texture averages out of it but not out of the mean gradient.
    private let emaBuf: MTLBuffer
    private var hudPar = 0, hudClear = true, hudPairs = 0
    private let badCount: MTLBuffer
    private var last: (pb: CVPixelBuffer, guide: CVPixelBuffer)?
    private(set) var started = false
    private var pairs = 0
    /// Pixels (fraction) above the photometric threshold that make a pair be skipped.
    static let rejectFraction: Float = 0.30

    /// Bench-only phase timing (guide ms, motion ms).
    nonisolated(unsafe) static var trace = false
    nonisolated(unsafe) static var phase: [(Double, Double)] = []
    nonisolated(unsafe) static var gpu: [Double] = []
    nonisolated(unsafe) static var profile = ProcessInfo.processInfo.environment["MFG_PROFILE"] == "1"
    nonisolated(unsafe) static var stageMs: [String: [Double]] = [:]
    /// Bench-only: 1 = occlusion side map, 2 = chosen x vector, in the luma output.
    nonisolated(unsafe) static var debugView: Int32 = 0
    /// Motion block size on the half-res guide: 16×16 (32×32 full-res) or 4×4 (8×8 full-res). VideoToolbox offers
    /// no 8×8. Env MFG_BLOCK=4 / 16 overrides (bench A/B).
    nonisolated(unsafe) static var smallBlocks = ProcessInfo.processInfo.environment["MFG_BLOCK"].map { $0 == "4" } ?? false
    /// 3×3 vector median on the selected field before the warp (MFG_SMOOTH=0 turns it off for A/B).
    /// Candidate block radius (1 = 3×3 blocks, 2 = 5×5) and match window (0 = 5-tap cross, 1 = 3×3 at 2 px,
    /// 2 = 5×5 at 2 px: default, measured best on real GTA VI footage).
    nonisolated(unsafe) static var candRadius = Int32(ProcessInfo.processInfo.environment["MFG_NB"] ?? (splat ? "0" : "1")) ?? 1
    /// Smoothness prior per guide px of distance from the local block field (MFG_LAMBDA, bench A/B).
    nonisolated(unsafe) static var lambda = Float(ProcessInfo.processInfo.environment["MFG_LAMBDA"] ?? "0.0004") ?? 0.0004
    /// Propagation passes after the selection (MFG_REFINE): each pixel also tries its neighbours' chosen vectors and
    /// ±1 px around its own, so thin objects (poles, lines, sign edges) pick up the vector of the object they belong to.
    nonisolated(unsafe) static var refinePasses = Int(ProcessInfo.processInfo.environment["MFG_REFINE"] ?? "0") ?? 0
    /// Low-confidence pixels (match cost from .x to .y) fade to a plain 50/50 blend: soft ghost instead of a tear.
    nonisolated(unsafe) static var fallback: SIMD2<Float> = {
        let e = ProcessInfo.processInfo.environment
        return SIMD2(Float(e["MFG_FB0"] ?? "0.03") ?? 0.03, Float(e["MFG_FB1"] ?? "0.08") ?? 0.08)
    }()
    nonisolated(unsafe) static var wideSAD = Int32(ProcessInfo.processInfo.environment["MFG_SAD9"] ?? "2") ?? 2
    /// Multi-pass motion search (MFG_MULTIPASS=0 for single pass).
    nonisolated(unsafe) static var multiPass = ProcessInfo.processInfo.environment["MFG_MULTIPASS"] != "0"
    /// Bench: size of the last motion field (cells) and pixels changed by the median.
    nonisolated(unsafe) static var lastField = Size(0, 0)
    nonisolated(unsafe) static var smooth = ProcessInfo.processInfo.environment["MFG_SMOOTH"] == "1"
    /// Splat candidates (MFG_SPLAT=0 off): anchored propagation passes, splat cost cap.
    nonisolated(unsafe) static var splat = ProcessInfo.processInfo.environment["MFG_SPLAT"] != "0"
    nonisolated(unsafe) static var anchorPasses = Int(ProcessInfo.processInfo.environment["MFG_APROP"] ?? "3") ?? 3
    nonisolated(unsafe) static var splatMaxCost = Float(ProcessInfo.processInfo.environment["MFG_SMAX"] ?? "0.05") ?? 0.05
    /// Anchored propagation step per pass, in field cells (MFG_STEPS, comma list; MFG_APROP passes).
    nonisolated(unsafe) static var propSteps: [Int32] = (ProcessInfo.processInfo.environment["MFG_STEPS"] ?? "4,2,1").split(separator: ",").compactMap { Int32($0) }
    /// Guide tone curve: video-range luma to 0...1, then ^MFG_GPOW (< 1 lifts dark detail so shadows still match).
    nonisolated(unsafe) static var guidePow = Float(ProcessInfo.processInfo.environment["MFG_GPOW"] ?? "1") ?? 1
    /// A consistent splat whose midpoint cost is under this wins outright (MFG_SHARD, 0 = off).
    nonisolated(unsafe) static var splatHard = Float(ProcessInfo.processInfo.environment["MFG_SHARD"] ?? "0.06") ?? 0.06
    /// Anchored match window: 3×3 taps MFG_ATAP guide px apart + 8-tap context ring MFG_ATAP2 px out (0 = none);
    /// sub-pixel steps on every propagation pass (MFG_SUBALL=1) or only the step-1 pass.
    nonisolated(unsafe) static var subAll = Int32(ProcessInfo.processInfo.environment["MFG_SUBALL"] ?? "1") ?? 1
    nonisolated(unsafe) static var anchorTap = Float(ProcessInfo.processInfo.environment["MFG_ATAP"] ?? "2") ?? 2
    nonisolated(unsafe) static var anchorTap2 = Float(ProcessInfo.processInfo.environment["MFG_ATAP2"] ?? "14") ?? 14
    /// Both VideoToolbox motion searches (B -> A and A -> B, MFG_VT2=1) or only B -> A, its negation standing in
    /// for A -> B (the anchored A -> B search refines it anyway).
    nonisolated(unsafe) static var bothVT = ProcessInfo.processInfo.environment["MFG_VT2"] != "0"
    /// Seed the anchored search with the previous pair's fields (MFG_TSEED): steadier vectors pair to pair.
    nonisolated(unsafe) static var temporalSeed = ProcessInfo.processInfo.environment["MFG_TSEED"] != "0"
    /// Context ring in the propagation passes as well (MFG_PRING=1) or only in the initial search.
    nonisolated(unsafe) static var propRing = ProcessInfo.processInfo.environment["MFG_PRING"] != "0"
    nonisolated(unsafe) static var oneSide = ProcessInfo.processInfo.environment["MFG_ONESIDE"] == "1"
    /// Frame borders (MFG_BORDER=0 off): match windows skip taps outside the frame instead of clamp-sampling the edge
    /// row, an anchored match may not leave the frame, and a midpoint vector with one endpoint outside (content
    /// scrolling in or out) is judged one-sided and warped from the in-frame side only, without the fallback blend.
    nonisolated(unsafe) static var border = ProcessInfo.processInfo.environment["MFG_BORDER"] != "0"
    /// Static HUD mask (MFG_HUD=0 off): per field cell, a hysteresis state of "a static edge is here" (HUD panel border,
    /// text, icons, the player car outline: edges that survive in the long-term mean of the guide). Masked cells keep the
    /// zero vector unless a good moving match clearly beats it, no splat crosses into or out of them, and vectors whose
    /// path crosses one are penalised: the translucent minimap over a moving road keeps its border and the road stays out.
    nonisolated(unsafe) static var hud = ProcessInfo.processInfo.environment["MFG_HUD"] != "0"
    /// Field cell size in guide px for the anchored fields and the midpoint selection (MFG_GRID, 1 to 4, default 4: the context-ringed
    /// matching window keeps quality equal to grid 2 at a third of the cost); mfg_upsample brings it back to guide px.
    nonisolated(unsafe) static var grid = max(1, min(4, Int(ProcessInfo.processInfo.environment["MFG_GRID"] ?? "4") ?? 4))
    /// v0.5 quality changes, a bit mask (MFG_P2, default 85; MFG_V04=1 = none, the v0.4 output). Bits:
    /// 1 occlusion only when proven (MFG_OCC), 2 bilinear fade cost in the upsample (MFG_UPZ, off: blurs object edges on the synthetic bench), 4 bilinear vector
    /// candidate and one-sided cells reconsidered per pixel (MFG_UPSIDE), 8 unique-match margin for splat overrides
    /// (MFG_MARGIN, off: measured worse), 16 parabolic sub-pixel fit (MFG_SUBPIX), 64 1-ring pull for poor midpoint
    /// cells (MFG_HOLE). A per-feature switch (=0 / =1) overrides its bit.
    nonisolated(unsafe) static var p2: Int32 = {
        let e = ProcessInfo.processInfo.environment
        var m = Int32(e["MFG_P2"] ?? (e["MFG_V04"] == "1" ? "0" : "85")) ?? 85
        for (k, b) in [("MFG_OCC", 1), ("MFG_UPZ", 2), ("MFG_UPSIDE", 4), ("MFG_MARGIN", 8), ("MFG_SUBPIX", 16), ("MFG_HOLE", 64)] as [(String, Int32)] {
            if e[k] == "0" { m &= ~b } else if e[k] == "1" { m |= b }
        }
        return m
    }()
    /// 1-ring pull: midpoint cells with a match cost over this try their neighbours' vectors (MFG_HOLEC).
    nonisolated(unsafe) static var holeCost = Float(ProcessInfo.processInfo.environment["MFG_HOLEC"] ?? "0.08") ?? 0.08
    nonisolated(unsafe) static var subC = Float(ProcessInfo.processInfo.environment["MFG_SUBC"] ?? "0.04") ?? 0.04
    nonisolated(unsafe) static var upZ = Int32(ProcessInfo.processInfo.environment["MFG_UPZM"] ?? "1") ?? 1
    nonisolated(unsafe) static var occE = Float(ProcessInfo.processInfo.environment["MFG_OCCE"] ?? "0.04") ?? 0.04
    nonisolated(unsafe) static var occM = Float(ProcessInfo.processInfo.environment["MFG_OCCM"] ?? "0.02") ?? 0.02
    /// Upsample and warp in one pass (MFG_FUSE=0: the staged upsample -> field texture -> warp path, which the debug
    /// views, MFG_REFINE and MFG_SMOOTH always use). Same output either way.
    nonisolated(unsafe) static var fuse = ProcessInfo.processInfo.environment["MFG_FUSE"] != "0"
    /// v0.5 Pass 3 (the 8-px squares), a bit mask (MFG_P3; MFG_V04=1 or MFG_P3=0 = Pass 2 output). Bits: 1 soft
    /// upsample (MFG_SOFT), 2 per-pixel static-overlay lock (MFG_LOCK), 4 frame-edge fix (MFG_EDGE), 8 contrast-adjusted
    /// correspondence on ambiguous cells (MFG_CMATCH), 16 nearest-cell upsample (MFG_NEAR). Fused path only.
    nonisolated(unsafe) static var p3: Int32 = {
        let e = ProcessInfo.processInfo.environment
        var m = Int32(e["MFG_P3"] ?? "0") ?? 0   // off by default: +4.6 ms p50 / +6 ms p95 (MOTION_PROGRESS v0.5 Pass 3)
        for (k, b) in [("MFG_SOFT", 1), ("MFG_LOCK", 2), ("MFG_EDGE", 4), ("MFG_CMATCH", 8), ("MFG_NEAR", 16), ("MFG_HUDPX", 32)] as [(String, Int32)] {
            if e[k] == "0" { m &= ~b } else if e[k] == "1" { m |= b }
        }
        return m
    }()
    /// Pass 3 tuning (bench A/B): soft-upsample cost scale, lock thresholds (A/B difference, local contrast).
    nonisolated(unsafe) static var p3f: SIMD4<Float> = {
        let e = ProcessInfo.processInfo.environment
        func f(_ k: String, _ d: Float) -> Float { Float(e[k] ?? "") ?? d }
        return SIMD4(f("MFG_SOFTS", 0.06), f("MFG_LOCKD", 0.02), f("MFG_LOCKC", 0.25), f("MFG_CMT", 0.04))
    }()

    static var isSupported: Bool { MTLCreateSystemDefaultDevice() != nil }

    public convenience init?(width: Int, height: Int) { self.init(size: Size(width, height)) }

    public init?(size: Size) {
        guard size.width % 4 == 0, size.height % 4 == 0, size.width <= 3840,
              let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        self.size = size
        guide = Size(size.width / 2, size.height / 2)
        device = dev; queue = q
        do {
            let lib = try dev.makeLibrary(source: Self.shaders, options: nil)
            let fc = MTLFunctionConstantValues()
            var bo = Self.border
            fc.setConstantValue(&bo, type: .bool, index: 0)
            var ws = Self.wideSAD
            fc.setConstantValue(&ws, type: .int, index: 1)
            var t1 = Self.anchorTap, t2 = Self.anchorTap2, pr = Self.propRing, sa = Self.subAll != 0
            fc.setConstantValue(&t1, type: .float, index: 2); fc.setConstantValue(&t2, type: .float, index: 3)
            fc.setConstantValue(&pr, type: .bool, index: 4); fc.setConstantValue(&sa, type: .bool, index: 5)
            var p2 = Self.p2, oe = Self.occE, om = Self.occM
            fc.setConstantValue(&p2, type: .int, index: 6)
            fc.setConstantValue(&oe, type: .float, index: 7); fc.setConstantValue(&om, type: .float, index: 8)
            var uz = Self.upZ; fc.setConstantValue(&uz, type: .int, index: 9)
            var sc = Self.subC; fc.setConstantValue(&sc, type: .float, index: 10)
            var p3 = Self.p3; fc.setConstantValue(&p3, type: .int, index: 11)
            var p3f = Self.p3f
            fc.setConstantValue(&p3f.x, type: .float, index: 12); fc.setConstantValue(&p3f.y, type: .float, index: 13)
            fc.setConstantValue(&p3f.z, type: .float, index: 14); fc.setConstantValue(&p3f.w, type: .float, index: 15)
            var ssd = ProcessInfo.processInfo.environment["MFG_SOFTSIDE"] != "0"; fc.setConstantValue(&ssd, type: .bool, index: 16)
            var p3d = Int32(ProcessInfo.processInfo.environment["MFG_P3DBG"] ?? "0") ?? 0; fc.setConstantValue(&p3d, type: .int, index: 17)
            var sz = Int32(ProcessInfo.processInfo.environment["MFG_SOFTZ"] ?? "0") ?? 0; fc.setConstantValue(&sz, type: .int, index: 18)
            var sk = ProcessInfo.processInfo.environment["MFG_SOFTK"] != "0"; fc.setConstantValue(&sk, type: .bool, index: 19)
            var zs = ProcessInfo.processInfo.environment["MFG_SOFTZS"] != "0"; fc.setConstantValue(&zs, type: .bool, index: 20)
            var sm = Float(ProcessInfo.processInfo.environment["MFG_SOFTM"] ?? "2") ?? 2; fc.setConstantValue(&sm, type: .float, index: 21)
            var scc = ProcessInfo.processInfo.environment["MFG_SOFTC"] == "1"; fc.setConstantValue(&scc, type: .bool, index: 22)
            var lg = Float(ProcessInfo.processInfo.environment["MFG_LOCKG"] ?? "1") ?? 1; fc.setConstantValue(&lg, type: .float, index: 23)
            var lw = Float(ProcessInfo.processInfo.environment["MFG_LOCKW"] ?? "0.15") ?? 0.15; fc.setConstantValue(&lw, type: .float, index: 24)
            var lsr = ProcessInfo.processInfo.environment["MFG_LOCKSRC"] != "0"; fc.setConstantValue(&lsr, type: .bool, index: 25)
            var lb = Float(ProcessInfo.processInfo.environment["MFG_LOCKB"] ?? "0.45") ?? 0.45; fc.setConstantValue(&lb, type: .float, index: 26)
            var lk = Float(ProcessInfo.processInfo.environment["MFG_LOCKK"] ?? "0.2") ?? 0.2; fc.setConstantValue(&lk, type: .float, index: 27)
            var sh = Float(ProcessInfo.processInfo.environment["MFG_SOFTH"] ?? "0.012") ?? 0.012; fc.setConstantValue(&sh, type: .float, index: 28)
            // Threadgroup shape per kernel (bench sweep: MFG_TG="mfg_anchor=16x8,mfg_upwarp=32x4"; default 16×16).
            let tgEnv = (ProcessInfo.processInfo.environment["MFG_TG"] ?? "").split(separator: ",").reduce(into: [String: MTLSize]()) {
                let kv = $1.split(separator: "="), wh = kv.count == 2 ? kv[1].split(separator: "x").compactMap { Int($0) } : []
                if wh.count == 2 { $0[String(kv[0])] = MTLSize(width: wh[0], height: wh[1], depth: 1) }
            }
            var tgs: [ObjectIdentifier: MTLSize] = [:]
            func pso(_ n: String) throws -> MTLComputePipelineState {
                let p = try dev.makeComputePipelineState(function: lib.makeFunction(name: n, constantValues: fc))
                if let t = tgEnv[n] ?? Self.defaultTG[n] { tgs[ObjectIdentifier(p)] = t }
                return p
            }
            psoGuide = try pso("mfg_guide"); psoNeg = try pso("mfg_neg"); psoSelect = try pso("mfg_select"); psoAnchor = try pso("mfg_anchor"); psoUpsample = try pso("mfg_upsample"); psoAnchorProp = try pso("mfg_anchor_prop"); psoSplat = try pso("mfg_splat"); psoSmooth = try pso("mfg_smooth"); psoRefine = try pso("mfg_refine"); psoWarpY = try pso("mfg_warp_y"); psoWarpC = try pso("mfg_warp_c"); psoEma = try pso("mfg_ema"); psoUpwarp = try pso("mfg_upwarp"); psoGridFix = try pso("mfg_gridfix"); psoLock = try pso("mfg_lock"); psoLockDil = try pso("mfg_lockdil")
            session = try VTMotionEstimationSession(width: UInt32(guide.width), height: UInt32(guide.height),
                                                    motionVectorSize: Self.smallBlocks ? .blockSize4x4 : .blockSize16x16, useMultiPassSearch: Self.multiPass, label: "Motion")
            tg = tgs
        } catch {
            if ProcessInfo.processInfo.environment["MFG_VERBOSE"] == "1" { print("Motion init: \(error)") }
            return nil
        }
        let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: guide.width, height: guide.height, mipmapped: false)
        md.usage = [.shaderRead, .shaderWrite]; md.storageMode = .private
        guard let t = dev.makeTexture(descriptor: md), let t2 = dev.makeTexture(descriptor: md),
              let b = dev.makeBuffer(length: 4, options: .storageModeShared) else { return nil }
        midTex = t; smoothTex = t2; badCount = b
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: guide.width, height: guide.height, mipmapped: false)
        ld.usage = [.shaderRead, .shaderWrite]; ld.storageMode = .private
        guard let lt = dev.makeTexture(descriptor: ld), let lt3 = dev.makeTexture(descriptor: ld), let lt4 = dev.makeTexture(descriptor: ld) else { return nil }
        lockTex = lt; lockDil = lt3; lockBits = lt4
        gridSize = Size((guide.width + Self.grid - 1) / Self.grid, (guide.height + Self.grid - 1) / Self.grid)
        let gd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: gridSize.width, height: gridSize.height, mipmapped: false)
        gd.usage = [.shaderRead, .shaderWrite]; gd.storageMode = .private
        guard let a1 = dev.makeTexture(descriptor: gd), let a2 = dev.makeTexture(descriptor: gd), let a3 = dev.makeTexture(descriptor: gd),
              let a4 = dev.makeTexture(descriptor: gd), let a5 = dev.makeTexture(descriptor: gd),
              let a6 = dev.makeTexture(descriptor: gd), let a7 = dev.makeTexture(descriptor: gd), let a8 = dev.makeTexture(descriptor: gd),
              let a9 = dev.makeTexture(descriptor: gd),
              let s1 = dev.makeBuffer(length: gridSize.width * gridSize.height * 4, options: .storageModePrivate),
              let s2 = dev.makeBuffer(length: gridSize.width * gridSize.height * 4, options: .storageModePrivate),
              let h1 = dev.makeBuffer(length: guide.width * guide.height * 8, options: .storageModePrivate),
              let h2 = dev.makeBuffer(length: guide.width * guide.height * 8, options: .storageModePrivate),
              let em = dev.makeBuffer(length: guide.width * guide.height * 8, options: .storageModePrivate) else { return nil }
        fAs = [a1, a6]; fBs = [a2, a7]; fTmp = a3; fTmpB = a5; gridMid = a4; gridFix = a8; gridAlt = a9; splatA = s1; splatB = s2; hudS = [h1, h2]; emaBuf = em
        CVMetalTextureCacheCreate(nil, nil, dev, nil, &cache)
    }

    public func start() throws {
        // 8-bit one-component guides, laid out as the motion session wants them (extended rows), Metal-writable.
        var a = session.sourcePixelBufferAttributes.reduce(into: [String: Any]()) { $0[$1.key] = $1.value }
        a[kCVPixelBufferPixelFormatTypeKey as String] = kCVPixelFormatType_OneComponent8
        a[kCVPixelBufferWidthKey as String] = guide.width
        a[kCVPixelBufferHeightKey as String] = guide.height
        a[kCVPixelBufferMetalCompatibilityKey as String] = true
        a[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, a as CFDictionary, &p)
        guard let p else { throw NSError(domain: "Motion", code: 1, userInfo: [NSLocalizedDescriptionKey: "no guide pool"]) }
        guidePool = p
        started = true
    }

    public func end() { started = false; last = nil; outPool = nil; hasPrev = false; hudClear = true }

    /// Midpoint of (prev, cur) as a new buffer in the decoder's own format (420v / x420), attachments from `cur`.
    /// completion(nil, nil) = pair skipped on purpose (unreliable motion): show the real frame only.
    /// Asynchronous: returns once the guides are queued; the motion search, the GPU passes and the completion follow
    /// on their own. One pair at a time: a call made before the previous pair's completion waits for it.
    public func interpolate(prev: CVPixelBuffer, cur: CVPixelBuffer, completion: @escaping @Sendable (CVPixelBuffer?, Error?) -> Void) {
        let tIn = CACurrentMediaTime()
        inflight.wait()
        let j = Job(prev: prev, cur: cur) { [inflight] pb, err in inflight.signal(); completion(pb, err) }
        let fmt = CVPixelBufferGetPixelFormatType(cur)
        let tenBit = fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        guard started, CVPixelBufferGetWidth(cur) == size.width, CVPixelBufferGetHeight(cur) == size.height,
              CVPixelBufferGetPlaneCount(cur) == 2 else { return j.done(nil, Self.err("unsupported frame")) }
        if outPool == nil || outFormat != fmt { outPool = Self.pool(fmt, size); outFormat = fmt }
        pairs &+= 1
        if pairs % 120 == 0, let c = cache { CVMetalTextureCacheFlush(c, 0) }   // drop stale plane textures (footprint)
        let yFmt: MTLPixelFormat = tenBit ? .r16Unorm : .r8Unorm, cFmt: MTLPixelFormat = tenBit ? .rg16Unorm : .rg8Unorm

        // Guides: reuse prev's when it was the last cur.
        guard let cmd0 = queue.makeCommandBuffer() else { return j.done(nil, Self.err("command buffer")) }
        func makeGuide(_ pb: CVPixelBuffer) -> CVPixelBuffer? {
            var g: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, guidePool!, &g)
            guard let g, let y = tex(j, pb, 0, yFmt), let gt = guideTex(j, g), let e = cmd0.makeComputeCommandEncoder() else { return nil }
            e.setTexture(y, index: 0); e.setTexture(gt, index: 1); var gp = Self.guidePow; e.setBytes(&gp, length: 4, index: 0)
            dispatch(e, psoGuide, guide)
            e.endEncoding()
            return g
        }
        let gA: CVPixelBuffer
        if let l = last, l.pb === prev { gA = l.guide } else if let g = makeGuide(prev) { gA = g } else { return j.done(nil, Self.err("guide")) }
        guard let gB = makeGuide(cur) else { return j.done(nil, Self.err("guide")) }
        last = (cur, gB)
        let tg0 = CACurrentMediaTime()
        if Self.trace { Self.stageMs["cpu.pre", default: []].append((tg0 - tIn) * 1000) }

        // Once the guides are done, both motion searches run (concurrently); the pair continues when they and the
        // frame textures below are ready, on whichever side gets there last.
        let ses = session
        let ra = CVReadOnlyPixelBuffer(unsafeBuffer: gA), rb = CVReadOnlyPixelBuffer(unsafeBuffer: gB)
        cmd0.addCompletedHandler { [self] _ in
            let tg1 = CACurrentMediaTime()
            Task.detached(priority: .userInitiated) { [self] in
                do {
                    if !Self.bothVT {                                       // one search; A -> B candidates = -vB
                        let mb = try await ses.motion(of: rb, comparedTo: ra)
                        mb.motionVector.withUnsafeBuffer { j.vB = $0 }
                    } else {
                        async let fB = ses.motion(of: rb, comparedTo: ra)   // B(p) ≈ A(p + vB)
                        async let fA = ses.motion(of: ra, comparedTo: rb)   // A(p) ≈ B(p + vA)
                        let (mb, ma) = try await (fB, fA)
                        mb.motionVector.withUnsafeBuffer { j.vB = $0 }
                        ma.motionVector.withUnsafeBuffer { j.vA = $0 }
                    }
                } catch { j.error = error }
                let tm1 = CACurrentMediaTime()
                if Self.trace { Self.phase.append(((tg1 - tg0) * 1000, (tm1 - tg1) * 1000)) }
                if j.arrive() { run(j) }
            }
        }
        cmd0.commit()

        // While the guides and the motion search run: the output buffer and the plane textures.
        var outPB: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, outPool!, &outPB)
        j.out = outPB
        if let out = outPB {
            j.ga = guideTex(j, gA); j.gb = guideTex(j, gB)
            j.yA = tex(j, prev, 0, yFmt); j.cA = tex(j, prev, 1, cFmt); j.yB = tex(j, cur, 0, yFmt); j.cB = tex(j, cur, 1, cFmt)
            j.yO = tex(j, out, 0, yFmt); j.cO = tex(j, out, 1, cFmt)
        }
        if j.arrive() { run(j) }
    }

    /// One pair in flight: its buffers, textures and the motion search results. `pending` counts the two halves
    /// (frame textures, motion search) still running.
    private final class Job: @unchecked Sendable {
        let prev: CVPixelBuffer, cur: CVPixelBuffer
        let done: @Sendable (CVPixelBuffer?, Error?) -> Void
        var keep: [CVMetalTexture] = []
        var vA: CVPixelBuffer?, vB: CVPixelBuffer?, error: Error?
        var out: CVPixelBuffer?
        var ga, gb, yA, cA, yB, cB, yO, cO: MTLTexture?
        private var pending = 2
        private let lock = NSLock()
        init(prev: CVPixelBuffer, cur: CVPixelBuffer, done: @escaping @Sendable (CVPixelBuffer?, Error?) -> Void) {
            self.prev = prev; self.cur = cur; self.done = done
        }
        /// True for the side that completes the pair's preparation.
        func arrive() -> Bool { lock.lock(); defer { lock.unlock() }; pending -= 1; return pending == 0 }
    }
    private let inflight = DispatchSemaphore(value: 1)

    private func tex(_ j: Job, _ pb: CVPixelBuffer, _ plane: Int, _ f: MTLPixelFormat) -> MTLTexture? {
        var t: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache!, pb, nil, f, CVPixelBufferGetWidthOfPlane(pb, plane),
                                                  CVPixelBufferGetHeightOfPlane(pb, plane), plane, &t)
        guard let t else { return nil }
        j.keep.append(t)
        return CVMetalTextureGetTexture(t)
    }
    private func guideTex(_ j: Job, _ g: CVPixelBuffer) -> MTLTexture? {
        var t: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache!, g, nil, .r8Unorm, guide.width, guide.height, 0, &t)
        guard let t else { return nil }
        j.keep.append(t)
        return CVMetalTextureGetTexture(t)
    }

    /// Second half of a pair: both motion fields are in. Encodes and commits the GPU passes.
    private func run(_ j: Job) {
        let tm1 = CACurrentMediaTime()
        let prev = j.prev, cur = j.cur, completion = j.done
        guard let vBpb = j.vB else { return completion(nil, j.error ?? Self.err("motion")) }
        let vApb = j.vA
        guard let out = j.out, let ga = j.ga, let gb = j.gb,
              let vb = tex(j, vBpb, 0, .rg16Float), let va = vApb.flatMap({ tex(j, $0, 0, .rg16Float) }) ?? negField(vb.width, vb.height),
              let yA = j.yA, let cA = j.cA, let yB = j.yB, let cB = j.cB, let yO = j.yO, let cO = j.cO,
              var cmd = queue.makeCommandBuffer()
        else { return completion(nil, Self.err("textures")) }
        _ = (prev, cur)
        if Self.splat || hudClear, let bl = cmd.makeBlitCommandEncoder() {
            if Self.splat { bl.fill(buffer: splatA, range: 0..<splatA.length, value: 0); bl.fill(buffer: splatB, range: 0..<splatB.length, value: 0) }
            if hudClear { for b in hudS { bl.fill(buffer: b, range: 0..<b.length, value: 0) } }
            bl.endEncoding()
        }
        var hudOn: Int32 = Self.hud ? (Self.debugView == 1 ? (ProcessInfo.processInfo.environment["MFG_HUDDBG"] == "1" ? 3 : 2) : 1) : 0
        if hudClear { hudPairs = 0 }
        hudPairs += 1
        if hudOn != 0 && hudPairs < 48 { hudOn |= 4 }        // the long-term means are still settling: no evidence yet
        // One concurrent encoder: independent dispatches (the A and B sides) overlap, barriers mark the dependencies.
        guard var e = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else { return completion(nil, Self.err("encoder")) }
        func barrier() { e.memoryBarrier(scope: [.textures, .buffers]) }
        // Bench profiling (MFG_PROFILE=1): each stage in its own command buffer, GPU ms per stage.
        func cut(_ name: String) {
            guard Self.profile else { return barrier() }
            e.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            Self.stageMs[name, default: []].append((cmd.gpuEndTime - cmd.gpuStartTime) * 1000)
            cmd = queue.makeCommandBuffer()!; e = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)!
        }
        if vApb == nil {
            e.setTexture(vb, index: 0); e.setTexture(va, index: 1)
            dispatch(e, psoNeg, Size(vb.width, vb.height))
            barrier()
        }
        badCount.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        if (Self.p3 & 2) != 0 {
            e.setTexture(yA, index: 0); e.setTexture(yB, index: 1); e.setTexture(lockTex, index: 2)
            e.setTexture(ga, index: 4); e.setTexture(gb, index: 5)
            dispatch(e, psoLock, guide)
            if Self.profile { cut("lock") }
        }
        if Self.trace { Self.lastField = Size(vb.width, vb.height) }
        var blk = SIMD2<Float>(Float(guide.width) / Float(vb.width), Float(guide.height) / Float(vb.height))   // guide px per vector cell
        let fs = Self.fieldStats(vBpb, &medX, &medY)
        var glob = fs.median
        // A still camera makes the whole scene "static": the HUD evidence holds until the picture moves again.
        if hudOn != 0 && fs.level < 1.5 { hudOn |= 8 }
        var opt = SIMD2<Int32>(Self.candRadius, Self.wideSAD)
        var lam = Self.lambda
        var G = Float(Self.grid)
        if Self.splat {
            // Dense anchored fields A -> B and B -> A on the grid, propagated, splatted to the midpoint.
            // Ping-pong so that the last pass lands in F.
            let n = Self.anchorPasses
            let sides: [(S: MTLTexture, D: MTLTexture, vs: MTLTexture, vd: MTLTexture, F: MTLTexture, T: MTLTexture, g: SIMD2<Float>, sgn: Float, buf: MTLBuffer)] =
                [(ga, gb, va, vb, fA, fTmp, -glob, -1, splatA), (gb, ga, vb, va, fB, fTmpB, glob, 1, splatB)].filter { !(Self.oneSide && $0.sgn < 0) }
            for sd in sides {
                var ap = AnchorParams(blk: blk, glob: sd.g, lam: lam, step: 0, g: G)
                ap.temporal = hasPrev && Self.temporalSeed ? 1 : 0
                let po = 1 - par
                e.setTexture(sd.sgn < 0 ? fAs[po] : fBs[po], index: 5); e.setTexture(sd.sgn < 0 ? fBs[po] : fAs[po], index: 6)
                e.setTexture(sd.S, index: 0); e.setTexture(sd.D, index: 1); e.setTexture(sd.vs, index: 2); e.setTexture(sd.vd, index: 3)
                e.setTexture(n % 2 == 1 ? sd.T : sd.F, index: 4)
                e.setBytes(&ap, length: MemoryLayout<AnchorParams>.stride, index: 0)
                let t = tg[ObjectIdentifier(psoAnchor)] ?? MTLSize(width: 16, height: 16, depth: 1)
                e.setThreadgroupMemoryLength((t.width * t.height * 22 * 4 + 15) / 16 * 16, index: 0)
                dispatch(e, psoAnchor, gridSize)
            }
            for k in 0..<n {
                cut(k == 0 ? "anchor0" : "anchorP")
                for sd in sides {
                    var ap = AnchorParams(blk: blk, glob: sd.g, lam: lam, step: Self.propSteps[k % Self.propSteps.count], g: G)
                    let src = (n - k) % 2 == 1 ? sd.T : sd.F, dst = src === sd.T ? sd.F : sd.T
                    e.setTexture(sd.S, index: 0); e.setTexture(sd.D, index: 1); e.setTexture(src, index: 2); e.setTexture(dst, index: 3)
                    e.setBytes(&ap, length: MemoryLayout<AnchorParams>.stride, index: 0)
                    dispatch(e, psoAnchorProp, gridSize)
                }
            }
            cut(n == 0 ? "anchor0" : "anchorP")
            for sd in sides {
                var spp = SIMD4<Float>(glob.x, glob.y, sd.sgn, Self.splatMaxCost)
                e.setTexture(sd.F, index: 0); e.setBuffer(sd.buf, offset: 0, index: 0); e.setBytes(&spp, length: 16, index: 1)
                e.setBytes(&G, length: 4, index: 2)
                e.setTexture(sd.F === fA ? fB : fA, index: 1)
                e.setBuffer(hudS[hudPar], offset: 0, index: 3); e.setBytes(&hudOn, length: 4, index: 4)
                dispatch(e, psoSplat, gridSize)
            }
            cut("splat")
        }
        let selSize = Self.splat ? gridSize : guide
        var selOut = Self.splat && Self.grid > 1 ? gridMid : midTex
        e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(vb, index: 2); e.setTexture(va, index: 3)
        e.setTexture(selOut, index: 4)
        e.setBytes(&blk, length: 8, index: 0)
        e.setBuffer(badCount, offset: 0, index: 1)
        var spOn = SIMD4<Float>(Self.splat ? 1 : 0, 0, Self.splat ? G : 1, Self.splatHard)
        e.setTexture(fA, index: 5); e.setTexture(fB, index: 6)
        e.setBuffer(splatA, offset: 0, index: 5); e.setBuffer(splatB, offset: 0, index: 6); e.setBytes(&spOn, length: 16, index: 7)
        e.setBytes(&glob, length: 8, index: 2)
        e.setBytes(&opt, length: 8, index: 3)
        e.setBytes(&lam, length: 4, index: 4)
        e.setBuffer(hudS[hudPar], offset: 0, index: 8); e.setBuffer(hudS[1 - hudPar], offset: 0, index: 9); e.setBytes(&hudOn, length: 4, index: 10)
        e.setBuffer(emaBuf, offset: 0, index: 11)
        e.setTexture(gridAlt, index: 7)
        dispatch(e, psoSelect, selSize)
        if (Self.p3 & 2) != 0 {                           // the lock mask was written before the anchored search
            e.setTexture(lockTex, index: 0); e.setTexture(lockDil, index: 1); e.setTexture(lockBits, index: 2)
            dispatch(e, psoLockDil, guide)
        }
        if selOut === gridMid && (Self.p2 & 64) != 0 {
            cut("select")
            var th = SIMD2<Float>(Self.holeCost, 0.06)
            e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(gridMid, index: 2); e.setTexture(gridFix, index: 3)
            e.setBytes(&G, length: 4, index: 0); e.setBuffer(hudS[hudPar], offset: 0, index: 1); e.setBytes(&hudOn, length: 4, index: 2)
            e.setBytes(&th, length: 8, index: 3)
            dispatch(e, psoGridFix, gridSize)
            selOut = gridFix
        }
        var fb = Self.fallback
        let fused = Self.fuse && selOut !== midTex && Self.refinePasses == 0 && !Self.smooth && Self.debugView == 0
        if fused {
            cut(selOut === gridFix ? "gridfix" : "select")
            e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(selOut, index: 2)
            e.setTexture(yA, index: 3); e.setTexture(yB, index: 4); e.setTexture(yO, index: 5)
            e.setTexture(cA, index: 6); e.setTexture(cB, index: 7); e.setTexture(cO, index: 8)
            e.setBytes(&G, length: 4, index: 0); e.setBytes(&fb, length: 8, index: 1)
            e.setTexture(lockDil, index: 9); e.setTexture(lockBits, index: 10); e.setTexture(lockTex, index: 12); e.setTexture(gridAlt, index: 11)
            dispatch(e, psoUpwarp, guide)
        } else if selOut !== midTex {
            cut(selOut === gridFix ? "gridfix" : "select")
            e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(selOut, index: 2); e.setTexture(midTex, index: 3)
            e.setBytes(&G, length: 4, index: 0)
            dispatch(e, psoUpsample, guide)
        }
        if !fused {
        cut("upsample")
        var field = midTex, spare = smoothTex
        for k in 0..<Self.refinePasses {
            var step = Int32([8, 4, 2, 1][k % 4])
            e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(field, index: 2); e.setTexture(spare, index: 3)
            e.setBytes(&step, length: 4, index: 0); e.setBytes(&lam, length: 4, index: 1); e.setBytes(&opt, length: 8, index: 2)
            dispatch(e, psoRefine, guide)
            swap(&field, &spare)
            barrier()
        }
        if Self.smooth {
            e.setTexture(ga, index: 0); e.setTexture(gb, index: 1); e.setTexture(field, index: 2); e.setTexture(spare, index: 3)
            dispatch(e, psoSmooth, guide)
            swap(&field, &spare)
        }
        cut("smooth")
        e.setTexture(yA, index: 0); e.setTexture(yB, index: 1); e.setTexture(Self.debugView == 5 ? fA : Self.debugView == 6 ? fB : field, index: 2); e.setTexture(yO, index: 3)
        var dbg = Self.debugView >= 5 ? 5 : Self.debugView
        e.setBytes(&dbg, length: 4, index: 0)
        e.setBytes(&fb, length: 8, index: 1)
        dispatch(e, psoWarpY, size)
        e.setTexture(cA, index: 0); e.setTexture(cB, index: 1); e.setTexture(field, index: 2); e.setTexture(cO, index: 3)
        dispatch(e, psoWarpC, guide)
        }
        if Self.hud {
            // After the selection (it read the mean of the previous pairs): fold the newer frame in, ~1 s time constant.
            barrier()
            var rate: Float = hudClear ? 1 : 1.0 / 32
            e.setTexture(gb, index: 0); e.setBuffer(emaBuf, offset: 0, index: 0); e.setBytes(&rate, length: 4, index: 1)
            dispatch(e, psoEma, guide)
        }
        hudClear = false
        if Self.profile { cut("warp") }
        e.endEncoding()
        let total = Float(Self.splat ? gridSize.width * gridSize.height : guide.width * guide.height), counter = badCount
        let tc = CACurrentMediaTime()
        if Self.trace { Self.stageMs["cpu.post", default: []].append((tc - tm1) * 1000) }
        cmd.addCompletedHandler { cb in
            _ = j.keep
            if Self.trace {
                Self.gpu.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                Self.stageMs["lat.start", default: []].append((cb.gpuStartTime - tc) * 1000)
                Self.stageMs["lat.done", default: []].append((CACurrentMediaTime() - cb.gpuEndTime) * 1000)
            }
            if let err = cb.error { return completion(nil, err) }
            let bad = Float(counter.contents().load(as: UInt32.self)) / total
            if bad > Self.rejectFraction { return completion(nil, nil) }
            CVBufferPropagateAttachments(cur, out)
            completion(out, nil)
        }
        cmd.commit()
        if Self.splat { par = 1 - par; hasPrev = true }
        hudPar = 1 - hudPar
    }

    // MARK: helpers

    /// -vB as the A -> B block field when only one motion search runs (MFG_VT2=0).
    private var negTex: MTLTexture?
    private func negField(_ w: Int, _ h: Int) -> MTLTexture? {
        if let t = negTex, t.width == w, t.height == h { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        negTex = device.makeTexture(descriptor: d)
        return negTex
    }

    /// Threadgroup shapes that differ from 16×16 (measured on an M2, see MOTION_PROGRESS v0.5).
    static let defaultTG: [String: MTLSize] = ["mfg_anchor": MTLSize(width: 8, height: 8, depth: 1)]
    private let tg: [ObjectIdentifier: MTLSize]

    private func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ s: Size) {
        e.setComputePipelineState(p)
        let t = tg[ObjectIdentifier(p)] ?? MTLSize(width: 16, height: 16, depth: 1)
        e.dispatchThreadgroups(MTLSize(width: (s.width + t.width - 1) / t.width, height: (s.height + t.height - 1) / t.height, depth: 1),
                               threadsPerThreadgroup: t)
    }

    private static func pool(_ fmt: OSType, _ s: Size) -> CVPixelBufferPool? {
        let a: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: fmt, kCVPixelBufferWidthKey as String: s.width,
                                kCVPixelBufferHeightKey as String: s.height, kCVPixelBufferMetalCompatibilityKey as String: true,
                                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary, a as CFDictionary, &p)
        return p
    }

    /// Median vector of a motion field (rg16Float, guide px): the camera motion. Occluded pixels take the candidate
    /// nearest to it as background. And the mean vector length (guide px), on the same sampled grid: forward driving
    /// has a near-zero median but a large mean. One locked pass; xs / ys are reusable scratch.
    static func fieldStats(_ pb: CVPixelBuffer, _ xs: inout [Float], _ ys: inout [Float]) -> (median: SIMD2<Float>, level: Float) {
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return (.zero, 0) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), rb = CVPixelBufferGetBytesPerRow(pb)
        // 4×4 blocks give a 16× larger field: sample a grid of about 8k cells (the median does not need them all).
        let step = max(1, Int((Double(w * h) / 8192).squareRoot().rounded(.up)))
        xs.removeAll(keepingCapacity: true); ys.removeAll(keepingCapacity: true)
        var sum: Float = 0
        for y in stride(from: step / 2, to: h, by: step) {
            let row = base.advanced(by: y * rb).assumingMemoryBound(to: Float16.self)
            for x in stride(from: step / 2, to: w, by: step) {
                let vx = Float(row[2 * x]), vy = Float(row[2 * x + 1])
                xs.append(vx); ys.append(vy)
                sum += (vx * vx + vy * vy).squareRoot()
            }
        }
        guard !xs.isEmpty else { return (.zero, 0) }
        let level = sum / Float(xs.count)
        xs.sort(); ys.sort()
        return (SIMD2(xs[xs.count / 2], ys[ys.count / 2]), level)
    }
    private var medX: [Float] = [], medY: [Float] = []

    private static func err(_ m: String) -> NSError { NSError(domain: "Motion", code: 2, userInfo: [NSLocalizedDescriptionKey: m]) }

    struct AnchorParams { var blk: SIMD2<Float>; var glob: SIMD2<Float>; var lam: Float; var step: Int32; var g: Float
                          var tap: Float = MotionInterpolator.anchorTap; var tap2: Float = MotionInterpolator.anchorTap2
                          var subAll: Int32 = MotionInterpolator.subAll; var temporal: Int32 = 0
                          var pring: Int32 = MotionInterpolator.propRing ? 1 : 0 }

    private final class MotionBox: @unchecked Sendable { var vA: CVPixelBuffer?; var vB: CVPixelBuffer?; var error: Error? }

    // MARK: shaders

    static let shaders = """
    #include <metal_stdlib>
    using namespace metal;
    constexpr sampler lc(coord::pixel, address::clamp_to_edge, filter::linear);
    constant bool kBorder [[function_constant(0)]];         // border-aware matching (MFG_BORDER)
    constant int kSad [[function_constant(1)]];             // selection match window (MFG_SAD9), fixed per pipeline
    // Anchored search, fixed per pipeline so the tap offsets and loops compile to constants: tap spacing (MFG_ATAP),
    // ring radius (MFG_ATAP2), ring in propagation (MFG_PRING), sub-pixel steps every pass (MFG_SUBALL).
    constant float kTap [[function_constant(2)]];
    constant float kTap2 [[function_constant(3)]];
    constant bool kPring [[function_constant(4)]];
    constant bool kSubAll [[function_constant(5)]];
    // v0.5 quality changes (bit set = on; MFG_P2 / the per-feature switches, see MOTION.md).
    constant int kP2 [[function_constant(6)]];
    constant float kOccE [[function_constant(7)]];          // proven occlusion: foreground match under this (MFG_OCCE)
    constant float kOccM [[function_constant(8)]];          // ... and the other end this much worse (MFG_OCCM)
    constant int kUpZ [[function_constant(9)]];             // P2_UPZ cost: 1 bilinear cells, 2 the pixel's own, 3 max
    constant float kSubC [[function_constant(10)]];         // P2_SUBPIX: only winners under this cost (MFG_SUBC)
    #define P2_OCC 1
    #define P2_UPZ 2
    #define P2_UPSIDE 4
    #define P2_MARGIN 8
    #define P2_SUBPIX 16
    #define P2_HOLE 64
    // v0.5 Pass 3 (MFG_P3 / per-feature switches, see MOTION.md).
    constant int kP3 [[function_constant(11)]];
    constant float kSoftS [[function_constant(12)]];        // soft upsample: cost scale (MFG_SOFTS)
    constant float kLockD [[function_constant(13)]];        // overlay lock: max A/B difference (MFG_LOCKD)
    constant float kLockC [[function_constant(14)]];        // overlay lock: min local contrast (MFG_LOCKC)
    constant float kCmT [[function_constant(15)]];          // contrast-adjusted match above this cost (MFG_CMT)
    constant bool kSoftSide [[function_constant(16)]];      // soft upsample across occlusion cells too (MFG_SOFTSIDE)
    constant int kP3Dbg [[function_constant(17)]];           // bench: 1 = lock mask white, 2 = soft-blended px white
    constant int kSoftZ [[function_constant(18)]];          // soft upsample fade cost: 0 cell, 1 pixel, 2 mean, 3 max (MFG_SOFTZ)
    constant bool kSoftK [[function_constant(19)]];          // soft upsample over 3×3 cells, tent 1.5 cells (MFG_SOFTK)
    constant bool kSoftZS [[function_constant(20)]];         // soft upsample: one vector, cells' fade costs blended (MFG_SOFTZS)
    constant float kSoftM [[function_constant(21)]];         // soft upsample: vectors this close (guide px) merge (MFG_SOFTM)
    constant bool kSoftC [[function_constant(22)]];          // soft upsample weights from the 3×3 window only (MFG_SOFTC)
    constant float kLockG [[function_constant(23)]];         // overlay lock pre-check: max guide A/B difference (MFG_LOCKG)
    constant float kLockW [[function_constant(24)]];         // overlay lock, weak px (next to a strong lock): max difference (MFG_LOCKW)
    constant bool kLockSrc [[function_constant(25)]];        // overlay lock: no sample taken from a locked px (MFG_LOCKSRC)
    constant float kLockB [[function_constant(26)]];         // overlay lock: a stroke at least this bright in the patch (MFG_LOCKB)
    constant float kLockK [[function_constant(27)]];         // overlay lock: a shadow at most this dark in the patch (MFG_LOCKK)
    constant float kSoftH [[function_constant(28)]];         // soft upsample: an unchanged pixel (zero vector below this cost) stays hard (MFG_SOFTH)
    #define P3_SOFT 1
    #define P3_LOCK 2
    #define P3_EDGE 4
    #define P3_CMATCH 8
    #define P3_NEAR 16
    #define P3_HUDPX 32
    inline bool inF(float2 p, float2 s) { return all(p >= 0.0) && all(p <= s); }
    // Both windows (radius r around pa and pb) entirely inside the frame: the per-tap checks can be skipped.
    inline bool inW(float2 pa, float2 pb, float r, float2 s) { return all(min(pa, pb) >= r) && all(max(pa, pb) <= s - r); }

    // Y plane (full res) -> 8-bit half-res luma guide (2×2 box via one bilinear tap at the shared corner).
    kernel void mfg_neg(texture2d<float, access::read> v [[texture(0)]], texture2d<float, access::write> o [[texture(1)]],
                        uint2 id [[thread_position_in_grid]]) {
        if (id.x >= o.get_width() || id.y >= o.get_height()) return;
        o.write(-v.read(id), id);
    }

    kernel void mfg_guide(texture2d<float, access::sample> y [[texture(0)]], texture2d<float, access::write> g [[texture(1)]],
                          constant float &gp [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
        if (id.x >= g.get_width() || id.y >= g.get_height()) return;
        float v = clamp((y.sample(lc, float2(id) * 2.0 + 1.0).r - 0.0627) * 1.1644, 0.0, 1.0);
        g.write(float4(pow(v, gp)), id);
    }

    // 5-tap cross SAD between A at pa and B at pb (guide pixels, bilinear = half-pel).
    // 3×3 taps 2 guide px apart (a 5×5 support = 10×10 full-res): steadier on flat / repetitive areas and thin lines.
    inline float sad9(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 pa, float2 pb) {
        float s = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float2 o = float2(i, j) * 2.0;
            float w = (i == 0 && j == 0) ? 0.2 : 0.1;
            s += w * fabs(a.sample(lc, pa + o).r - b.sample(lc, pb + o).r);
        }
        return s;
    }

    // 5×5 taps 2 guide px apart (9×9 guide = 18×18 full-res support), centre-weighted.
    inline float sad25(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 pa, float2 pb) {
        float s = 0, ws = 0;
        for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
            float2 o = float2(i, j) * 2.0;
            float w = 1.0 / (1.0 + float(i * i + j * j));
            s += w * fabs(a.sample(lc, pa + o).r - b.sample(lc, pb + o).r); ws += w;
        }
        return s / ws;
    }

    // Match window near the frame edge (3×3 taps 3 px apart, about the 5×5 window's reach): taps outside the frame on
    // either side are skipped, not clamp-sampled from the edge row.
    inline float sadP(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 pa, float2 pb, float2 sz) {
        float s = 0, ws = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float2 o = float2(i, j) * 3.0;
            if (!(inF(pa + o, sz) && inF(pb + o, sz))) continue;
            float w = 1.0 / (1.0 + float(i * i + j * j));
            s += w * fabs(a.sample(lc, pa + o).r - b.sample(lc, pb + o).r); ws += w;
        }
        return ws > 0 ? s / ws : 1.0;
    }

    inline float sad5(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 pa, float2 pb) {
        // Centre-weighted: near a HUD edge the outer taps reach the moving background, the centre tap decides.
        const float2 o[5] = { float2(0, 0), float2(1.5, 0), float2(-1.5, 0), float2(0, 1.5), float2(0, -1.5) };
        const float w[5] = { 0.4, 0.15, 0.15, 0.15, 0.15 };
        float s = 0;
        for (int i = 0; i < 5; i++) s += w[i] * fabs(a.sample(lc, pa + o[i]).r - b.sample(lc, pb + o[i]).r);
        return s;
    }

    // P3_CMATCH: 5-tap match after a gain / offset fit of A to B (a shadow or light change over the road between
    // the frames): s = clamp(cov / var, 0.80, 1.25), o = clamp(mean_B - s mean_A, ±0.04). Correspondence only: the
    // warp never applies s or o.
    inline float sad5c(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 pa, float2 pb) {
        const float2 o[5] = { float2(0, 0), float2(1.5, 0), float2(-1.5, 0), float2(0, 1.5), float2(0, -1.5) };
        const float w[5] = { 0.4, 0.15, 0.15, 0.15, 0.15 };
        float va[5], vb[5], ma = 0, mb = 0;
        for (int i = 0; i < 5; i++) { va[i] = a.sample(lc, pa + o[i]).r; vb[i] = b.sample(lc, pb + o[i]).r; ma += va[i]; mb += vb[i]; }
        ma *= 0.2; mb *= 0.2;
        float cv = 0, vr = 0;
        for (int i = 0; i < 5; i++) { cv += (va[i] - ma) * (vb[i] - mb); vr += (va[i] - ma) * (va[i] - ma); }
        float g = vr > 1e-5 ? clamp(cv / vr, 0.80, 1.25) : 1.0;
        float off = clamp(mb - g * ma, -0.04, 0.04);
        float r = 0;
        for (int i = 0; i < 5; i++) r += w[i] * fabs(g * va[i] + off - vb[i]);
        return r;
    }

    // A vector whose path (endpoints and quarter points) crosses a static-edge cell carries content through the HUD
    // layer (road pulled into the minimap, a panel border torn open): penalised.
    inline float crossPen(device const float2 *hud, int2 gs, float G, float2 m, float2 d) {
        const float f[4] = { 0.5, 0.25, -0.25, -0.5 };
        for (int k = 0; k < 4; k++) {
            int2 c = clamp(int2((m + f[k] * d) / G), int2(0), gs - 1);
            if (hud[c.y * gs.x + c.x].y > 0.5) return 0.03;
        }
        return 0.0;
    }

    // Midpoint candidate cost. kBorder: a vector with one endpoint outside the frame (content scrolling in or out)
    // has no two-sided evidence; it is judged by its distance from the local motion, just above a good match, and
    // the warp then samples the in-frame side only.
    inline float candCost(texture2d<float, access::sample> a, texture2d<float, access::sample> b, float2 m, float2 d,
                          int mode, float2 vref) {
        float2 pa = m + 0.5 * d, pb = m - 0.5 * d;
        float2 sz = float2(a.get_width(), a.get_height());
        if (kBorder && !inW(pa, pb, 4.0, sz)) {
            bool ia = inF(pa, sz), ib = inF(pb, sz);
            if (!ia && !ib) return 1.0;
            if (!ia || !ib) {
                // One endpoint outside: test the same vector just inside the frame (shifted so the outside endpoint
                // lands 4 px in), as the scrolling content around it moves coherently; small one-sided penalty.
                float2 po = ia ? pb : pa;
                float2 sh = clamp(po, float2(4.0), sz - 4.0) - po;
                float2 qa = pa + sh, qb = pb + sh;
                if (!(inF(qa, sz) && inF(qb, sz))) return 1.0;
                return sadP(a, b, qa, qb, sz) + 0.004 + 0.0005 * length(sh);
            }
            return sadP(a, b, pa, pb, sz);
        }
        return mode == 2 ? sad25(a, b, pa, pb) : mode != 0 ? sad9(a, b, pa, pb) : sad5(a, b, pa, pb);
    }

    // Per guide pixel m: best displacement d (A(x + d) = B(x)); midpoint samples A at m + d/2 and B at m - d/2.
    // Candidates: zero, vB of the 3×3 blocks around m, -vA of the 3×3 blocks around m. Small |d| bias keeps static
    // HUD / text on the zero vector. Out: (d.x, d.y, cost, side) ; side 0 = blend, 1 = A only, 2 = B only.
    kernel void mfg_select(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                           texture2d<float, access::read> vb [[texture(2)]], texture2d<float, access::read> va [[texture(3)]],
                           texture2d<half, access::write> mid [[texture(4)]], constant float2 &blk [[buffer(0)]],
                           device atomic_uint *bad [[buffer(1)]], constant float2 &glob [[buffer(2)]], constant int2 &opt [[buffer(3)]], constant float &lam [[buffer(4)]],
                           texture2d<half, access::read> fa [[texture(5)]], texture2d<half, access::read> fb [[texture(6)]],
                           device const uint *spA [[buffer(5)]], device const uint *spB [[buffer(6)]], constant float4 &sp [[buffer(7)]],
                           device const float2 *hudP [[buffer(8)]], device float2 *hudN [[buffer(9)]], constant int &hudOn [[buffer(10)]],
                           device const float2 *ema [[buffer(11)]], texture2d<half, access::write> alt [[texture(7)]],
                           uint2 id [[thread_position_in_grid]]) {
        #define CROSS(d) (hudOn != 0 ? crossPen(hudP, int2(gs), sp.z, m, d) : 0.0)
        #define SAD(pa, pb) (kSad == 2 ? sad25(ga, gb, pa, pb) : kSad != 0 ? sad9(ga, gb, pa, pb) : sad5(ga, gb, pa, pb))
        float2 gs = float2(mid.get_width(), mid.get_height());
        if (id.x >= uint(gs.x) || id.y >= uint(gs.y)) return;
        float2 m = (float2(id) + 0.5) * sp.z;                     // grid cell centre in guide px
        int2 cells = int2(vb.get_width(), vb.get_height());
        int2 c0 = int2(m / blk);
        // Local prior: the block field at m (bilinear between block centres). Flat / aperture areas tie on SAD, the
        // prior then keeps the vector of their own block instead of a neighbour's.
        float2 cf = m / blk - 0.5;
        int2 i0 = clamp(int2(floor(cf)), int2(0), cells - 1), i1 = clamp(i0 + 1, int2(0), cells - 1);
        float2 fr = clamp(cf - floor(cf), 0.0, 1.0);
        float2 vloc = mix(mix(float2(vb.read(uint2(i0)).rg), float2(vb.read(uint2(i1.x, i0.y)).rg), fr.x),
                          mix(float2(vb.read(uint2(i0.x, i1.y)).rg), float2(vb.read(uint2(i1)).rg), fr.x), fr.y);
        float2 best = float2(0);
        float bestCost = SAD(m, m) - 0.004;                    // zero-vector bias (static HUD)
        // Neighbouring blocks mostly agree: test each distinct vector once (quarter-pel tolerance). Gathered first,
        // then scored in one loop (one inlined cost path keeps the kernel small).
        // Kept small (half precision, 20 entries): large per-thread arrays spill and slow the whole kernel.
        half2 seen[20]; half sc[20]; uchar fl[20]; int n = 1;   // fl: 1 = consistent splat, 2 = block vector (prior)
        int R = clamp(opt.x, -1, 2);
        seen[0] = half2(0); sc[0] = half(bestCost); fl[0] = 0;
        // Splatted candidates: dense endpoint vectors projected to the midpoint, foreground first. A thin pole's own
        // vector lands where the pole is at the midpoint (the symmetric match alone ties with the background there).
        if (sp.x > 0) {
            uint W = mid.get_width();
            for (int t = 0; t < 5; t++) {
                const int2 o5[5] = { int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
                int2 q = clamp(int2(id) + o5[t], int2(0), int2(gs) - 1);
                uint qi = uint(q.y) * W + uint(q.x);
                for (int sd = 0; sd < 2; sd++) {
                    uint key = sd == 0 ? spA[qi] : spB[qi];
                    if (key == 0) continue;
                    uint src = key & 0xFFFFFu;
                    uint2 sxy = uint2(src % W, src / W);
                    float2 d = sd == 0 ? -float2(fa.read(sxy).xy) : float2(fb.read(sxy).xy);
                    uchar cons = key >= (2048u << 20) ? 1 : 0;
                    int dq = -1;
                    for (int q2 = 0; q2 < n; q2++) if (all(fabs(float2(seen[q2]) - d) < 0.25)) { dq = q2; break; }
                    if (dq >= 0) { fl[dq] |= cons; continue; }
                    if (n >= 20) continue;
                    seen[n] = half2(d); fl[n] = cons; n++;
                }
            }
        }
        for (int j = -R; j <= R; j++) for (int i = -R; i <= R; i++) {
            int2 c = clamp(c0 + int2(i, j), int2(0), cells - 1);
            float2 cands[2] = { float2(vb.read(uint2(c)).rg), -float2(va.read(uint2(c)).rg) };
            for (int k = 0; k < 2; k++) {
                float2 d = cands[k];
                if (all(fabs(d) < 0.25)) continue;                       // zero already tested
                bool dup = false;
                for (int q = 1; q < n; q++) if (all(fabs(float2(seen[q]) - d) < 0.25)) { dup = true; break; }
                if (dup || n >= 20) continue;
                seen[n] = half2(d); fl[n] = 2; n++;
            }
        }
        float2 cbest = float2(0); float ccost = (fl[0] & 1) ? bestCost : 1e9;   // best forward-backward consistent splat
        float2 mbest = float2(0); float mcost = 1e9;            // P3_HUDPX: best moving vector, HUD crossing not penalised
        for (int q = 1; q < n; q++) {
            float2 d = float2(seen[q]);
            float c1 = candCost(ga, gb, m, d, kSad, vloc);
            float cost = c1 + CROSS(d) + ((fl[q] & 2) ? lam * min(distance(d, vloc), 12.0) : 0.0);
            if ((kP3 & P3_HUDPX) && c1 < mcost) { mcost = c1; mbest = d; }
            sc[q] = half(cost);
            if (cost < bestCost) { bestCost = cost; best = d; }
            if ((fl[q] & 1) && cost < ccost) { ccost = cost; cbest = d; }
        }
        // P3_CMATCH: an ambiguous cell (no good plain match) re-ranks its candidates by the gain / offset-fitted match;
        // the cost kept is the plain one of the vector chosen (the fade to the blend stays honest).
        if ((kP3 & P3_CMATCH) && bestCost > kCmT) {
            float cb2 = 1e9; int qb = -1;
            for (int q = 0; q < n; q++) {
                float2 d = float2(seen[q]);
                float2 pa = m + 0.5 * d, pb = m - 0.5 * d;
                if (!inW(pa, pb, 3.0, float2(ga.get_width(), ga.get_height()))) continue;
                float c = sad5c(ga, gb, pa, pb) + CROSS(d) - (q == 0 ? 0.004 : 0.0);
                if (c < cb2) { cb2 = c; qb = q; }
            }
            if (qb >= 0) { best = float2(seen[qb]); bestCost = float(sc[qb]); }
        }
        // A consistent splat that matches well enough is the occupancy at the midpoint: it wins over a cheaper vector
        // that pairs two unrelated look-alike areas (dark road under the player's car).
        float preBest = bestCost;
        bool ovr = sp.w > 0 && ccost < sp.w;
        if (ovr && (kP2 & P2_MARGIN)) {
            // Unique-match margin (repeated roof lights, railings): a splat of the wrong period is consistent too.
            // It may override a cheaper vector more than 1.5 guide px away only while that one is not clearly
            // better: C_splat - C_far <= max(0.004, 0.15 C_far).
            float cfar = 1e9;
            for (int q = 0; q < n; q++) if (distance(float2(seen[q]), cbest) > 1.5) cfar = min(cfar, float(sc[q]));
            if (ccost - cfar > max(0.004, 0.15 * cfar)) ovr = false;
        }
        if (ovr) { best = cbest; bestCost = ccost; }
        // Static HUD: evidence "a static edge is here", rising slowly and falling fast (after a settling period for the
        // long-term means); the mask turns on at 0.75 and off below 0.35. A masked cell keeps the zero vector unless
        // a good moving match clearly beats it (the player's car shifting, a cloud edge entering flat sky).
        bool masked = false, stE = false;
        half4 altv = half4(0);                                 // P3_HUDPX: the moving vector a masked cell gave up
        if (hudOn != 0) {
            uint hi = id.y * uint(gs.x) + id.x;
            float2 h = hudP[hi], hn = h;
            float zc = float(sc[0]) + 0.004;
            // Static edge: the long-term mean of the guide keeps most of the mean gradient magnitude here (HUD panel
            // border, text, the player's car outline). Flat areas have no gradient, moving texture averages out.
            int2 gmax = int2(ga.get_width() - 1, ga.get_height() - 1);
            int GW = gmax.x + 1, cnt = 0;
            for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
                int2 q = clamp(int2(m) + int2(i, j) * 2, int2(1), gmax - 1);
                int k = q.y * GW + q.x;
                float2 gm = 0.5 * float2(ema[k + 1].x - ema[k - 1].x, ema[k + GW].x - ema[k - GW].x);
                float g = length(gm);
                float2 pq = float2(q) + 0.5;                       // and both frames show it now (not an oscillation)
                bool now = fabs(ga.sample(lc, pq).r - ema[k].x) < 0.12 && fabs(gb.sample(lc, pq).r - ema[k].x) < 0.12;
                if (g > 0.02 && g > 0.6 * ema[k].y && now) cnt++;
            }
            bool st = cnt >= 2 && (hudOn & 4) == 0;
            stE = st;
            if ((hudOn & 8) == 0) {                              // the camera moves: static edges mean HUD
                hn.x = clamp(h.x + (st ? 1.0 / 16.0 : -0.25), 0.0, 1.0);
                hn.y = hn.x >= 0.75 ? 1.0 : (hn.x < 0.35 ? 0.0 : h.y);
            }
            hudN[hi] = hn;
            if (h.y > 0.5 && !(preBest < 0.03 && zc > preBest + 0.03)) {
                masked = true; best = float2(0); bestCost = zc;
            }
        }
        // Occlusion: no vector fits both ends, so the midpoint pixel is background seen from one side only.
        // Take the two most different good candidates (foreground f, background g). For a guess g, one of its two
        // endpoint samples lands on the foreground; that sample is explained by f (it moved with f between frames).
        // The guess whose foreground sample is best explained wins; keep the other endpoint only.
        float side = 0;
        const float T = 0.06;
        // HUD edge: the cross taps reach the moving background, but the pixel itself did not change: keep it still.
        if (masked) {}
        else if (bestCost > T && fabs(ga.sample(lc, m).r - gb.sample(lc, m).r) < 0.008) { best = float2(0); }
        else if (bestCost > T) {
            // g = candidate nearest the camera motion, f = cheapest candidate clearly different from g.
            float2 g = float2(0); float gd = 1e9;
            for (int q = 0; q < n; q++) { float dd = distance(float2(seen[q]), glob); if (dd < gd) { gd = dd; g = float2(seen[q]); } }
            float2 f = g; float altCost = 1e9;
            for (int q = 0; q < n; q++) if (distance(float2(seen[q]), g) > 1.5 && float(sc[q]) < altCost) { altCost = float(sc[q]); f = float2(seen[q]); }
            float2 cheap = best;
            best = g;
            side = 1;
            bool proven = false;
            if (altCost < 1e8) {
                float2 qa = m + 0.5 * g, rb = m - 0.5 * g;
                float eA = SAD(qa, qa - f);              // A's sample moved with f: foreground there, use B
                float eB = SAD(rb + f, rb);              // B's sample moved with f: foreground there, use A
                side = eA < eB ? 2 : 1;
                // Proven occlusion: one end of g clearly lands on the foreground f, and f explains it well.
                proven = min(eA, eB) < kOccE && max(eA, eB) > min(eA, eB) + kOccM;
            }
            // Unresolved bad match (P2_OCC): no proof that one side is covered. A one-sided sample along the camera
            // motion then pastes unrelated content (a dark shadow block on fast road); keep the cheapest two-sided
            // match instead, and let the low-confidence fade decide.
            if ((kP2 & P2_OCC) && !proven) { best = cheap; side = 0; }
            atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
        }
        // P3_HUDPX: a cell left on the zero vector (static HUD, or moving vectors penalised for crossing it) offers
        // its best moving vector; each pixel then decides by its own match (the road between subtitle lines moves).
        if (kP3 & P3_HUDPX) alt.write(all(fabs(best) < 0.25) && mcost < float(sc[0]) && hudOn != 0 ? half4(half2(mbest), half(mcost), 1) : altv, id);
        if (((hudOn & 3) == 2 && masked) || ((hudOn & 3) == 3 && stE)) side = 3;          // bench: FG_DEBUG_VIEW=1 shows the mask (MFG_HUDDBG=1: the evidence) white
        mid.write(half4(half2(best), half(bestCost), half(side)), id);
        #undef SAD
        #undef CROSS
    }

    // Midpoint grid, 1-ring pull (P2_HOLE): a cell whose match is poor (a hole in the splat, a wrong block vector on
    // fast road) tries the vectors its 8 neighbours chose. It takes the best one that clearly improves its own cost;
    // a pull that matches well also clears an occlusion side. Static-HUD cells keep their zero vector.
    kernel void mfg_gridfix(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                            texture2d<half, access::read> gi [[texture(2)]], texture2d<half, access::write> go [[texture(3)]],
                            constant float &G [[buffer(0)]], device const float2 *hudP [[buffer(1)]], constant int &hudOn [[buffer(2)]],
                            constant float2 &thr [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
        int2 hi = int2(gi.get_width() - 1, gi.get_height() - 1);
        if (int(id.x) > hi.x || int(id.y) > hi.y) return;
        half4 c = gi.read(id);
        float c0 = float(c.z);
        int2 gs = hi + 1;
        bool masked = hudOn != 0 && hudP[id.y * uint(gs.x) + id.x].y > 0.5;
        if (masked || c0 <= thr.x) { go.write(c, id); return; }
        float2 m = (float2(id) + 0.5) * G, own = float2(c.xy);
        float2 tried[8]; int nt = 0;
        float bc = c0, bt = min(c0 - 0.002, c0 * 0.85); float2 best = own; bool hit = false;
        for (int k = 0; k < 9; k++) {
            if (k == 4) continue;
            float2 d = float2(gi.read(uint2(clamp(int2(id) + int2(k % 3 - 1, k / 3 - 1), int2(0), hi))).xy);
            if (all(fabs(d - own) < 0.25)) continue;
            bool dup = false;
            for (int q = 0; q < nt; q++) if (all(fabs(tried[q] - d) < 0.25)) { dup = true; break; }
            if (dup) continue;
            tried[nt++] = d;
            float cost = candCost(ga, gb, m, d, kSad, d) + (hudOn != 0 ? crossPen(hudP, gs, G, m, d) : 0.0);
            if (cost < bt && cost < bc) { bc = cost; best = d; hit = true; }
        }
        if (!hit) { go.write(c, id); return; }
        go.write(half4(half2(best), half(bc), (c.w == 0 || bc <= thr.y) ? half(0) : c.w), id);
    }

    // Anchored dense field S -> D at guide resolution: F(x) with S(x) ≈ D(x + F). Unlike the midpoint match, a pixel
    // of a thin foreground object in S only matches D where that object went: no tie with the background vector.
    struct AnchorParams { float2 blk; float2 glob; float lam; int step; float g; float tap; float tap2; int subAll; int temporal; int pring; };
    // Anchored match cost: 3×3 taps `tap` guide px apart around S's position (cached in sv[0..8]), centre-weighted,
    // plus (tap2 > 0) a context ring of 8 taps tap2 px out (sv[9..16]), weighted like the whole inner window.
    // Returns early (≥ bound) once the inner part alone cannot beat `bound`.
    inline float sadS(thread const float *sv, texture2d<float, access::sample> D, float2 pd, float tp, float tp2, float bound, float2 ps) {
        if (kBorder && !inF(pd, float2(D.get_width(), D.get_height()))) return 1.0;   // a match may not leave the frame
        float r = 0; int t = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float w = 1.0 / (1.0 + float(i * i + j * j));
            r += w * fabs(sv[t] - D.sample(lc, pd + float2(i, j) * tp).r); t++;
        }
        r *= 1.0 / (1.0 + 2.0 + 4.0 / 3.0);                             // Σw
        if (tp2 <= 0) return r;
        r *= 0.5;
        if (r >= bound) return r;
        float c = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) continue;
            c += fabs(sv[t] - D.sample(lc, pd + float2(i, j) * tp2).r); t++;
        }
        return r + c * (0.5 / 8.0);
    }
    inline void loadS(thread float *sv, texture2d<float, access::sample> S, float2 m, float tp, float tp2) {
        int t = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) sv[t++] = S.sample(lc, m + float2(i, j) * tp).r;
        if (tp2 > 0) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) if (i != 0 || j != 0) sv[t++] = S.sample(lc, m + float2(i, j) * tp2).r;
    }
    // Field cells are P.g guide px apart (cell centre m = (id + 0.5) * g). Candidates: zero, camera motion, 3×3 blocks
    // of both VT fields (distinct ones only), with a pull toward the local block field.
    kernel void mfg_anchor(texture2d<float, access::sample> S [[texture(0)]], texture2d<float, access::sample> D [[texture(1)]],
                           texture2d<float, access::read> vs [[texture(2)]], texture2d<float, access::read> vd [[texture(3)]],
                           texture2d<half, access::write> F [[texture(4)]], constant AnchorParams &P [[buffer(0)]],
                           texture2d<half, access::read> ps [[texture(5)]], texture2d<half, access::read> po [[texture(6)]],
                           threadgroup half2 *triedTG [[threadgroup(0)]],
                           uint2 id [[thread_position_in_grid]], uint tix [[thread_index_in_threadgroup]]) {
        if (id.x >= F.get_width() || id.y >= F.get_height()) return;
        float2 m = (float2(id) + 0.5) * P.g;
        float sv[17]; loadS(sv, S, m, kTap, kTap2);
        int2 cells = int2(vs.get_width(), vs.get_height());
        int2 c0 = int2(m / P.blk);
        float2 cf = m / P.blk - 0.5;
        int2 i0 = clamp(int2(floor(cf)), int2(0), cells - 1), i1 = clamp(i0 + 1, int2(0), cells - 1);
        float2 fr = clamp(cf - floor(cf), 0.0, 1.0);
        float2 vloc = mix(mix(float2(vs.read(uint2(i0)).rg), float2(vs.read(uint2(i1.x, i0.y)).rg), fr.x),
                          mix(float2(vs.read(uint2(i0.x, i1.y)).rg), float2(vs.read(uint2(i1)).rg), fr.x), fr.y);
        // Distinct-vector list in threadgroup memory: a per-thread array with a dynamic index would live in slow
        // per-thread memory or take registers the match loop needs (texture latency is what limits this kernel).
        threadgroup half2 *tried = triedTG + tix * 22;
        int nt = 0;
        float2 best = float2(0);
        float bc = sadS(sv, D, m, kTap, kTap2, 1e9, m) - 0.004;
        tried[nt++] = half2(0);
        for (int k = -3; k < 18; k++) {
            float2 d;
            if (k < -1) {                                   // previous pair: same side, and the other side reversed
                if (P.temporal == 0) continue;
                d = k == -3 ? float2(ps.read(id).xy) : -float2(po.read(id).xy);
            }
            else if (k < 0) d = P.glob;
            else {
                int2 c = clamp(c0 + int2((k >> 1) % 3 - 1, (k >> 1) / 3 - 1), int2(0), cells - 1);
                d = (k & 1) ? -float2(vd.read(uint2(c)).rg) : float2(vs.read(uint2(c)).rg);
            }
            bool dup = false;
            for (int q = 0; q < nt; q++) if (all(fabs(float2(tried[q]) - d) < 0.25)) { dup = true; break; }
            if (dup) continue;
            tried[nt++] = half2(d);
            float pen = P.lam * min(distance(d, vloc), 12.0);
            float cost = sadS(sv, D, m + d, kTap, kTap2, bc - pen, m) + pen;
            if (cost < bc) { bc = cost; best = d; }
        }
        F.write(half4(half2(best), half(bc), 0), id);
    }

    // Propagation on an anchored field: the 8 cells `step` away (distinct vectors only) and, on the step-1 pass (or
    // every pass with subAll), ±0.5 px around the winner (sub-pixel).
    kernel void mfg_anchor_prop(texture2d<float, access::sample> S [[texture(0)]], texture2d<float, access::sample> D [[texture(1)]],
                                texture2d<half, access::read> Fi [[texture(2)]], texture2d<half, access::write> Fo [[texture(3)]],
                                constant AnchorParams &P [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
        int2 hi = int2(Fi.get_width() - 1, Fi.get_height() - 1);
        if (int(id.x) > hi.x || int(id.y) > hi.y) return;
        float2 m = (float2(id) + 0.5) * P.g;
        half4 c = Fi.read(id);
        float2 own = float2(c.xy), best = own;
        float bc = float(c.z);
        float sv[17]; bool loaded = false;
        const float t2 = kPring ? kTap2 : 0.0;     // context ring in the propagation passes too?
        if (!kPring) { loadS(sv, S, m, kTap, t2); loaded = true; bc = sadS(sv, D, m + own, kTap, t2, 1e9, m); }
        float2 cd[8];                               // the 8 neighbours, distinct-vector test on registers
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int kk = k < 4 ? k : k + 1;
            cd[k] = float2(Fi.read(uint2(clamp(int2(id) + int2(kk % 3 - 1, kk / 3 - 1) * P.step, int2(0), hi))).xy);
        }
        uint tried = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 d = cd[k];
            bool dup = all(fabs(own - d) < 0.25);
            #pragma unroll
            for (int q = 0; q < k; q++) dup = dup || (((tried >> q) & 1) && all(fabs(cd[q] - d) < 0.25));
            if (dup) continue;
            tried |= 1u << k;
            if (!loaded) { loadS(sv, S, m, kTap, t2); loaded = true; }
            float pen = P.lam * min(distance(d, own), 12.0);
            float cost = sadS(sv, D, m + d, kTap, t2, bc - pen, m) + pen;
            if (cost < bc) { bc = cost; best = d; }
        }
        if (P.step == 1 || kSubAll) {
            if (!loaded) loadS(sv, S, m, kTap, t2);
            const float2 sub[4] = { float2(0.5, 0), float2(-0.5, 0), float2(0, 0.5), float2(0, -0.5) };
            float2 b0 = best;
            // P2_SUBPIX on the last pass: the four offsets are scored in full, then a parabola per axis through
            // (-0.5, 0, +0.5) gives a sub-pixel minimum (bounded to ±0.5) for a confident winner on an edge; it is
            // kept only if its own resampled cost is lower.
            bool fit = (kP2 & P2_SUBPIX) && P.step == 1 && bc < kSubC;
            float cs[4];
            for (int k = 0; k < 4; k++) {
                float2 d = b0 + sub[k];
                float raw = sadS(sv, D, m + d, kTap, t2, fit ? 1e9 : bc - 0.002, m);
                cs[k] = raw;
                float cost = raw + 0.002;
                if (cost < bc) { bc = cost; best = d; }
            }
            if (fit) {
                float c0 = sadS(sv, D, m + b0, kTap, t2, 1e9, m);
                float2 o = 0;
                float dx = cs[0] - 2.0 * c0 + cs[1], dy = cs[2] - 2.0 * c0 + cs[3];
                if (dx > 0.004) o.x = clamp(0.25 * (cs[1] - cs[0]) / dx, -0.5, 0.5);
                if (dy > 0.004) o.y = clamp(0.25 * (cs[3] - cs[2]) / dy, -0.5, 0.5);
                if (any(fabs(o) > 0.06)) {
                    float2 d = b0 + o;
                    float cost = sadS(sv, D, m + d, kTap, t2, bc, m);
                    if (cost < bc) { bc = cost; best = d; }
                }
            }
        }
        Fo.write(half4(half2(best), half(bc), 0), id);
    }

    // Forward splat of an anchored field to the midpoint (nearest pixel). Collisions: the vector most different from
    // the camera motion wins (foreground / HUD over background), key = priority << 20 | source index.
    // P.z: midpoint vector d = P.z * F (A side -1, B side +1); P.w: max match cost to splat.
    // Forward-backward check first: a source pixel whose vector the other side's field does not send back (it is
    // covered at the other end, e.g. road sliding under the player's car) ranks below every consistent one.
    kernel void mfg_splat(texture2d<half, access::read> F [[texture(0)]], device atomic_uint *sp [[buffer(0)]],
                          constant float4 &P [[buffer(1)]], constant float &G [[buffer(2)]],
                          texture2d<half, access::read> Fo [[texture(1)]], device const float2 *hud [[buffer(3)]],
                          constant int &hudOn [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
        uint W = F.get_width(), H = F.get_height();
        if (id.x >= W || id.y >= H) return;
        half4 c = F.read(id);
        if (float(c.z) > P.w) return;
        float2 v = float2(c.xy);
        float2 t = ((float2(id) + 0.5) * G + 0.5 * v) / G;              // target cell
        if (any(t < 0.0) || t.x >= float(W) || t.y >= float(H)) return;
        float2 d = P.z * v;
        int2 o = int2(floor(((float2(id) + 0.5) * G + v) / G));
        bool cons = false;
        if (all(o >= 0) && o.x < int(W) && o.y < int(H)) cons = distance(float2(Fo.read(uint2(o)).xy), -v) < 1.0 + 0.05 * length(v);
        uint pr = (cons ? 2048u : 1u) + uint(clamp(distance(d, P.xy) * 16.0, 0.0, 2046.0));
        uint2 q = uint2(t);
        if (hudOn != 0 && (hud[id.y * W + id.x].y > 0.5 || hud[q.y * W + q.x].y > 0.5)) return;   // static HUD
        atomic_fetch_max_explicit(&sp[q.y * W + q.x], (pr << 20) | (id.y * W + id.x), memory_order_relaxed);
    }

    // Grid field (cells G guide px apart) -> per guide pixel: where the 4 surrounding cells disagree, each pixel takes
    // the one whose vector matches it best (3×3 symmetric SAD), so object outlines follow the pixels, not the grid.
    // Upsample cost of one cell's vector at m: 0.5 sad9 + 0.5 sad5 (the shared centre tap read once), summed in the
    // same order as the two windows on their own. Exits early (returning a value ≥ bound) once the sad9 part alone
    // cannot get under `bound`: every term is non-negative, so the partial sum is a lower bound of the full cost.
    inline float upCost(texture2d<float, access::sample> ga, texture2d<float, access::sample> gb, float2 m, float2 d, float bound) {
        float2 pa = m + 0.5 * d, pb = m - 0.5 * d;
        float c0 = fabs(ga.sample(lc, pa).r - gb.sample(lc, pb).r);
        float s9 = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            float2 o = float2(i, j) * 2.0;
            float w = (i == 0 && j == 0) ? 0.2 : 0.1;
            s9 += w * ((i == 0 && j == 0) ? c0 : fabs(ga.sample(lc, pa + o).r - gb.sample(lc, pb + o).r));
        }
        if (s9 * 0.5 >= bound) return s9 * 0.5;
        const float2 o5[4] = { float2(1.5, 0), float2(-1.5, 0), float2(0, 1.5), float2(0, -1.5) };
        float s5 = 0.4 * c0;
        for (int i = 0; i < 4; i++) s5 += 0.15 * fabs(ga.sample(lc, pa + o5[i]).r - gb.sample(lc, pb + o5[i]).r);
        return s9 * 0.5 + 0.5 * s5;
    }

    // Grid field -> vector of guide pixel id (see mfg_upsample). The nearest cell is one of the 4 surrounding ones;
    // identical vectors are scored once (first-wins tie order kept), the nearest one first as the early-exit bound.
    inline half4 upPick(texture2d<float, access::sample> ga, texture2d<float, access::sample> gb, texture2d<half, access::read> q,
                        float G, uint2 id) {
        int2 hi = int2(q.get_width() - 1, q.get_height() - 1);
        float2 m = float2(id) + 0.5;
        float2 u = m / G - 0.5;
        int2 c0 = int2(floor(u));
        half4 v[4]; int k = 0;
        for (int j = 0; j <= 1; j++) for (int i = 0; i <= 1; i++) v[k++] = q.read(uint2(clamp(c0 + int2(i, j), int2(0), hi)));
        int2 cn = int2(floor(u + 0.5)) - c0;                         // 0 or 1 per axis
        int ni = cn.y * 2 + cn.x;
        half4 near = v[ni];
        bool same = true;
        for (int t = 0; t < 4; t++) if (any(fabs(float2(v[t].xy) - float2(near.xy)) >= 0.25) || v[t].w != near.w) same = false;
        // P2_UPZ: the fade cost follows the pixel, not the cell: the four cells' costs interpolated bilinearly
        // (side-0 cells only), so a poor cell fades in smoothly instead of as an 8 px block.
        float2 fr = clamp(u - float2(c0), 0.0, 1.0);
        float zb = float(near.z);
        if (kP2 & P2_UPZ) {
            float zs = 0, ws = 0;
            for (int t = 0; t < 4; t++) {
                float w = (t & 1 ? fr.x : 1.0 - fr.x) * (t & 2 ? fr.y : 1.0 - fr.y);
                if (v[t].w == 0) { zs += w * float(v[t].z); ws += w; }
            }
            if (ws > 0) zb = zs / ws;
        }
        if (same) return (kP2 & P2_UPZ) && near.w == 0 ? half4(near.xy, half(zb), near.w) : near;
        if (near.w != 0 && !(kP2 & P2_UPSIDE)) return near;
        float2 gsz = float2(ga.get_width(), ga.get_height());
        // Score of candidate t; the lowest score wins, ties go to the lowest index (the original loop order).
        int bi = -1; float bc = 1e9;
        for (int r = 0; r < 4; r++) {
            int t = r == 0 ? ni : (r <= ni ? r - 1 : r);             // nearest first, then the others in order
            if (v[t].w != 0) continue;
            bool dup = false;                                        // an identical vector at a lower index scores the same
            for (int s = 0; s < t; s++) if (v[s].w == 0 && all(v[s].xy == v[t].xy)) { dup = true; break; }
            if (dup) continue;
            float2 d = float2(v[t].xy);
            bool z = all(fabs(d) < 0.25);
            float c;
            if (kBorder && !(inF(m + 0.5 * d, gsz) && inF(m - 0.5 * d, gsz))) c = 0.04 + 0.002 * distance(d, float2(near.xy));   // one-sided: keep the nearest cell's choice
            else {
                // Exit bound with a small margin: only a partial sum that is surely worse than the best so far exits.
                float bnd = bi < 0 ? 1e9 : bc + (z ? 0.004 : 0.0) + 1e-5;
                c = upCost(ga, gb, m, d, bnd);
            }
            if (z) c -= 0.004;
            if (c < bc || (c == bc && t < bi)) { bc = c; bi = t; }
        }
        if (!(kP2 & (P2_UPZ | P2_UPSIDE))) return bi < 0 ? near : v[bi];
        // P2_UPSIDE: the bilinear blend of the side-0 cells' vectors is one more candidate (a smooth field such as
        // fast road keeps its gradient instead of stair-stepping between cells), and a pixel of a one-sided cell
        // takes a two-sided vector when it matches well here.
        half4 r = bi < 0 ? near : v[bi];
        if (kP2 & P2_UPSIDE) {
            float2 vs = 0; float ws = 0;
            for (int t = 0; t < 4; t++) {
                float w = (t & 1 ? fr.x : 1.0 - fr.x) * (t & 2 ? fr.y : 1.0 - fr.y);
                if (v[t].w == 0) { vs += w * float2(v[t].xy); ws += w; }
            }
            if (ws > 0) {
                float2 d = vs / ws;
                if (inF(m + 0.5 * d, gsz) && inF(m - 0.5 * d, gsz)) {
                    float c = upCost(ga, gb, m, d, bi < 0 ? 1e9 : bc - 1e-4);
                    if (c < bc - 1e-4) { bc = c; r = half4(half2(d), 0, 0); bi = 4; }
                }
            }
            if (near.w != 0) {
                if (bi < 0 || bc > 0.06) return near;            // nothing two-sided fits: keep the occlusion side
                return half4(r.xy, half(bc), 0);
            }
        }
        if (bi < 0) return near;
        return half4(r.xy, (kP2 & P2_UPZ) ? half(kUpZ == 1 ? zb : (kUpZ == 2 ? bc : max(zb, bc))) : r.z, r.w);
    }

    kernel void mfg_upsample(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                             texture2d<half, access::read> q [[texture(2)]], texture2d<half, access::write> mid [[texture(3)]],
                             constant float &G [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
        if (id.x >= mid.get_width() || id.y >= mid.get_height()) return;
        mid.write(upPick(ga, gb, q, G, id), id);
    }

    // 3×3 vector median on the selected field: a pixel whose vector disagrees with its neighbours (ties in flat or
    // repetitive areas flip between candidates frame to frame = shimmer) takes the vector closest to all 9. Guarded:
    // only when that vector matches this pixel nearly as well (static HUD keeps its zero vector, real motion edges
    // keep theirs), and never on occlusion pixels (their side choice is per pixel).
    kernel void mfg_smooth(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                           texture2d<half, access::read> mid [[texture(2)]], texture2d<half, access::write> outv [[texture(3)]],
                           uint2 id [[thread_position_in_grid]]) {
        int2 hi = int2(mid.get_width() - 1, mid.get_height() - 1);
        if (int(id.x) > hi.x || int(id.y) > hi.y) return;
        half4 c = mid.read(id);
        if (c.w != 0) { outv.write(c, id); return; }
        float2 v[9]; int k = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++)
            v[k++] = float2(mid.read(uint2(clamp(int2(id) + int2(i, j), int2(0), hi))).xy);
        int bi = 4; float bs = 1e9;
        for (int a = 0; a < 9; a++) {
            float s = 0;
            for (int b = 0; b < 9; b++) { float2 d = fabs(v[a] - v[b]); s += d.x + d.y; }
            if (s < bs) { bs = s; bi = a; }
        }
        float2 med = v[bi];
        if (all(fabs(med - v[4]) < 0.25)) { outv.write(c, id); return; }
        float2 m = float2(id) + 0.5;
        float cost = sad5(ga, gb, m + 0.5 * med, m - 0.5 * med);
        if (cost <= float(c.z) + 0.006) outv.write(half4(half2(med), half(cost), 0), id);
        else outv.write(c, id);
    }

    // Propagation: own vector, the vectors chosen `step` guide px away (8 directions) and ±1 px around its own;
    // lowest SAD with a small pull toward its own vector wins. Occlusion pixels keep their side and vector.
    kernel void mfg_refine(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                           texture2d<half, access::read> mid [[texture(2)]], texture2d<half, access::write> outv [[texture(3)]],
                           constant int &step [[buffer(0)]], constant float &lam [[buffer(1)]], constant int2 &opt [[buffer(2)]],
                           uint2 id [[thread_position_in_grid]]) {
        int2 hi = int2(mid.get_width() - 1, mid.get_height() - 1);
        if (int(id.x) > hi.x || int(id.y) > hi.y) return;
        half4 c = mid.read(id);
        if (c.w != 0) { outv.write(c, id); return; }
        float2 m = float2(id) + 0.5, own = float2(c.xy), best = own;
        #define SADR(pa, pb) (opt.y == 2 ? sad25(ga, gb, pa, pb) : opt.y != 0 ? sad9(ga, gb, pa, pb) : sad5(ga, gb, pa, pb))
        float bc = SADR(m + 0.5 * own, m - 0.5 * own);
        if (all(fabs(own) < 0.25)) bc -= 0.004;                         // keep the static-HUD bias
        float2 cand[12]; int n = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) continue;
            cand[n++] = float2(mid.read(uint2(clamp(int2(id) + int2(i, j) * step, int2(0), hi))).xy);
        }
        cand[n++] = own + float2(1, 0); cand[n++] = own - float2(1, 0); cand[n++] = own + float2(0, 1); cand[n++] = own - float2(0, 1);
        for (int k = 0; k < n; k++) {
            float2 d = cand[k];
            if (all(fabs(d - own) < 0.25)) continue;
            float cost = SADR(m + 0.5 * d, m - 0.5 * d) + lam * min(distance(d, own), 12.0);
            if (all(fabs(d) < 0.25)) cost -= 0.004;
            if (cost < bc) { bc = cost; best = d; }
        }
        #undef SADR
        outv.write(half4(half2(best), half(bc), 0), id);
    }

    kernel void mfg_ema(texture2d<float, access::sample> g [[texture(0)]], device float2 *M [[buffer(0)]],
                        constant float &rate [[buffer(1)]], uint2 id [[thread_position_in_grid]]) {
        if (id.x >= g.get_width() || id.y >= g.get_height()) return;
        uint i = id.y * g.get_width() + id.x;
        float2 p = float2(id) + 0.5;
        float2 gr = 0.5 * float2(g.sample(lc, p + float2(1, 0)).r - g.sample(lc, p - float2(1, 0)).r,
                                 g.sample(lc, p + float2(0, 1)).r - g.sample(lc, p - float2(0, 1)).r);
        M[i] = mix(M[i], float2(g.sample(lc, p).r, length(gr)), rate);
    }

    inline bool inside(float2 p, float2 s) { return all(p >= 0.0) && all(p <= s); }
    // P3_EDGE: weight of a sample by its distance inside the frame (0 at the edge or outside, 1 from M px in).
    inline float edgeW(float2 p, float2 s, float M) { return clamp(min(min(p.x, p.y), min(s.x - p.x, s.y - p.y)) / M, 0.0, 1.0); }

    // Warp of one luma pixel (full res; v in guide px, ×2 here) / one CbCr texel (guide res) with field value v.
    // Warp before the low-confidence fade; f = the fade amount (toward the plain blend at the pixel).
    inline float warpY0(texture2d<float, access::sample> ya, texture2d<float, access::sample> yb, float4 v, uint2 id, float2 s, float2 fb, thread float &f) {
        float2 p = float2(id) + 0.5, d = v.xy * 2.0;
        float2 pa = p + 0.5 * d, pb = p - 0.5 * d;
        bool okA = inside(pa, s), okB = inside(pb, s);
        float a = ya.sample(lc, pa).r, b = yb.sample(lc, pb).r;
        float o = (v.w == 1 || !okB) && okA ? a : ((v.w == 2 || !okA) && okB ? b : (okA && okB ? 0.5 * (a + b) : ya.sample(lc, p).r));
        if ((kP3 & P3_EDGE) && v.w == 0 && !inW(pa, pb, 8.0, s)) {   // the side leaving the frame fades out over 8 px
            float wa = edgeW(pa, s, 8.0), wb = edgeW(pb, s, 8.0);
            if (wa + wb > 0) o = (wa * a + wb * b) / (wa + wb);
        }
        f = v.w == 0 && (!kBorder || (okA && okB) || (kP3 & P3_EDGE)) ? smoothstep(fb.x, fb.y, v.z) : 0.0;
        return o;
    }
    inline float warpY(texture2d<float, access::sample> ya, texture2d<float, access::sample> yb, float4 v, uint2 id, float2 s, float2 fb) {
        float k; float o = warpY0(ya, yb, v, id, s, fb, k);
        float2 p = float2(id) + 0.5;
        if (k > 0) o = mix(o, 0.5 * (ya.sample(lc, p).r + yb.sample(lc, p).r), k);
        return o;
    }
    inline float2 warpC0(texture2d<float, access::sample> ca, texture2d<float, access::sample> cb, float4 v, uint2 id, float2 s, float2 fb, thread float &f) {
        float2 p = float2(id) + 0.5, d = v.xy;
        float2 pa = p + 0.5 * d, pb = p - 0.5 * d;
        bool okA = inside(pa, s), okB = inside(pb, s);
        float2 a = ca.sample(lc, pa).rg, b = cb.sample(lc, pb).rg;
        float2 o = (v.w == 1 || !okB) && okA ? a : ((v.w == 2 || !okA) && okB ? b : (okA && okB ? 0.5 * (a + b) : ca.sample(lc, p).rg));
        if ((kP3 & P3_EDGE) && v.w == 0 && !inW(pa, pb, 4.0, s)) {
            float wa = edgeW(pa, s, 4.0), wb = edgeW(pb, s, 4.0);
            if (wa + wb > 0) o = (wa * a + wb * b) / (wa + wb);
        }
        f = v.w == 0 && (!kBorder || (okA && okB) || (kP3 & P3_EDGE)) ? smoothstep(fb.x, fb.y, v.z) : 0.0;
        return o;
    }
    inline float2 warpC(texture2d<float, access::sample> ca, texture2d<float, access::sample> cb, float4 v, uint2 id, float2 s, float2 fb) {
        float k; float2 o = warpC0(ca, cb, v, id, s, fb, k);
        float2 p = float2(id) + 0.5;
        if (k > 0) o = mix(o, 0.5 * (ca.sample(lc, p).rg + cb.sample(lc, p).rg), k);
        return o;
    }

    // Full-res luma: guide vectors are in guide pixels (half res): ×2 here.
    kernel void mfg_warp_y(texture2d<float, access::sample> ya [[texture(0)]], texture2d<float, access::sample> yb [[texture(1)]],
                           texture2d<half, access::read> mid [[texture(2)]], texture2d<float, access::write> yo [[texture(3)]],
                           constant int &dbg [[buffer(0)]], constant float2 &fb [[buffer(1)]], uint2 id [[thread_position_in_grid]]) {
        float2 s = float2(yo.get_width(), yo.get_height());
        if (dbg == 1) { float4 v = float4(mid.read(id / 2)); yo.write(float4(0.1 + 0.3 * v.w + (v.z > 0.06 ? 0.1 : 0)), id); return; }
        if (dbg == 2) { float4 v = float4(mid.read(id / 2)); yo.write(float4(0.5 + v.x / 50.0), id); return; }
        if (dbg == 3) { float4 v = float4(mid.read(id / 2)); yo.write(float4(0.06 + min(length(v.xy) / 30.0, 0.9)), id); return; }
        if (dbg == 4) { float4 v = float4(mid.read(id / 2)); yo.write(float4(0.06 + min(float(v.z) * 10.0, 0.9)), id); return; }
        if (dbg == 5) { float4 v = float4(mid.read(id / 4)); yo.write(float4(0.06 + min(length(v.xy) / 30.0, 0.9)), id); return; }
        if (id.x >= uint(s.x) || id.y >= uint(s.y)) return;
        float4 v = float4(mid.read(min(id / 2, uint2(mid.get_width() - 1, mid.get_height() - 1))));
        yo.write(float4(warpY(ya, yb, v, id, s, fb)), id);
    }

    // Chroma plane is at guide resolution: vectors apply as-is.
    kernel void mfg_warp_c(texture2d<float, access::sample> ca [[texture(0)]], texture2d<float, access::sample> cb [[texture(1)]],
                           texture2d<half, access::read> mid [[texture(2)]], texture2d<float, access::write> co [[texture(3)]],
                           constant float2 &fb [[buffer(1)]], uint2 id [[thread_position_in_grid]]) {
        float2 s = float2(co.get_width(), co.get_height());
        if (id.x >= uint(s.x) || id.y >= uint(s.y)) return;
        co.write(float4(warpC(ca, cb, float4(mid.read(id)), id, s, fb), 0, 1), id);
    }

    // Static overlay (P3_LOCK), per full-res luma px: the same in both frames (centre, and at least 4 of its 3×3) and
    // a strong local contrast (5×5 range): subtitle and HUD text strokes with their drop shadow, also when the learned
    // HUD mask has not seen them (subtitles come and go). Fast blurred road does not match itself with that contrast.
    // One thread per guide px: its 2×2 luma px as 4 bits in Lg (bit (y & 1) * 2 + (x & 1)).
    kernel void mfg_lock(texture2d<float, access::read> ya [[texture(0)]], texture2d<float, access::read> yb [[texture(1)]],
                         texture2d<float, access::write> Lg [[texture(2)]],
                         texture2d<float, access::read> ga [[texture(4)]], texture2d<float, access::read> gb [[texture(5)]],
                         uint2 id [[thread_position_in_grid]]) {
        if (id.x >= Lg.get_width() || id.y >= Lg.get_height()) return;
        uint bits = 0;
        // Cheap pre-check on the half-res guides: some contrast around, and the pixel itself unchanged.
        int2 gh = int2(ga.get_width() - 1, ga.get_height() - 1);
        float glo = 1, ghi = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {      // either frame (as the full-res check)
            uint2 g = uint2(clamp(int2(id) + int2(i, j), int2(0), gh));
            float r = ga.read(g).r, r2 = gb.read(g).r;
            glo = min(glo, min(r, r2)); ghi = max(ghi, max(r, r2));
        }
        if (ghi - glo < 0.5 * kLockC || ghi < 0.6 * kLockB || fabs(ga.read(id).r - gb.read(id).r) > kLockG) {
            Lg.write(float4(kP3Dbg == 9 ? 64.0 / 255.0 : 0.0), id);
            return;
        }
        // Full res: the 2×2 px share one 4×4 patch of each frame (3×3 differences and the contrast), read once.
        int2 hi = int2(ya.get_width() - 1, ya.get_height() - 1), o = int2(id) * 2;
        float D[16], lo = 1, hv = 0;
        for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
            uint2 p = uint2(clamp(o + int2(i - 1, j - 1), int2(0), hi));
            float a = ya.read(p).r, bb = yb.read(p).r;
            D[j * 4 + i] = fabs(a - bb);
            lo = min(lo, min(a, bb)); hv = max(hv, max(a, bb));
        }
        // Full-res contrast over the 4×4 patch of either frame (a white stroke over a bright road in one frame still
        // stands out against the darker road of the other).
        // Overlay text is a bright stroke with a dark drop shadow (subtitles, HUD labels): both in the patch, not
        // just any contrast (the player's car, wires against the sky keep their place on screen too, and are better
        // warped).
        if (hv - lo < kLockC || hv < kLockB || lo > kLockK) { Lg.write(float4(kP3Dbg == 9 ? 128.0 / 255.0 : 0.0), id); return; }
        for (int q = 0; q < 4; q++) {
            int qx = q & 1, qy = q >> 1;
            int same = 0;
            for (int j = 0; j <= 2; j++) for (int i = 0; i <= 2; i++) same += D[(qy + j) * 4 + qx + i] < 1.5 * kLockD ? 1 : 0;
            // The stroke and the pixels next to it agree; the soft edge of the drop shadow over the road may not.
            float dq = D[(qy + 1) * 4 + qx + 1];
            if (dq < kLockD && same >= 4) bits |= 1u << q;
            else if (dq < kLockW) bits |= 16u << q;            // weak: locked only next to a strong lock (mfg_lockdil)
        }
        Lg.write(float4(float(bits) / 255.0), id);
    }
    // Lg dilated over the 4×4 px around each px (the 2×2 a bilinear tap reads, and one more px for the
    // anti-aliased rim of a stroke, which may not lock itself): one read per source check.
    // Also the final per-px bits (Lb): the strong ones, and the weak ones (a semi-transparent stroke that differs a
    // little between frames) within 3 guide px of a strong lock, so a stroke is not locked in dashes.
    kernel void mfg_lockdil(texture2d<half, access::read> Lg [[texture(0)]], texture2d<float, access::write> Ld [[texture(1)]],
                            texture2d<float, access::write> Lb [[texture(2)]], uint2 id [[thread_position_in_grid]]) {
        int2 hi = int2(Lg.get_width() - 1, Lg.get_height() - 1);
        if (int(id.x) > hi.x || int(id.y) > hi.y) return;
        uint own = uint(round(float(Lg.read(id).r) * 255.0)), strong = 0;
        if ((own >> 4) != 0)                                  // a strong lock within 3 px (guide) along the stroke
            for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++)
                strong |= uint(round(float(Lg.read(uint2(clamp(int2(id) + int2(i, j), int2(0), hi))).r) * 255.0)) & 15u;
        uint fin = (own & 15u) | (strong != 0 ? (own >> 4) : 0u);
        bool on = false;
        for (int j = -1; j <= 2; j++) for (int i = -1; i <= 2; i++) on = on || (uint(round(float(Lg.read(uint2(clamp(int2(id) + int2(i, j), int2(0), hi))).r) * 255.0)) & 15u) != 0;
        Ld.write(float4(on || fin != 0 ? 1.0 : 0.0), id);
        Lb.write(float4(float(fin) / 255.0), id);
    }
    inline bool locked(texture2d<half, access::read> Ld, float2 p) {
        int2 hi = int2(Ld.get_width() - 1, Ld.get_height() - 1);
        return Ld.read(uint2(clamp(int2(floor(p - 0.5)), int2(0), hi))).r > 0.5;
    }
    // A moving pixel may not take its sample from a locked overlay pixel (text copied onto the road): that side is
    // dropped; both sides locked = the plain blend at the pixel.
    inline float4 lockFix(texture2d<half, access::read> L, float2 m, float4 v) {
        if (all(fabs(v.xy) < 0.25)) return v;
        bool la = locked(L, m + 0.5 * v.xy), lb = locked(L, m - 0.5 * v.xy);
        if (!la && !lb) return v;
        float2 sz = float2(L.get_width(), L.get_height());
        bool oa = inF(m + 0.5 * v.xy, sz), ob = inF(m - 0.5 * v.xy, sz);  // the side left must be inside the frame
        if ((la && lb) || (la && v.w == 1) || (lb && v.w == 2)) return float4(0, 0, 0, 0);
        // The side left is outside the frame (fast motion at the edge): its sample is pulled back to the edge (a
        // streak of the edge row, as the motion blur there looks), not the plain blend of two unrelated pixels.
        if (la && !ob) { float2 pb = clamp(m - 0.5 * v.xy, float2(0.5), sz - 0.5); return float4(2.0 * (m - pb), v.z, 2.0); }
        if (lb && !oa) { float2 pa = clamp(m + 0.5 * v.xy, float2(0.5), sz - 0.5); return float4(2.0 * (pa - m), v.z, 1.0); }
        return float4(v.xyz, la ? 2.0 : 1.0);
    }

    // Locked overlay px of this 2×2 block (bits of mfg_lock) take the plain blend; chroma too if any is locked.
    inline void lockOut(thread float *oy, thread float2 &oc, uint bits, uint2 id, texture2d<float, access::sample> ya,
                        texture2d<float, access::sample> yb, texture2d<float, access::sample> ca, texture2d<float, access::sample> cb) {
        for (uint q = 0; q < 4; q++) if ((bits >> q) & 1) {
            uint2 p = id * 2 + uint2(q & 1, q >> 1);
            oy[q] = kP3Dbg == 1 ? 1.0 : 0.5 * (ya.read(p).r + yb.read(p).r);
        }
        oc = 0.5 * (ca.read(id).rg + cb.read(id).rg);
    }

    // Upsample and warp fused (the shipping path): one thread per guide pixel picks its vector and writes its 2×2
    // luma block and its CbCr texel, so the per-pixel field never goes through memory.
    kernel void mfg_upwarp(texture2d<float, access::sample> ga [[texture(0)]], texture2d<float, access::sample> gb [[texture(1)]],
                           texture2d<half, access::read> q [[texture(2)]],
                           texture2d<float, access::sample> ya [[texture(3)]], texture2d<float, access::sample> yb [[texture(4)]],
                           texture2d<float, access::write> yo [[texture(5)]],
                           texture2d<float, access::sample> ca [[texture(6)]], texture2d<float, access::sample> cb [[texture(7)]],
                           texture2d<float, access::write> co [[texture(8)]], texture2d<half, access::read> L [[texture(9)]],
                           texture2d<half, access::read> Lb [[texture(10)]], texture2d<half, access::read> Lraw [[texture(12)]], texture2d<half, access::read> alt [[texture(11)]],
                           constant float &G [[buffer(0)]], constant float2 &fb [[buffer(1)]], uint2 id [[thread_position_in_grid]]) {
        float2 cs = float2(co.get_width(), co.get_height());
        if (id.x >= uint(cs.x) || id.y >= uint(cs.y)) return;
        float2 s = float2(yo.get_width(), yo.get_height());
        // Static overlay (P3_LOCK): its pixels are never warped, the plain blend of both frames.
        if (kP3Dbg == 9) {
            uint r = uint(round(float(Lraw.read(id).r) * 255.0));
            for (uint q = 0; q < 4; q++) yo.write(float4(r == 64 ? 0.3 : r == 128 ? 0.45 : ((r >> q) & 1) ? 1.0 : (((r >> (q + 4)) & 1) ? 0.6 : 0.1)), id * 2 + uint2(q & 1, q >> 1));
            co.write(float4(0.5, 0.5, 0, 1), id); return;
        }
        bool lk = false, lkAll = false; uint lbits = 0;
        if ((kP3 & P3_LOCK) && L.read(id).r > 0.5) {             // dilated mask first: most pixels stop here
            lbits = uint(round(float(Lb.read(id).r) * 255.0));
            lk = lbits != 0; lkAll = lbits == 15;
        }
        if (lkAll) {
            for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) yo.write(float4(kP3Dbg == 1 ? 1.0 : 0.5 * (ya.read(id * 2 + uint2(i, j)).r + yb.read(id * 2 + uint2(i, j)).r)), id * 2 + uint2(i, j));
            co.write(float4(0.5 * (ca.read(id).rg + cb.read(id).rg), 0, 1), id);
            return;
        }
        if (kP3 & P3_SOFT) {
            // Soft upsample (P3_SOFT): where the 4 surrounding cells disagree and their vectors match this pixel about
            // equally well, a hard per-pixel pick leaves the cell grid's 8-px staircase in the picture (each side
            // samples different content). Each cell's warp is weighted by its bilinear position weight and by how
            // well its vector matches here (exp(-(c - c_min) / kSoftS)): a clear winner stays a hard edge (object
            // outlines), a near tie becomes a smooth cross-fade (overlapped block motion compensation).
            int2 hi = int2(q.get_width() - 1, q.get_height() - 1);
            float2 m = float2(id) + 0.5, u = m / G - 0.5;
            int2 c0 = int2(floor(u));
            float2 fr = clamp(u - float2(c0), 0.0, 1.0);
            // Static-HUD cells (P3_HUDPX) also offer the moving vector they gave up: the learned mask is per 8-px
            // cell, so the road under a subtitle line would otherwise keep the zero vector (a blended block). The
            // pixel decides by its own match; locked text and real HUD pixels match the zero vector best.
            // Entries: distinct (vector, fade cost, side) with their summed spatial weight. kSoftK = 1: the 3×3 cells
            // around the pixel with a tent 1.5 cells wide, so a vector that matches well reaches past its own cell
            // (the edge between two vector regions follows the match, not the cell grid).
            float4 v[10]; float bw[10]; int n = 0; bool side = false, zdiff = false;
            int K = kSoftK ? 3 : 2;
            int2 cbase = kSoftK ? int2(floor(u + 0.5)) - 1 : c0;
            for (int jj = 0; jj < K; jj++) for (int ii = 0; ii < K; ii++) {
                int2 cc = cbase + int2(ii, jj);
                float wg;
                if (kSoftK) { float2 dd = fabs(u - float2(cc)); wg = max(0.0, 1.0 - dd.x / 1.5) * max(0.0, 1.0 - dd.y / 1.5); }
                else wg = (ii ? fr.x : 1.0 - fr.x) * (jj ? fr.y : 1.0 - fr.y);
                if (wg <= 0) continue;
                uint2 cl = uint2(clamp(cc, int2(0), hi));
                for (int h = 0; h < 2; h++) {
                    float4 e;
                    if (h == 0) e = float4(q.read(cl));
                    else {
                        if (!(kP3 & P3_HUDPX)) break;
                        half4 a = alt.read(cl);
                        if (a.w <= 0) break;
                        e = float4(float2(a.xy), float(a.z), 0);
                    }
                    side = side || e.w != 0;
                    int r = -1;
                    for (int t = 0; t < n; t++) if (all(fabs(v[t].xy - e.xy) < kSoftM) && v[t].w == e.w) { r = t; break; }
                    // Same vector (within kSoftM px: warping with the mean vector ≈ blending the two warps): one entry,
                    // vector and fade cost the weighted means of the cells' (no 8-px fade steps).
                    if (r >= 0) {
                        zdiff = zdiff || any(v[r].xyz != e.xyz);
                        v[r].xyz = (v[r].xyz * bw[r] + e.xyz * wg) / (bw[r] + wg); bw[r] += wg;
                    }
                    else if (n < 10) { v[n] = e; bw[n] = wg; n++; }
                }
            }
            bool same = n == 1 && (!zdiff || !kSoftZS);
            if (kP3Dbg == 3) { for (uint q2 = 0; q2 < 4; q2++) yo.write(float4(n > 4 ? 1.0 : 0.0), id * 2 + uint2(q2 & 1, q2 >> 1)); co.write(float4(0.5, 0.5, 0, 1), id); return; }
            if (!same && (!side || kSoftSide)) {
                float2 gsz = float2(ga.get_width(), ga.get_height());
                float w[10], cmin = 1e9; int z0 = -1;
                if (n > 1) for (int t = 0; t < n; t++) {
                    float2 d = v[t].xy;
                    float c;
                    if (kBorder && !(inF(m + 0.5 * d, gsz) && inF(m - 0.5 * d, gsz))) c = (kP3 & P3_EDGE) ? max(0.04, v[t].z) : 0.04;
                    else c = (kSoftC ? sad9(ga, gb, m + 0.5 * d, m - 0.5 * d) : upCost(ga, gb, m, d, 1e9)) - (all(fabs(d) < 0.25) ? 0.004 : 0.0);
                    w[t] = c; cmin = min(cmin, c);
                    if (all(fabs(d) < 0.25) && v[t].w == 0) z0 = t;
                }
                // A pixel that did not change between the frames and wins on the zero vector (static HUD before the
                // mask has learned it, a still background) keeps it alone: the soft mix would pull in moving content.
                bool zh = n > 1 && z0 >= 0 && w[z0] <= cmin && w[z0] < kSoftH;
                float ws = 0;
                for (int t = 0; t < n; t++) { w[t] = zh ? (t == z0 ? 1.0 : 0.0) : n > 1 ? bw[t] * exp(-(w[t] - cmin) / kSoftS) : 1.0; ws += w[t]; }
                // Σ k_t ((1 - f_t) o_t + f_t blend) = Σ k_t (1 - f_t) o_t + (Σ k_t f_t) blend: the plain blend once.
                float oy[4] = { 0, 0, 0, 0 }; float2 oc = 0; float kn = 0, kf = 0, kfc = 0, dvx = 0, dfz = 0;
                for (int t = 0; t < n; t++) {
                    float k = w[t] / ws;
                    if (k < 0.02) continue;
                    kn += k;
                    float4 v0 = v[t];
                    if ((kP3 & P3_LOCK) && kLockSrc) v[t] = lockFix(L, m, v[t]);
                    dvx += k * (kP3Dbg == 8 ? float(any(v0 != v[t])) * 50.0 : v[t].y); dfz += k * v[t].z;
                    float f = 0;
                    for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) {
                        float o = warpY0(ya, yb, v[t], id * 2 + uint2(i, j), s, fb, f);
                        oy[j * 2 + i] += k * (1.0 - f) * o;
                    }
                    kf += k * f;
                    float fc = 0;
                    float2 c2 = warpC0(ca, cb, v[t], id, cs, fb, fc);
                    oc += k * (1.0 - fc) * c2; kfc += k * fc;
                }
                if (kf > 0) for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) {
                    float2 p = float2(id * 2 + uint2(i, j)) + 0.5;
                    oy[j * 2 + i] += kf * 0.5 * (ya.sample(lc, p).r + yb.sample(lc, p).r);
                }
                if (kfc > 0) { float2 p = float2(id) + 0.5; oc += kfc * 0.5 * (ca.sample(lc, p).rg + cb.sample(lc, p).rg); }
                if (kP3Dbg >= 4) { for (uint q2 = 0; q2 < 4; q2++) yo.write(float4(kP3Dbg == 4 || kP3Dbg == 8 ? 0.5 + dvx / kn / 50.0 : kP3Dbg == 6 ? 0.5 + dvx / kn / 200.0 : kP3Dbg == 7 ? 0.3 : min(dfz / kn * 10.0, 1.0)), id * 2 + uint2(q2 & 1, q2 >> 1)); co.write(float4(0.5, 0.5, 0, 1), id); return; }
                for (uint q2 = 0; q2 < 4; q2++) oy[q2] = kP3Dbg == 2 ? 1.0 : oy[q2] / kn;
                oc /= kn;
                if (lk) lockOut(oy, oc, lbits, id, ya, yb, ca, cb);
                for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) yo.write(float4(oy[j * 2 + i]), id * 2 + uint2(i, j));
                co.write(float4(oc, 0, 1), id);
                return;
            }
        }
        float4 v = float4(upPick(ga, gb, q, G, id));
        if ((kP3 & P3_LOCK) && kLockSrc) v = lockFix(L, float2(id) + 0.5, v);
        float hy[4];
        if (kP3Dbg >= 4) { for (uint q2 = 0; q2 < 4; q2++) yo.write(float4(kP3Dbg == 4 ? 0.5 + v.y / 50.0 : kP3Dbg == 6 ? 0.5 + v.y / 200.0 : kP3Dbg == 7 ? 0.7 + 0.1 * v.w : min(v.z * 10.0, 1.0)), id * 2 + uint2(q2 & 1, q2 >> 1)); co.write(float4(0.5, 0.5, 0, 1), id); return; }
        for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) {
            uint2 p = id * 2 + uint2(i, j);
            hy[j * 2 + i] = warpY(ya, yb, v, p, s, fb);
        }
        float2 hc = warpC(ca, cb, v, id, cs, fb);
        if (lk) lockOut(hy, hc, lbits, id, ya, yb, ca, cb);
        for (uint j = 0; j < 2; j++) for (uint i = 0; i < 2; i++) yo.write(float4(hy[j * 2 + i]), id * 2 + uint2(i, j));
        co.write(float4(hc, 0, 1), id);
    }
    """
}
