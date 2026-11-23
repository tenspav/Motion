// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Motion",
    platforms: [.macOS("26.0"), .iOS("26.0")],
    products: [
        .library(name: "Motion", targets: ["Motion"]),
        .executable(name: "motion-bench", targets: ["motion-bench"]),
    ],
    targets: [
        .target(name: "Motion", path: "Sources/Motion"),
        .executableTarget(name: "motion-bench", dependencies: ["Motion"], path: "Sources/motion-bench"),
    ],
    swiftLanguageModes: [.v5]
)
