import Foundation
import ARKit
import CoreImage
import ImageIO
import UIKit

struct CapturedFrame: Codable {
    let file: String
    let t: Double
    let transform: [Float]      // ARKit 相機位姿（4x4，column-major，單位：公尺）
    let tracking: String
}

struct CaptureMeta: Codable {
    var jobId: String
    var imageWidth: Int
    var imageHeight: Int
    var orientation: String
    var frames: [CapturedFrame]
}

/// 錄影：每 0.5 秒存一張影格，同時記錄 ARKit 算出的手機位置（用來換算真實尺寸）
final class CaptureManager: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published var isRecording = false
    @Published var frameCount = 0
    @Published var trackingText = "準備中…"
    @Published var trackingGood = false

    private let saveQueue = DispatchQueue(label: "obj3d.capture.save")
    private let ciContext = CIContext(options: nil)
    private let interval: TimeInterval = 0.35
    private let scale: CGFloat = 2.0 / 3.0           // 1440x1920 → 960x1280
    private var lastSavedTime: TimeInterval = 0      // 主執行緒
    private var inFlight = 0                         // 主執行緒
    private var meta: CaptureMeta?                   // 只在 saveQueue 上存取
    private var captureDir: URL?                     // 只在 saveQueue 上存取

    override init() {
        super.init()
        session.delegate = self
    }

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            trackingText = "這台裝置不支援 ARKit"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func pauseSession() {
        session.pause()
    }

    func beginRecording() {
        let jobId = String(UUID().uuidString.prefix(8)).lowercased()
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("captures/capture_\(jobId)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir.appendingPathComponent("frames"),
                                                 withIntermediateDirectories: true)
        saveQueue.sync {
            self.captureDir = dir
            self.meta = CaptureMeta(jobId: jobId, imageWidth: 0, imageHeight: 0,
                                    orientation: "portrait", frames: [])
        }
        frameCount = 0
        lastSavedTime = 0
        isRecording = true
    }

    /// 結束錄影：等所有影格寫完後寫入 meta.json
    func endRecording(completion: @escaping (URL?, String?, Int) -> Void) {
        isRecording = false
        saveQueue.async {
            guard let meta = self.meta, let dir = self.captureDir else {
                DispatchQueue.main.async { completion(nil, nil, 0) }
                return
            }
            do {
                let data = try JSONEncoder().encode(meta)
                try data.write(to: dir.appendingPathComponent("meta.json"))
                let n = meta.frames.count
                DispatchQueue.main.async { completion(n > 0 ? dir : nil, meta.jobId, n) }
            } catch {
                DispatchQueue.main.async { completion(nil, nil, 0) }
            }
        }
    }

    // MARK: - ARSessionDelegate（預設在主執行緒呼叫）

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateTracking(frame.camera.trackingState)
        guard isRecording else { return }
        guard case .normal = frame.camera.trackingState else { return }
        guard frame.timestamp - lastSavedTime >= interval, inFlight < 3 else { return }
        lastSavedTime = frame.timestamp
        inFlight += 1

        let pixelBuffer = frame.capturedImage
        let timestamp = frame.timestamp
        let m = frame.camera.transform
        let transform: [Float] = [
            m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
            m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
            m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
            m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w,
        ]
        saveQueue.async { [weak self] in
            guard let self = self else { return }
            let count = self.saveFrame(pixelBuffer, timestamp: timestamp, transform: transform)
            DispatchQueue.main.async {
                self.inFlight -= 1
                if let c = count { self.frameCount = c }
            }
        }
    }

    /// 在 saveQueue 上執行：轉成直式 JPEG 存檔
    private func saveFrame(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval, transform: [Float]) -> Int? {
        guard var meta = self.meta, let dir = self.captureDir else { return nil }
        let rotated = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let scaled = rotated.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let image = scaled.transformed(by: CGAffineTransform(translationX: -scaled.extent.origin.x,
                                                            y: -scaled.extent.origin.y))
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let data = ciContext.jpegRepresentation(
                of: image, colorSpace: colorSpace,
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.88])
        else { return nil }
        let name = String(format: "frames/%05d.jpg", meta.frames.count)
        do {
            try data.write(to: dir.appendingPathComponent(name))
        } catch {
            return nil
        }
        meta.frames.append(CapturedFrame(file: name, t: timestamp, transform: transform, tracking: "normal"))
        meta.imageWidth = Int(image.extent.width.rounded())
        meta.imageHeight = Int(image.extent.height.rounded())
        self.meta = meta
        return meta.frames.count
    }

    private func updateTracking(_ state: ARCamera.TrackingState) {
        var text = ""
        var good = false
        switch state {
        case .normal:
            good = true
            text = isRecording ? "錄製中：慢慢繞著物體走" : "追蹤良好，可以開始錄影"
        case .notAvailable:
            text = "追蹤無法使用"
        case .limited(let reason):
            switch reason {
            case .initializing: text = "初始化中：左右移動一下手機"
            case .excessiveMotion: text = "移動太快，請放慢"
            case .insufficientFeatures: text = "畫面特徵不足，請照亮或換個角度"
            case .relocalizing: text = "重新定位中…"
            @unknown default: text = "追蹤受限"
            }
        }
        if text != trackingText { trackingText = text }
        if good != trackingGood { trackingGood = good }
    }
}
