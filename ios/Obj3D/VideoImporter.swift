import Foundation
import AVFoundation
import UIKit
import SwiftUI
import UniformTypeIdentifiers

/// 從「照片」選到的影片（PhotosPicker 傳來的檔案）
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("import_\(UUID().uuidString).\(ext)")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest)
        }
    }
}

/// 把一般影片轉成 App 的錄影存檔：平均擷取影格（最長邊 1280），標記為 "video" 模式
enum VideoImporter {
    static func importVideo(_ url: URL, fps: Double, maxFrames: Int = 300,
                            progress: @escaping (Double) -> Void) async throws -> (URL, Int) {
        defer { try? FileManager.default.removeItem(at: url) }
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration >= 4 else {
            throw KaggleError(message: "影片太短（至少需要約 10 秒，建議 40~90 秒）")
        }
        let count = min(max(Int(duration * fps), 12), maxFrames)

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true            // 依拍攝方向轉正
        gen.maximumSize = CGSize(width: 1280, height: 1280)
        let tol = CMTime(seconds: 0.05, preferredTimescale: 600)
        gen.requestedTimeToleranceBefore = tol
        gen.requestedTimeToleranceAfter = tol

        let jobId = String(UUID().uuidString.prefix(8)).lowercased()
        let dir = CaptureStore.root.appendingPathComponent("capture_\(jobId)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("frames"), withIntermediateDirectories: true)

        let identity: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        var frames: [CapturedFrame] = []
        var w = 0, h = 0
        for k in 0..<count {
            try Task.checkCancellation()
            let t = CMTime(seconds: duration * (Double(k) + 0.5) / Double(count), preferredTimescale: 600)
            guard let cg = try? await gen.image(at: t).image,
                  let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.9) else { continue }
            let name = String(format: "frames/%05d.jpg", frames.count)
            try data.write(to: dir.appendingPathComponent(name))
            frames.append(CapturedFrame(file: name, t: t.seconds, transform: identity, tracking: "imported"))
            w = cg.width; h = cg.height
            progress(Double(k + 1) / Double(count))
        }
        guard frames.count >= 12 else {
            try? FileManager.default.removeItem(at: dir)
            throw KaggleError(message: "無法從影片擷取足夠的影格")
        }
        let meta = CaptureMeta(jobId: jobId, imageWidth: w, imageHeight: h,
                               orientation: h >= w ? "portrait" : "landscape", frames: frames, mode: "video",
                               scale_length_cm: nil, scale_marks: nil, object_point: nil)
        try JSONEncoder().encode(meta).write(to: dir.appendingPathComponent("meta.json"))
        return (dir, frames.count)
    }
}
