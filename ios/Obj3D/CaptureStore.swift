import Foundation
import UIKit

/// 已存檔的錄影（Documents/captures/capture_xxxx），更新 App 後也能直接載入，不用重拍
struct CaptureInfo: Identifiable {
    let url: URL
    let frames: Int
    let date: Date
    let mode: String
    var id: String { url.path }
}

enum CaptureStore {
    static var root: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures")
    }

    static func list() -> [CaptureInfo] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.creationDateKey],
                                                      options: [.skipsHiddenFiles]) else { return [] }
        var out: [CaptureInfo] = []
        for d in dirs {
            let metaURL = d.appendingPathComponent("meta.json")
            guard let data = try? Data(contentsOf: metaURL),
                  let meta = try? JSONDecoder().decode(CaptureMeta.self, from: data),
                  !meta.frames.isEmpty else { continue }
            let date = (try? d.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            out.append(CaptureInfo(url: d, frames: meta.frames.count, date: date, mode: meta.mode ?? "orbit"))
        }
        return out.sorted { $0.date > $1.date }
    }

    static func delete(_ c: CaptureInfo) {
        try? FileManager.default.removeItem(at: c.url)
    }

    static func thumbnail(_ c: CaptureInfo) -> UIImage? {
        let first = c.url.appendingPathComponent("frames/00000.jpg")
        guard let img = UIImage(contentsOfFile: first.path) else { return nil }
        return img.preparingThumbnail(of: CGSize(width: 90, height: 120)) ?? img
    }

    /// 每次生成都換一個新的工作代號（同一段錄影可以重複生成）
    static func assignNewJobId(_ dir: URL) throws -> String {
        let url = dir.appendingPathComponent("meta.json")
        var meta = try JSONDecoder().decode(CaptureMeta.self, from: Data(contentsOf: url))
        let newId = String(UUID().uuidString.prefix(8)).lowercased()
        meta.jobId = newId
        try JSONEncoder().encode(meta).write(to: url)
        return newId
    }
}
