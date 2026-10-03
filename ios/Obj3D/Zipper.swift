import Foundation

/// 用系統內建功能把資料夾壓成 zip（不需要第三方套件）
enum Zipper {
    static func zip(directory: URL) throws -> URL {
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent("capture.zip")
        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: directory, options: [.forUploading], error: &coordError) { zipURL in
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: zipURL, to: dest)
            } catch {
                copyError = error
            }
        }
        if let e = coordError { throw e }
        if let e = copyError { throw e }
        return dest
    }
}
