import Foundation

/// 版本資訊：App 版本、GitHub 編譯編號、commit、內含的 Kaggle 運算程式版本
enum AppInfo {
    static var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }
    static var build: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
    }
    static var commit: String {
        Bundle.main.object(forInfoDictionaryKey: "OBJ3DCommit") as? String ?? "?"
    }
    /// 從 App 內附的 runner.py 讀出 RUNNER_VERSION
    static var runnerVersion: String {
        guard let url = Bundle.main.url(forResource: "runner", withExtension: "py"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "?" }
        for line in text.split(separator: "\n").prefix(40) where line.hasPrefix("RUNNER_VERSION") {
            let parts = line.split(separator: "\"")
            if parts.count >= 2 { return String(parts[1]) }
        }
        return "?"
    }
    static var summary: String {
        "App \(version)（build \(build)・\(commit)）・運算程式 \(runnerVersion)"
    }
}
