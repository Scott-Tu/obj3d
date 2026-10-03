import Foundation
import UIKit

enum AppPaths {
    static var results: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("results")
    }
}

struct ResultMeta: Codable {
    let jobId: String
    let status: String
    let error: String?
    let message: String?
    let size_cm: [Double]?
    let watertight: Bool?
    let volume_cm3: Double?
    let has_base: Bool?
    let scale_residual_cm: Double?
    let elapsed_s: Double?
    let warnings: [String]?
}

struct JobResult {
    let jobId: String
    let dir: URL
    let meta: ResultMeta

    var previewURL: URL { dir.appendingPathComponent("preview.bin") }
    var shareFiles: [URL] {
        ["model.glb", "model_mm.stl", "model_mm.ply"]
            .map { dir.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func load(jobId: String) -> JobResult? {
        let dir = AppPaths.results.appendingPathComponent(jobId)
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("result_meta.json")),
              let meta = try? JSONDecoder().decode(ResultMeta.self, from: data) else { return nil }
        return JobResult(jobId: jobId, dir: dir, meta: meta)
    }
}

/// 整個流程：壓縮 → 上傳 Kaggle → 啟動運算 → 等待 → 下載結果
@MainActor
final class JobManager: ObservableObject {
    enum Phase { case idle, captured, working, done, failed }

    @Published var phase: Phase = .idle
    @Published var status = "按「錄影」開始拍攝"
    @Published var detail = ""
    @Published var captureURL: URL?
    @Published var jobId: String?
    @Published var result: JobResult?
    @Published var canResume = false

    private var task: Task<Void, Never>?
    let datasetSlug = "obj3d-capture-data"
    let kernelSlug = "obj3d-runner"

    var busy: Bool { phase == .working }

    init() {
        let d = UserDefaults.standard
        if let last = d.string(forKey: "lastResultJobId"), let r = JobResult.load(jobId: last) {
            result = r
            phase = .done
            status = "上一次的模型"
        }
        if let pending = d.string(forKey: "pendingJobId") {
            jobId = pending
            canResume = true
            status = "上一次的工作還沒取回結果"
            detail = "按「繼續查詢」取回結果"
        }
    }

    func setCapture(url: URL, jobId: String, frames: Int) {
        captureURL = url
        self.jobId = jobId
        canResume = false
        if frames < 12 {
            phase = .idle
            status = "只錄到 \(frames) 張影格，太少了"
            detail = "請慢慢繞物體走一圈以上，至少錄 20 秒"
            return
        }
        phase = .captured
        status = "已錄好 \(frames) 張影格"
        detail = "按「生成 3D 模型」上傳到 Kaggle 運算"
    }

    func generate() {
        task?.cancel()
        task = Task { await runGenerate() }
    }

    func resume() {
        guard let id = jobId else { return }
        task?.cancel()
        task = Task { await runResume(jobId: id) }
    }

    func stopWaiting() {
        task?.cancel()
    }

    // MARK: -

    private func setBusy(_ on: Bool) {
        UIApplication.shared.isIdleTimerDisabled = on      // 等待時螢幕不自動關閉
    }

    private func runGenerate() async {
        guard let dir = captureURL, let jobId = jobId else { return }
        guard let client = KaggleClient.fromSettings() else {
            fail("請先到右上角「設定」填入 Kaggle 使用者名稱與 API 金鑰")
            return
        }
        phase = .working
        setBusy(true)
        defer { setBusy(false) }
        do {
            status = "壓縮拍攝資料…"; detail = ""
            let zipURL = try await Task.detached { try Zipper.zip(directory: dir) }.value
            let attrs = try? FileManager.default.attributesOfItem(atPath: zipURL.path)
            let mb = ((attrs?[.size] as? NSNumber)?.doubleValue ?? 0) / 1e6
            status = "上傳到 Kaggle…"; detail = String(format: "%.0f MB", mb)
            let token = try await client.uploadBlob(file: zipURL)
            status = "建立 Kaggle 資料集…"; detail = ""
            try await client.pushDataset(slug: datasetSlug, title: "obj3d capture data", fileToken: token)
            status = "等待 Kaggle 處理資料…"
            try await Task.sleep(nanoseconds: 20_000_000_000)
            try await client.waitDatasetReady(slug: datasetSlug, timeout: 1200)
            UserDefaults.standard.set(jobId, forKey: "pendingJobId")
            try await startKernel(client: client, jobId: jobId)
            try await poll(client: client, jobId: jobId)
        } catch is CancellationError {
            paused()
        } catch {
            fail(error.localizedDescription)
        }
    }

    private func runResume(jobId: String) async {
        guard let client = KaggleClient.fromSettings() else {
            fail("請先到「設定」填入 Kaggle 帳號")
            return
        }
        phase = .working
        canResume = false
        setBusy(true)
        defer { setBusy(false) }
        do {
            try await poll(client: client, jobId: jobId)
        } catch is CancellationError {
            paused()
        } catch {
            fail(error.localizedDescription)
        }
    }

    private func startKernel(client: KaggleClient, jobId: String) async throws {
        guard let url = Bundle.main.url(forResource: "runner", withExtension: "py"),
              let template = try? String(contentsOf: url, encoding: .utf8) else {
            throw KaggleError(message: "App 內找不到 runner.py")
        }
        let script = template.replacingOccurrences(of: "__JOB_ID__", with: jobId)
        let shape = UserDefaults.standard.string(forKey: "machineShape") ?? "NvidiaTeslaT4"
        status = "啟動 Kaggle 運算…"
        try await client.pushKernel(slug: kernelSlug, title: "obj3d runner", script: script,
                                    datasetSlug: datasetSlug, machineShape: shape)
    }

    private func poll(client: KaggleClient, jobId: String) async throws {
        let start = Date()
        var restarts = 0
        while true {
            try Task.checkCancellation()
            let elapsed = Date().timeIntervalSince(start)
            if elapsed > 4 * 3600 { throw KaggleError(message: "等待超過 4 小時，請到 Kaggle 網站查看 obj3d-runner") }
            let (st, failure) = try await client.kernelStatus(slug: kernelSlug)
            let s = st.lowercased()
            if s.contains("queued") { status = "Kaggle 排隊中…" }
            else if s.contains("running") { status = "Kaggle 運算中…" }
            else { status = "Kaggle 狀態：\(st)" }
            detail = "已等待 \(Int(elapsed / 60)) 分鐘（通常 20~60 分鐘），可以先離開 App，回來後按「繼續查詢」"

            if s.contains("complete") || s.contains("error") || s.contains("cancel") {
                let files = try await client.kernelOutputs(slug: kernelSlug)
                if let metaFile = files.first(where: { $0.name.hasSuffix("result_meta.json") }) {
                    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("result_meta.json")
                    try await client.download(metaFile.url, to: tmp)
                    if let meta = try? JSONDecoder().decode(ResultMeta.self, from: Data(contentsOf: tmp)),
                       meta.jobId == jobId {
                        if meta.status == "ok" {
                            try await fetchResult(client: client, files: files, jobId: jobId)
                            return
                        }
                        if meta.error == "stale_capture" && restarts < 3 {
                            restarts += 1
                            status = "Kaggle 資料集還沒更新，1 分鐘後重新啟動…"
                            try await Task.sleep(nanoseconds: 60_000_000_000)
                            try await startKernel(client: client, jobId: jobId)
                            continue
                        }
                        throw KaggleError(message: "運算失敗：\(meta.message ?? meta.error ?? "未知錯誤")")
                    }
                }
                // 還是上一次的輸出：繼續等；若確定失敗則停止
                if s.contains("error") && elapsed > 180 {
                    throw KaggleError(message: "Kaggle 執行失敗：\(failure ?? "請到 kaggle.com 查看 obj3d-runner 的紀錄")")
                }
            }
            try await Task.sleep(nanoseconds: 30_000_000_000)
        }
    }

    private func fetchResult(client: KaggleClient, files: [OutputFile], jobId: String) async throws {
        status = "下載模型…"
        let dir = AppPaths.results.appendingPathComponent(jobId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let wanted = ["result_meta.json", "preview.bin", "model.glb", "model_mm.stl", "model_mm.ply"]
        for name in wanted {
            if let f = files.first(where: { ($0.name as NSString).lastPathComponent == name }) {
                detail = name
                try await client.download(f.url, to: dir.appendingPathComponent(name))
            }
        }
        guard let r = JobResult.load(jobId: jobId) else { throw KaggleError(message: "結果檔案不完整") }
        result = r
        phase = .done
        canResume = false
        UserDefaults.standard.removeObject(forKey: "pendingJobId")
        UserDefaults.standard.set(jobId, forKey: "lastResultJobId")
        status = "完成！"
        var lines: [String] = []
        if let s = r.meta.size_cm, s.count == 3 {
            lines.append(String(format: "尺寸：寬 %.1f × 深 %.1f × 高 %.1f cm", s[0], s[1], s[2]))
        }
        if r.meta.watertight == true { lines.append("封閉無破孔") }
        if let w = r.meta.warnings, !w.isEmpty { lines.append("注意：" + w.joined(separator: "；")) }
        detail = lines.joined(separator: "\n")
    }

    private func paused() {
        phase = captureURL != nil ? .captured : .idle
        canResume = UserDefaults.standard.string(forKey: "pendingJobId") != nil
        status = canResume ? "已暫停查詢" : "已取消"
        detail = canResume ? "Kaggle 會繼續運算，稍後按「繼續查詢」取回結果" : ""
    }

    private func fail(_ message: String) {
        phase = .failed
        status = "發生錯誤"
        detail = message
        canResume = UserDefaults.standard.string(forKey: "pendingJobId") != nil
    }
}
