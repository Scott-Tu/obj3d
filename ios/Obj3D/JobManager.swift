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
    let target_method: String?
    let bad_mask_frames: Int?
    let frames_used: Int?
    let runner_version: String?
    let tsdf: Bool?
    let textured: Bool?
    let params_summary: String?
    let gs: Bool?
    let gs_count: Int?
    let gs_mb: Double?
    let scale_source: String?
    let ruler_vs_arkit: Double?
    let use_arkit: Bool?
    let smooth_level: String?
}

struct JobResult {
    let jobId: String
    let dir: URL
    let meta: ResultMeta

    var previewURL: URL { dir.appendingPathComponent("preview.bin") }
    var splatURL: URL? {
        let u = dir.appendingPathComponent("model.splat")
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }
    var shareFiles: [URL] {
        ["model.glb", "model_mm.stl", "model_mm.ply", "model.splat"]
            .map { dir.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    var diagnosticImages: [URL] {
        ["diag_masks.jpg", "diag_cameras.png", "model.splat"]
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
    @Published var showMarkScale = false

    /// 目前錄影的設定（拍攝方式、是否已標記比例尺）
    func currentMeta() -> CaptureMeta? {
        guard let dir = captureURL,
              let data = try? Data(contentsOf: dir.appendingPathComponent("meta.json")) else { return nil }
        return try? JSONDecoder().decode(CaptureMeta.self, from: data)
    }
    /// 需要比例尺：物體旋轉、從「照片」匯入的影片，或關閉 ARKit 時的手機繞物體
    var captureIsTurntable: Bool {
        let mode = currentMeta()?.mode ?? "orbit"
        return ["turntable", "video"].contains(mode) || (mode == "orbit" && DevParams.value(forKey: "use_arkit") == 0)
    }
    var minMarksNeeded: Int { (currentMeta()?.mode ?? "orbit") == "orbit" ? 2 : 1 }
    var captureHasMarks: Bool { (currentMeta()?.scale_marks ?? []).count >= minMarksNeeded }

    func marksSaved() {
        objectWillChange.send()
        if captureIsTurntable && captureHasMarks {
            phase = .captured
            status = "比例尺已標記"
            detail = "可以按「生成 3D 模型」"
        }
    }

    private var task: Task<Void, Never>?
    let datasetSlug = "obj3d-capture-data"
    let kernelSlug = "obj3d-runner"

    var busy: Bool { phase == .working }

    private var videoObserver: NSObjectProtocol?

    init() {
        videoObserver = NotificationCenter.default.addObserver(forName: .obj3dVideoSaved, object: nil, queue: .main) { [weak self] note in
            let ok = (note.object as? Bool) ?? false
            Task { @MainActor in
                guard let self = self else { return }
                let msg = ok ? "影片已另存到「照片」" : "影片未能存到「照片」（請到 設定 → 隱私權 → 照片 允許「3D 掃描」加入照片）"
                self.detail = self.detail.isEmpty ? msg : self.detail + "\n" + msg
            }
        }
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
        if let latest = CaptureStore.list().first {            // 自動載入最近一次的錄影
            captureURL = latest.url
            if phase != .done && !canResume {
                phase = .captured
                status = "已載入上次的錄影（\(latest.frames) 張）"
                detail = "可以直接按「生成 3D 模型」"
            }
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
        if captureIsTurntable && !captureHasMarks {
            status = currentMeta()?.mode == "video" ? "已匯入 \(frames) 張影格" : "已錄好 \(frames) 張影格"
            detail = "建議標記比例尺兩端（至少 \(minMarksNeeded) 張畫面）；不標記也能生成，但沒有真實尺寸"
            showMarkScale = true
        } else if !captureHasMarks {
            detail = "建議標記比例尺兩端（2~4 張畫面），尺寸會更準；也可以直接生成"
            showMarkScale = true
        }
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

    /// 網路不穩時自動重試（Kaggle 回傳的錯誤不重試）
    private func withRetry<T>(_ label: String, attempts: Int = 4, _ op: () async throws -> T) async throws -> T {
        var lastError: Error = KaggleError(message: "網路錯誤")
        for k in 0..<attempts {
            do {
                return try await op()
            } catch is CancellationError {
                throw CancellationError()
            } catch let e as KaggleError {
                throw e
            } catch {
                lastError = error
                if k == attempts - 1 { break }
                status = "\(label)：網路中斷，\(5 + 10 * k) 秒後重試（\(k + 1)/\(attempts - 1)）"
                try await Task.sleep(nanoseconds: UInt64(5 + 10 * k) * 1_000_000_000)
            }
        }
        throw KaggleError(message: "\(label)失敗：網路連線不穩（\(lastError.localizedDescription)）。建議改用 Wi-Fi，上傳時保持 App 在前景。")
    }

    private func setBusy(_ on: Bool) {
        UIApplication.shared.isIdleTimerDisabled = on      // 等待時螢幕不自動關閉
    }

    private func runGenerate() async {
        guard let dir = captureURL else { return }
        guard let client = KaggleClient.fromSettings() else {
            fail("請先到右上角「設定」填入 Kaggle 使用者名稱與 API 金鑰")
            return
        }
        phase = .working
        setBusy(true)
        defer { setBusy(false) }
        if captureIsTurntable && !captureHasMarks {
            detail = "沒有比例尺、也沒有使用手機位置：模型會是相對比例（沒有真實尺寸）"
        }
        do {
            let jobId = try CaptureStore.assignNewJobId(dir)     // 同一段錄影可以重複生成
            self.jobId = jobId
            status = "壓縮拍攝資料…"; detail = ""
            let zipURL = try await Task.detached { try Zipper.zip(directory: dir) }.value
            let attrs = try? FileManager.default.attributesOfItem(atPath: zipURL.path)
            let mb = ((attrs?[.size] as? NSNumber)?.doubleValue ?? 0) / 1e6
            // 讓 App 切到背景時還能多爭取一些時間把上傳做完
            var bgTask: UIBackgroundTaskIdentifier = .invalid
            bgTask = UIApplication.shared.beginBackgroundTask(withName: "obj3d-upload") {
                UIApplication.shared.endBackgroundTask(bgTask)
                bgTask = .invalid
            }
            defer { if bgTask != .invalid { UIApplication.shared.endBackgroundTask(bgTask) } }
            status = "上傳到 Kaggle…"; detail = String(format: "%.0f MB，上傳時請保持 App 在前景", mb)
            let token = try await withRetry("上傳") { try await client.uploadBlob(file: zipURL) }
            status = "建立 Kaggle 資料集…"; detail = ""
            try await withRetry("建立資料集") {
                try await client.pushDataset(slug: datasetSlug, title: "obj3d capture data", fileToken: token)
            }
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
        let d = UserDefaults.standard
        let smooth = d.string(forKey: "smoothLevel") ?? "medium"
        let paramsJSON = DevParams.json()
        let script = template.replacingOccurrences(of: "__JOB_ID__", with: jobId)
            .replacingOccurrences(of: "__SMOOTH__", with: smooth)
            .replacingOccurrences(of: "__PARAMS__", with: paramsJSON)
        let shape = UserDefaults.standard.string(forKey: "machineShape") ?? "NvidiaTeslaT4"
        status = "啟動 Kaggle 運算…"
        try await withRetry("啟動運算") {
            try await client.pushKernel(slug: kernelSlug, title: "obj3d runner", script: script,
                                        datasetSlug: datasetSlug, machineShape: shape)
        }
    }

    private func poll(client: KaggleClient, jobId: String) async throws {
        let start = Date()
        var restarts = 0
        while true {
            try Task.checkCancellation()
            let elapsed = Date().timeIntervalSince(start)
            if elapsed > 4 * 3600 { throw KaggleError(message: "等待超過 4 小時，請到 Kaggle 網站查看 obj3d-runner") }
            var statusPair: (String, String?)
            do {
                statusPair = try await client.kernelStatus(slug: kernelSlug)
            } catch let e as KaggleError {
                throw e
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                status = "網路不穩，30 秒後再查詢…"           // 查詢途中斷線不算失敗
                try await Task.sleep(nanoseconds: 30_000_000_000)
                continue
            }
            let (st, failure) = statusPair
            let s = st.lowercased()
            if s.contains("queued") { status = "Kaggle 排隊中…" }
            else if s.contains("running") { status = "Kaggle 運算中…" }
            else { status = "Kaggle 狀態：\(st)" }
            detail = "已等待 \(Int(elapsed / 60)) 分鐘（通常 20~60 分鐘），可以先離開 App，回來後按「繼續查詢」"

            if s.contains("complete") || s.contains("error") || s.contains("cancel") {
                let files = try await withRetry("讀取結果清單") { try await client.kernelOutputs(slug: kernelSlug) }
                if let metaFile = files.first(where: { $0.name.hasSuffix("result_meta.json") }) {
                    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("result_meta.json")
                    try await withRetry("下載結果") { try await client.download(metaFile.url, to: tmp) }
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
        let wanted = ["result_meta.json", "preview.bin", "preview_tex.jpg", "model.glb", "model_mm.stl", "model_mm.ply",
                      "diag_masks.jpg", "diag_cameras.png", "model.splat"]
        for name in wanted {
            if let f = files.first(where: { ($0.name as NSString).lastPathComponent == name }) {
                detail = name
                try await withRetry("下載 \(name)") { try await client.download(f.url, to: dir.appendingPathComponent(name)) }
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
        if let s = r.meta.size_cm, s.count == 3, r.meta.scale_source != "none" {
            lines.append(String(format: "尺寸：寬 %.1f × 深 %.1f × 高 %.1f cm", s[0], s[1], s[2]))
        }
        lines.append(r.meta.watertight == true ? "封閉無破孔 ✅" : "⚠️ 模型未完全封閉")
        if let t = r.meta.target_method {
            lines.append("物體定位：\(t)" + (r.meta.frames_used.map { "，使用 \($0) 張影格" } ?? ""))
        }
        if let b = r.meta.bad_mask_frames, b > 0 { lines.append("略過 \(b) 張遮罩異常的影格") }
        var feats: [String] = []
        if r.meta.tsdf == true { feats.append("TSDF") }
        if r.meta.textured == true { feats.append("貼圖") }
        lines.append("運算程式：\(r.meta.runner_version ?? "舊版（無版本資訊）")" + (feats.isEmpty ? "" : "（\(feats.joined(separator: "、"))）"))
        if r.meta.gs == true, let n = r.meta.gs_count {
            lines.append("3DGS 擬真模型：\(n / 1000) 千個元素" + (r.meta.gs_mb.map { String(format: "（%.1f MB）", $0) } ?? ""))
        }
        if let src = r.meta.scale_source {
            var t = "尺寸依據：" + (src == "ruler" ? "比例尺" : (src == "none" ? "無（相對比例，最大邊＝10，單位不是公分）" : "手機位置（ARKit）"))
            if let d = r.meta.ruler_vs_arkit { t += String(format: "（比例尺與 ARKit 相差 %+.1f%%）", d * 100) }
            lines.append(t)
        }
        if let p = r.meta.params_summary { lines.append("參數：\(p)、平滑度 \(r.meta.smooth_level ?? "?")") }
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
