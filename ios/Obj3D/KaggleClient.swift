import Foundation

struct KaggleError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct OutputFile {
    let name: String
    let url: URL
}

/// iOS 的 URLSession 在轉址時會把 Authorization 標頭拿掉（Python、Go 的用戶端不會），
/// 這裡在轉址到 kaggle.com 時把驗證標頭和 POST 內容補回去。
final class KaggleRedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        var r = request
        if let orig = task.originalRequest, let host = r.url?.host, host.hasSuffix("kaggle.com") {
            if let auth = orig.value(forHTTPHeaderField: "Authorization") {
                r.setValue(auth, forHTTPHeaderField: "Authorization")
            }
            if orig.httpMethod == "POST" && r.httpMethod != "POST" {
                r.httpMethod = "POST"
                r.httpBody = orig.httpBody
                r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            }
        }
        completionHandler(r)
    }
}

/// Kaggle 公開 API（https://api.kaggle.com/v1/<服務>/<方法>，POST JSON）
struct KaggleClient {
    static let apiSession = URLSession(configuration: .default, delegate: KaggleRedirectDelegate(), delegateQueue: nil)

    let username: String
    let key: String

    static func fromSettings() -> KaggleClient? {
        let user = (UserDefaults.standard.string(forKey: "kaggleUsername") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let key = (Keychain.get("kaggleKey") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !user.isEmpty, !key.isEmpty else { return nil }
        return KaggleClient(username: user, key: key)
    }

    private static let modeKey = "kaggleAuthMode"

    /// 新版 API Token 用 Bearer；舊版 kaggle.json 的 key 用 Basic。
    /// 先試比較可能的方式，遇到 401/403 自動改用另一種，成功後記住。
    private var authModes: [String] {
        if let saved = UserDefaults.standard.string(forKey: KaggleClient.modeKey) {
            return saved == "bearer" ? ["bearer", "basic"] : ["basic", "bearer"]
        }
        return key.hasPrefix("KGAT_") ? ["bearer", "basic"] : ["basic", "bearer"]
    }

    private func header(_ mode: String) -> String {
        if mode == "bearer" { return "Bearer \(key)" }
        return "Basic " + Data("\(username):\(key)".utf8).base64EncodedString()
    }

    private var authorization: String { header(authModes[0]) }

    static func resetAuthMode() {
        UserDefaults.standard.removeObject(forKey: modeKey)
    }

    @discardableResult
    func call(_ service: String, _ method: String, _ body: [String: Any]) async throws -> [String: Any] {
        guard let url = URL(string: "https://api.kaggle.com/v1/\(service)/\(method)") else {
            throw KaggleError(message: "網址錯誤")
        }
        let payload = try JSONSerialization.data(withJSONObject: body)
        let modes = authModes
        for (i, mode) in modes.enumerated() {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.timeoutInterval = 120
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("kaggle-api/v1.7.0", forHTTPHeaderField: "User-Agent")
            req.setValue(header(mode), forHTTPHeaderField: "Authorization")
            req.httpBody = payload
            let (data, resp) = try await KaggleClient.apiSession.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            let text = String(data: data, encoding: .utf8) ?? ""
            let finalURL = resp.url?.absoluteString ?? ""
            let moved = finalURL.isEmpty || finalURL == url.absoluteString ? "" : "（轉址到 \(finalURL)）"
            if code == 401 || code == 403 {
                if i < modes.count - 1 { continue }          // 換另一種驗證方式再試一次
                let detail = (obj["message"] as? String) ?? String(text.prefix(200))
                throw KaggleError(message: "Kaggle 驗證失敗（\(code)）：\(method)\(moved)。請到「設定」按「測試 Kaggle 連線」檢查使用者名稱與金鑰。\(detail)")
            }
            if code > 0 && code < 400 {
                UserDefaults.standard.set(mode, forKey: KaggleClient.modeKey)
            }
            if code >= 400 {
                throw KaggleError(message: "Kaggle \(method) 失敗（\(code)）：\((obj["message"] as? String) ?? String(text.prefix(300)))")
            }
            if let c = obj["code"] as? Int, c >= 400 {
                throw KaggleError(message: "Kaggle \(method) 失敗（\(c)）：\(obj["message"] as? String ?? "")")
            }
            if let e = obj["error"] as? String, !e.isEmpty {
                throw KaggleError(message: "Kaggle \(method) 錯誤：\(e)")
            }
            return obj
        }
        throw KaggleError(message: "Kaggle \(method) 沒有回應")
    }

    // MARK: - 上傳檔案並建立資料集版本

    func uploadBlob(file: URL) async throws -> String {
        let attrs = try FileManager.default.attributesOfItem(atPath: file.path)
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        let now = Int(Date().timeIntervalSince1970)
        var r: [String: Any]
        do {
            // 與已在真實 Kaggle 驗證過的用法相同（datasets.UploadDatasetFile）
            r = try await call("datasets.DatasetApiService", "UploadDatasetFile", [
                "fileName": file.lastPathComponent,
                "contentLength": size,
                "lastModifiedEpochSeconds": now,
            ])
        } catch {
            r = try await call("blobs.BlobApiService", "StartBlobUpload", [
                "type": "DATASET",
                "name": file.lastPathComponent,
                "contentLength": size,
                "lastModifiedEpochSeconds": now,
            ])
        }
        guard let token = r["token"] as? String, let s = r["createUrl"] as? String, let url = URL(string: s) else {
            throw KaggleError(message: "Kaggle 沒有回傳上傳網址")
        }
        var put = URLRequest(url: url)
        put.httpMethod = "PUT"
        put.timeoutInterval = 900
        let (_, resp) = try await URLSession.shared.upload(for: put, fromFile: file)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw KaggleError(message: "上傳失敗（HTTP \(code)）")
        }
        return token
    }

    func pushDataset(slug: String, title: String, fileToken: String) async throws {
        var exists = true
        do {
            try await call("datasets.DatasetApiService", "GetDatasetStatus",
                           ["ownerSlug": username, "datasetSlug": slug])
        } catch {
            exists = false
        }
        if exists {
            try await call("datasets.DatasetApiService", "CreateDatasetVersion", [
                "ownerSlug": username,
                "datasetSlug": slug,
                "body": [
                    "versionNotes": "capture \(Int(Date().timeIntervalSince1970))",
                    "deleteOldVersions": true,
                    "files": [["token": fileToken]],
                ] as [String: Any],
            ])
        } else {
            try await call("datasets.DatasetApiService", "CreateDataset", [
                "ownerSlug": username,
                "slug": slug,
                "title": title,
                "licenseName": "CC0-1.0",
                "isPrivate": true,
                "files": [["token": fileToken]],
            ])
        }
    }

    func waitDatasetReady(slug: String, timeout: TimeInterval) async throws {
        let start = Date()
        while Date().timeIntervalSince(start) < timeout {
            try Task.checkCancellation()
            let r = try? await call("datasets.DatasetApiService", "GetDatasetStatus",
                                    ["ownerSlug": username, "datasetSlug": slug])
            let st = ((r?["status"] as? String) ?? "").lowercased()
            if st.contains("ready") { return }
            if st.contains("failed") { throw KaggleError(message: "Kaggle 處理資料集失敗") }
            try await Task.sleep(nanoseconds: 10_000_000_000)
        }
        throw KaggleError(message: "等待資料集逾時，請稍後再試")
    }

    // MARK: - 執行運算腳本

    func pushKernel(slug: String, title: String, script: String, datasetSlug: String, machineShape: String) async throws {
        var body: [String: Any] = [
            "slug": "\(username)/\(slug)",
            "newTitle": title,
            "text": script,
            "language": "python",
            "kernelType": "script",
            "isPrivate": true,
            "enableGpu": true,
            "enableInternet": true,
            "datasetDataSources": ["\(username)/\(datasetSlug)"],
            "competitionDataSources": [String](),
            "kernelDataSources": [String](),
            "modelDataSources": [String](),
            "categoryIds": [String](),
        ]
        if !machineShape.isEmpty { body["machineShape"] = machineShape }
        try await call("kernels.KernelsApiService", "SaveKernel", body)
    }

    func kernelStatus(slug: String) async throws -> (status: String, failure: String?) {
        let r = try await call("kernels.KernelsApiService", "GetKernelSessionStatus",
                               ["userName": username, "kernelSlug": slug])
        return ((r["status"] as? String) ?? "", r["failureMessage"] as? String)
    }

    func kernelOutputs(slug: String) async throws -> [OutputFile] {
        let r = try await call("kernels.KernelsApiService", "ListKernelSessionOutput",
                               ["userName": username, "kernelSlug": slug, "pageSize": 100])
        let files = r["files"] as? [[String: Any]] ?? []
        return files.compactMap { f in
            guard let name = f["fileName"] as? String, let s = f["url"] as? String, let u = URL(string: s) else { return nil }
            return OutputFile(name: name, url: u)
        }
    }

    /// 下載輸出檔：先不帶驗證（簽名網址），失敗再帶驗證重試
    func download(_ url: URL, to dest: URL) async throws {
        for withAuth in [false, true] {
            var req = URLRequest(url: url)
            req.timeoutInterval = 600
            if withAuth { req.setValue(authorization, forHTTPHeaderField: "Authorization") }
            let (tmp, resp) = try await URLSession.shared.download(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if (200..<300).contains(code) {
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
                return
            }
            if !(code == 401 || code == 403) || withAuth {
                throw KaggleError(message: "下載失敗（HTTP \(code)）")
            }
        }
    }
}
