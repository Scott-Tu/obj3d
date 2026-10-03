import Foundation

struct KaggleError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct OutputFile {
    let name: String
    let url: URL
}

/// Kaggle 公開 API（https://api.kaggle.com/v1/<服務>/<方法>，POST JSON）
struct KaggleClient {
    let username: String
    let key: String

    static func fromSettings() -> KaggleClient? {
        let user = (UserDefaults.standard.string(forKey: "kaggleUsername") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let key = (Keychain.get("kaggleKey") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !user.isEmpty, !key.isEmpty else { return nil }
        return KaggleClient(username: user, key: key)
    }

    /// 新版 API Token（KGAT_ 開頭）用 Bearer；舊版 kaggle.json 的 key 用 Basic
    private var authorization: String {
        if key.hasPrefix("KGAT_") { return "Bearer \(key)" }
        return "Basic " + Data("\(username):\(key)".utf8).base64EncodedString()
    }

    @discardableResult
    func call(_ service: String, _ method: String, _ body: [String: Any]) async throws -> [String: Any] {
        guard let url = URL(string: "https://api.kaggle.com/v1/\(service)/\(method)") else {
            throw KaggleError(message: "網址錯誤")
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 120
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(authorization, forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let text = String(data: data, encoding: .utf8) ?? ""
        if code == 401 || code == 403 {
            throw KaggleError(message: "Kaggle 驗證失敗（\(code)）：請檢查設定裡的使用者名稱與 API 金鑰。\(method)")
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

    // MARK: - 上傳檔案並建立資料集版本

    func uploadBlob(file: URL) async throws -> String {
        let attrs = try FileManager.default.attributesOfItem(atPath: file.path)
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        let r = try await call("blobs.BlobApiService", "StartBlobUpload", [
            "type": "DATASET",
            "name": file.lastPathComponent,
            "contentLength": size,
            "lastModifiedEpochSeconds": Int(Date().timeIntervalSince1970),
        ])
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
