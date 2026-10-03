import SwiftUI

struct SettingsView: View {
    @AppStorage("kaggleUsername") private var username = ""
    @AppStorage("machineShape") private var machineShape = "NvidiaTeslaT4"
    @State private var apiKey = Keychain.get("kaggleKey") ?? ""
    @State private var testing = false
    @State private var testResult = ""

    var body: some View {
        Form {
            Section(header: Text("Kaggle 帳號"),
                    footer: Text("使用者名稱是 kaggle.com/ 後面那段（全小寫），不是顯示名稱。金鑰在 kaggle.com/settings → API → Generate New Token。金鑰只存在這支 iPhone 的鑰匙圈裡。")) {
                TextField("使用者名稱（例如 chiahsuntu）", text: $username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("API 金鑰", text: $apiKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .onChange(of: apiKey) { newValue in
                        Keychain.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), for: "kaggleKey")
                        KaggleClient.resetAuthMode()
                    }
            }

            Section(header: Text("檢查")) {
                Button(testing ? "測試中…" : "測試 Kaggle 連線") {
                    Task { await runTest() }
                }
                .disabled(testing)
                if !testResult.isEmpty {
                    Text(testResult)
                        .font(.footnote)
                        .textSelection(.enabled)
                }
            }

            Section(header: Text("進階"),
                    footer: Text("Kaggle 的 GPU 機型代號，預設 NvidiaTeslaT4；留空則使用 Kaggle 預設值。")) {
                TextField("GPU 機型", text: $machineShape)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }

            Section(header: Text("說明")) {
                Text("每次生成會在你的 Kaggle 帳號下更新私人資料集「obj3d-capture-data」與私人程式「obj3d-runner」，並使用 Kaggle 的每週 GPU 配額。")
                    .font(.footnote)
            }
        }
        .navigationTitle("設定")
    }

    private func runTest() async {
        testing = true
        defer { testing = false }
        let u = username.trimmingCharacters(in: .whitespacesAndNewlines)
        let k = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !u.isEmpty, !k.isEmpty else {
            testResult = "請先填寫使用者名稱和 API 金鑰"
            return
        }
        var lines: [String] = []
        lines.append("使用者名稱：\(u)")
        lines.append("金鑰長度 \(k.count) 個字元，開頭「\(String(k.prefix(5)))…」")
        if u != u.lowercased() { lines.append("⚠️ 使用者名稱應該全部是小寫") }
        if u.contains(" ") { lines.append("⚠️ 使用者名稱不應該有空白，請填 kaggle.com/ 後面那段") }

        KaggleClient.resetAuthMode()
        let client = KaggleClient(username: u, key: k)
        do {
            let r = try await client.call("security.OAuthService", "IntrospectToken", ["token": k])
            let active = (r["active"] as? Bool) ?? false
            let owner = (r["username"] as? String) ?? ""
            lines.append("Token 檢查：\(active ? "有效" : "無效")\(owner.isEmpty ? "" : "，屬於 \(owner)")")
            if !owner.isEmpty && owner.lowercased() != u.lowercased() {
                lines.append("⚠️ 這個金鑰屬於「\(owner)」，和你填的使用者名稱不同")
            }
        } catch {
            lines.append("Token 檢查：略過（\(error.localizedDescription.prefix(60))）")
        }
        do {
            _ = try await client.call("kernels.KernelsApiService", "GetKernelSessionStatus",
                                      ["userName": u, "kernelSlug": "obj3d-runner"])
            lines.append("✅ 驗證成功，可以開始使用")
        } catch let e as KaggleError {
            if e.message.contains("（401）") || e.message.contains("（403）") {
                lines.append("❌ 驗證失敗：請重新產生 API 金鑰並貼上，確認使用者名稱正確")
            } else {
                lines.append("✅ 驗證成功（目前還沒有 obj3d-runner 的執行紀錄，這是正常的）")
            }
        } catch {
            lines.append("網路錯誤：\(error.localizedDescription)")
        }
        testResult = lines.joined(separator: "\n")
    }
}
