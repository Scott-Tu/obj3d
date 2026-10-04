import SwiftUI

struct SettingsView: View {
    @AppStorage("kaggleUsername") private var username = ""
    @AppStorage("machineShape") private var machineShape = "NvidiaTeslaT4"
    @AppStorage("smoothLevel") private var smoothLevel = "medium"
    @AppStorage("saveVideoToPhotos") private var saveVideoToPhotos = true
    @AppStorage("scaleLengthCM") private var scaleLengthCM = 15.0
    @State private var apiKey = Keychain.get("kaggleKey") ?? ""
    @State private var testing = false
    @State private var testResult = ""

    var body: some View {
        Form {
            Section(header: Text("Kaggle 帳號"),
                    footer: Text("使用者名稱是 kaggle.com/ 後面那段（全小寫），不是顯示名稱。金鑰可用 kaggle.com/settings → API 的新版 Token（KGAT_ 開頭），或 Create Legacy API Key 下載的 kaggle.json 裡的 key。金鑰只存在這支 iPhone 的鑰匙圈裡。")) {
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

            Section(header: Text("錄影"),
                    footer: Text("錄影時同時錄一段一般影片（每秒 30 張）存到「照片」App，第一次會詢問權限。建模用的影格仍另外存在 App 裡。")) {
                Toggle("同時存影片到「照片」", isOn: $saveVideoToPhotos)
            }

            Section(header: Text("物體旋轉模式"),
                    footer: Text("比例尺要放在轉盤上、物體旁邊，跟著物體一起轉，且兩端在畫面中清楚可見。長度請量實際的兩端距離。")) {
                Stepper(value: $scaleLengthCM, in: 3...60, step: 0.5) {
                    HStack {
                        Text("比例尺長度")
                        Spacer()
                        Text(String(format: "%.1f cm", scaleLengthCM)).foregroundColor(.secondary)
                    }
                }
            }

            Section(header: Text("模型"),
                    footer: Text("光滑的物體（杯子、瓶子）選「高」；有細節紋理的物體選「低」。下次生成時套用，不用重拍。")) {
                Picker("表面平滑度", selection: $smoothLevel) {
                    Text("低（保留細節）").tag("low")
                    Text("中").tag("medium")
                    Text("高（光滑物體）").tag("high")
                }
            }

            Section(header: Text("開發者參數（測試用）")) {
                NavigationLink {
                    DevParamsView()
                } label: {
                    HStack {
                        Text("調整運算參數")
                        Spacer()
                        Text(DevParams.changedCount == 0 ? "全部預設" : "已修改 \(DevParams.changedCount) 項")
                            .foregroundColor(.secondary)
                    }
                }
            }

            Section(header: Text("進階"),
                    footer: Text("Kaggle 的 GPU 機型代號，預設 NvidiaTeslaT4；留空則使用 Kaggle 預設值。")) {
                TextField("GPU 機型", text: $machineShape)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }

            Section(header: Text("版本"),
                    footer: Text("build 編號是 GitHub 的編譯次數，每次更新都會變大；運算程式版本會寫進每次結果的 result_meta.json。")) {
                LabeledContent("App 版本", value: AppInfo.version)
                LabeledContent("Build", value: AppInfo.build)
                LabeledContent("Commit", value: AppInfo.commit)
                LabeledContent("運算程式", value: AppInfo.runnerVersion)
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
        func short(_ e: Error) -> String { String(e.localizedDescription.prefix(160)) }

        lines.append(await KaggleClient.headerEchoTest())

        var tokenOK = false
        do {
            let r = try await client.call("security.OAuthService", "IntrospectToken", ["token": k])
            let active = (r["active"] as? Bool) ?? false
            let owner = (r["username"] as? String) ?? ""
            let scope = (r["scope"] as? String) ?? ""
            tokenOK = active
            lines.append("① Token：\(active ? "有效" : "無效")\(owner.isEmpty ? "" : "，屬於 \(owner)")")
            if !scope.isEmpty { lines.append("　權限範圍：\(scope)") }
            if !owner.isEmpty && owner.lowercased() != u.lowercased() {
                lines.append("⚠️ 這個金鑰屬於「\(owner)」，和你填的使用者名稱不同")
            }
        } catch {
            lines.append("① Token：無法檢查（\(short(error))）")
        }

        var uploadOK = false
        do {
            let r = try await client.call("datasets.DatasetApiService", "UploadDatasetFile", [
                "fileName": "obj3d_test.txt", "contentLength": 1,
                "lastModifiedEpochSeconds": Int(Date().timeIntervalSince1970)])
            uploadOK = (r["createUrl"] as? String) != nil
            lines.append(uploadOK ? "② 上傳權限：✅" : "② 上傳權限：⚠️ 沒有回傳上傳網址")
        } catch {
            lines.append("② 上傳權限：❌ \(short(error))")
        }

        do {
            let r = try await client.call("datasets.DatasetApiService", "GetDatasetStatus",
                                          ["ownerSlug": u, "datasetSlug": "obj3d-capture-data"])
            lines.append("③ 資料集：已存在（\((r["status"] as? String) ?? "")）")
        } catch {
            lines.append("③ 資料集：尚未建立或無法讀取（\(short(error))）")
        }

        do {
            let r = try await client.call("kernels.KernelsApiService", "GetKernelSessionStatus",
                                          ["userName": u, "kernelSlug": "obj3d-runner"])
            lines.append("④ 運算程式：\((r["status"] as? String) ?? "")")
        } catch {
            lines.append("④ 運算程式：尚未建立或無法讀取（\(short(error))）")
        }

        lines.append("傳輸方式：\(KaggleClient.usesRawHTTP ? "直接連線（繞過 iOS 網路元件）" : "iOS 標準")")
        lines.append(uploadOK ? "✅ 驗證成功，可以開始生成（③④ 第一次使用時顯示「尚未建立」是正常的；① 用舊版 key 時顯示 404 也正常）"
                                         : "❌ 驗證未通過，請把這段結果截圖給我")
        testResult = lines.joined(separator: "\n")
    }
}
