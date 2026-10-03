import SwiftUI

struct SettingsView: View {
    @AppStorage("kaggleUsername") private var username = ""
    @AppStorage("machineShape") private var machineShape = "NvidiaTeslaT4"
    @State private var apiKey = Keychain.get("kaggleKey") ?? ""

    var body: some View {
        Form {
            Section(header: Text("Kaggle 帳號"),
                    footer: Text("到 kaggle.com → 右上角頭像 → Settings → API → Create New Token。金鑰只存在這支 iPhone 的鑰匙圈裡。")) {
                TextField("使用者名稱（username）", text: $username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("API 金鑰", text: $apiKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .onChange(of: apiKey) { newValue in
                        Keychain.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), for: "kaggleKey")
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
}
