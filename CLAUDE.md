# obj3d 專案說明（給 Claude Code）

iPhone App「3D 掃描」：用 iPhone 錄影（含 ARKit 相機軌跡），上傳到使用者的 Kaggle 帳號，
在 Kaggle 免費 GPU 上執行 `kaggle/runner.py` 建立 3D 模型，再下載回手機。

## 結構
- `ios/`：SwiftUI App 原始碼；`ios/project.yml` 由 XcodeGen 產生 Xcode 專案（這台電腦是 Windows，無法本機編譯）。
- `kaggle/runner.py`：Kaggle 上執行的建模程式，編譯時被打包進 App；App 每次生成時會把它上傳到 Kaggle。
- `.github/workflows/build-ios.yml`：在 GitHub 的雲端 Mac 上編譯，產出 artifact `Obj3D-ipa`（未簽名，由使用者用 Sideloadly 簽名安裝）。
- `.claude/skills/update-app/`：更新流程（/update-app）。

## 規則
- 使用者以繁體中文溝通。
- 不要讀取、輸入或保存任何金鑰與密碼（Kaggle 金鑰只存在 iPhone 鑰匙圈）。
- 程式碼由另一位 Claude 在 claude.ai 上開發，使用者會把新版打包成 obj3d.zip；除非使用者要求，不要自行改寫程式邏輯。
- 不要 force push。
