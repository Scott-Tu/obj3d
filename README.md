# 3D 掃描 App（iPhone 錄影 → Kaggle 自動建模）

用 iPhone 繞著物體錄影，按一下「生成」，App 會自動把資料上傳到你的 Kaggle 帳號，用免費 GPU 建出 3D 模型，完成後下載回手機。

- 不需要比例尺：App 用 iPhone 的動作追蹤（ARKit）記錄手機移動了多少公分，換算真實尺寸。
- 模型是封閉實體（無破孔），附桌面底座，可直接 3D 列印。
- 輸出：`model.glb`（公尺）、`model_mm.stl`、`model_mm.ply`（公釐）。

## 資料夾內容

| 路徑 | 說明 |
|---|---|
| `ios/` | iPhone App 原始碼（SwiftUI），`project.yml` 用來產生 Xcode 專案 |
| `kaggle/runner.py` | 在 Kaggle 上執行的建模程式（會被打包進 App，每次生成時上傳） |
| `.github/workflows/build-ios.yml` | 在 GitHub 的雲端 Mac 上自動編譯 App |

---

## 一、準備 Kaggle 帳號（一次就好）

1. 到 https://www.kaggle.com 註冊並登入。
2. 右上角頭像 → **Settings** → 完成 **Phone verification（手機驗證）**。沒驗證就不能用 GPU 和網路。
3. 打開 https://www.kaggle.com/settings ，在 **API** 區塊按 **Generate New Token**，把產生的金鑰（`KGAT_` 開頭）複製下來，稍後貼到 App 裡。
   - 也可以按 **Create Legacy API Key** 下載舊版 `kaggle.json`，裡面的 `key` 一樣可以用。
4. 記下你的 Kaggle **使用者名稱（username）**，在個人頁網址 `kaggle.com/你的名稱` 可以看到。

## 二、在 GitHub 編譯出 App 安裝檔（只有 Windows 也可以）

iOS App 必須在 Mac 上編譯，這裡借用 GitHub 免費提供的雲端 Mac。

1. 到 https://github.com 註冊帳號。
2. 右上角 **+** → **New repository**，名稱例如 `obj3d`。
   - 建議選 **Public**：程式碼裡沒有任何密碼或金鑰，公開的專案編譯不限時數。
   - 選 Private 也可以，但免費方案的 Mac 編譯時數有限（每次約 5~10 分鐘，一個月夠用好幾次）。
3. 把這個資料夾的**所有內容**上傳到專案：
   - 最簡單：安裝 **GitHub Desktop**（Windows 版），把專案 clone 到電腦，把這些檔案複製進去，按 **Commit** → **Push**。
   - 或在網頁上按 **Add file → Upload files** 拖曳上傳。注意：網頁上傳可能漏掉以 `.` 開頭的 `.github` 資料夾。如果漏了，按 **Add file → Create new file**，檔名輸入 `.github/workflows/build-ios.yml`，把該檔內容貼上後存檔。
4. 到專案的 **Actions** 分頁 → 左邊點 **Build iOS App** → 右邊 **Run workflow**。
5. 約 5~10 分鐘後出現綠色勾勾，點進去，頁面下方 **Artifacts** 有 **Obj3D-ipa**，下載後解壓縮得到 `Obj3D.ipa`。
   - 如果是紅色叉叉：點進去看「列出編譯錯誤」那一步的訊息，複製給我修正。

## 三、安裝到 iPhone（用 Windows + Sideloadly）

1. 在 Windows 安裝：
   - **iTunes、iCloud**：一定要用 Apple 官網下載的版本。如果電腦裡已經有 Microsoft Store 版，請先解除安裝。
   - **Sideloadly**（https://sideloadly.io）
2. 用 USB 線連接 iPhone，iPhone 上按「信任這部電腦」。
3. 打開 Sideloadly：
   - 把 `Obj3D.ipa` 拖進視窗
   - **Apple account** 填你的 Apple ID（建議另外註冊一個測試用的 Apple ID；全新的 Apple ID 要先在任一台 Apple 裝置上登入過一次，否則可能無法使用）
   - 按 **Start**，依指示輸入密碼與雙重驗證碼
4. iPhone 上第一次要做兩件事：
   - **設定 → 隱私權與安全性 → 開發者模式** → 開啟 → 重新開機 → 開機後按「開啟」確認
   - **設定 → 一般 → VPN 與裝置管理** → 點你的 Apple ID → **信任**
5. 主畫面出現「3D 掃描」就可以使用了。

> **限制：** 用免費 Apple ID 安裝的 App **7 天後會失效**，要用 Sideloadly 再裝一次（資料會保留）。付費 Apple 開發者帳號（每年 99 美元）可以用一年。

## 四、使用方式

1. 第一次打開：右上角齒輪 → 填 Kaggle **使用者名稱**和 **API 金鑰**。
2. **錄影**：
   - 物體放在桌上，背景盡量單純、光線充足
   - 讓物體一直在畫面中央的十字上
   - 慢慢繞一整圈，再從斜上方繞一圈，約 40~90 秒
   - 上方顯示「移動太快」時就放慢
3. **生成 3D 模型**：上傳約 1~3 分鐘，Kaggle 排隊加運算約 20~60 分鐘。
   - 等待時可以離開 App，Kaggle 會繼續算；回來後按「繼續查詢上一次的工作」。
4. 完成後可以用手指旋轉、縮放模型，並用「分享」把 GLB／STL／PLY 存到檔案或傳到電腦。
   - 模型檔也可以在「檔案」App → 我的 iPhone → 3D 掃描 → results 找到。

## 四之二、兩種拍攝方式

主畫面「拍攝方式」可選：

- **手機繞物體**：物體不動，拿手機繞一圈。尺寸由 iPhone 動作追蹤（ARKit）換算，不需比例尺。
- **物體旋轉**：手機大致不動（手持晃動沒關係），旋轉物體（例如放在轉盤上）。
  1. 把**已知長度的比例尺**（預設 15 cm，可在設定修改）放在轉盤上、物體旁邊，跟著物體一起轉。
  2. 錄完後在 2 張畫面上點選比例尺的兩端（可再點一下物體），按「完成」。
  3. 按「生成 3D 模型」。程式會先遮掉不動的背景，只用物體與比例尺計算相機位置，並用 iPhone 的姿態判斷重力方向。

## 五、常見問題

- **Kaggle 驗證失敗（401/403）**：檢查使用者名稱與金鑰；確認帳號已完成手機驗證。
- **運算失敗**：到 kaggle.com → Code → `obj3d-runner` 打開最新版本，看最下面的紀錄。
- **找不到物體**：錄影時物體沒有一直在十字中央，或物體太小、太遠。
- **尺寸不準**：ARKit 追蹤不穩（移動太快、畫面太暗、背景沒有紋理）時會影響尺寸。
- **反光、透明、很細的物體**（玻璃、鏡面、鐵絲）不容易重建。

每次生成都會更新你 Kaggle 帳號下的私人資料集 `obj3d-capture-data` 與私人程式 `obj3d-runner`，並使用 Kaggle 每週的免費 GPU 配額。


## 六、用 Claude Code 自動更新（選用）

專案內附 `.claude/skills/update-app/`。在專案資料夾開啟 Claude Code 後輸入 `/update-app`，
它會自動：找到「下載」裡最新的 obj3d.zip → 套用 → Commit、Push → 等 GitHub 編譯 → 下載 Obj3D.ipa 並開啟 Sideloadly。
最後用 Sideloadly 安裝到 iPhone 仍需手動（需要你的 Apple ID）。
