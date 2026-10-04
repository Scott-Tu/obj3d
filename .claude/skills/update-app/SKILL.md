---
name: update-app
description: 把「下載」資料夾裡最新的 obj3d.zip 套用到這個專案，Commit、Push，等 GitHub Actions 編譯完成並下載 Obj3D.ipa。使用者說「更新 App」「套用新的 obj3d.zip」或輸入 /update-app 時使用。
---

# 更新 3D 掃描 App（obj3d）

你要在這台 Windows 電腦上，替使用者完成「收到新版 obj3d.zip → 產生新的 Obj3D.ipa」的整個流程。
每一步完成後用一兩句**繁體中文**回報進度；遇到需要使用者決定的事就停下來問。

## 安全規則（一定要遵守）

- **絕對不要**讀取、詢問、輸入或印出 Kaggle 金鑰、Apple ID、密碼。這些只存在使用者的 iPhone 和 Sideloadly 裡。
- 不要用 `git push --force`，不要改寫 git 歷史，不要刪除 `.git`。
- 只動這個專案資料夾、使用者的「下載」資料夾、以及 `下載\obj3d-builds\`。其他位置都不要刪改。
- 除非使用者明確要求，不要自行修改 `ios/`、`kaggle/` 裡的程式邏輯（只有「第 7 步：編譯失敗」可以在使用者同意後修正編譯錯誤）。

## 第 0 步：環境檢查

1. 確認目前資料夾是專案根目錄：要有 `ios/project.yml`、`kaggle/runner.py`、`.github/workflows/build-ios.yml`、`.git/`。不是的話停下來告訴使用者。
2. 確認 `git` 和 `gh`（GitHub CLI）可用，並執行 `gh auth status` 確認已登入。
   - 沒裝 `gh`：請使用者在 PowerShell 執行 `winget install --id GitHub.cli`，然後重開終端機。
   - 沒登入：請使用者自己執行 `gh auth login`（選 GitHub.com → HTTPS → 用瀏覽器登入）。**不要替使用者登入。**
3. `git status --porcelain` 要是空的。如果有未 commit 的變更，列出來問使用者要先 commit、捨棄，還是取消。
4. 確認在 `main` 分支，執行 `git pull --ff-only` 取得最新版本。

## 第 1 步：找到新的 obj3d.zip

- 如果使用者在指令後面有給路徑，就用那個檔案。
- 否則找使用者「下載」資料夾（`%USERPROFILE%\Downloads`，Git Bash 中是 `~/Downloads`）裡**最新**的 `obj3d*.zip`。
- 回報檔名與下載時間。如果最新的檔案超過 24 小時，先問使用者是不是這個檔案。

## 第 2 步：解壓縮並檢查內容

1. 解壓縮到暫存資料夾（例如 `%TEMP%\obj3d-update-<時間>`）。
2. 在解壓結果中找到「同時含有 `ios/project.yml` 和 `kaggle/runner.py`」的那一層（通常是 `obj3d/`）。
3. 確認存在 `.github/workflows/build-ios.yml`。缺任何一項就停下來告訴使用者這個 zip 不完整。
4. 如果電腦有 Python，執行 `python -m py_compile kaggle/runner.py` 檢查語法（在暫存資料夾裡檢查）。

## 第 3 步：套用到專案

1. 用新版**完整取代**這三個資料夾：`ios/`、`kaggle/`、`.github/`（先刪除專案裡的舊資料夾再複製新的，這樣被移除的檔案也會同步）。
2. 用新版覆蓋根目錄的 `README.md`、`CLAUDE.md`，以及 `.claude/` 資料夾（只覆蓋，不刪除使用者自己另外加的檔案）。
3. 不要動 `.git/`，也不要複製解壓結果中的 `.git/`（如果有的話）。
4. 刪除暫存資料夾。

## 第 4 步：檢視變更

1. 執行 `git status` 和 `git diff --stat`。
2. 用三到五句話向使用者摘要這次改了什麼（看檔案名稱與 diff 內容判斷，例如「Kaggle 運算程式新增分批推論」「App 新增尺規」）。
3. 如果沒有任何變更，告訴使用者「這個 zip 和目前版本相同」，然後結束。

## 第 5 步：Commit 與 Push

1. `git add -A`
2. Commit 訊息用繁體中文，格式：`更新：<一句話摘要>（obj3d.zip <檔案日期>）`
3. `git push origin main`

## 第 6 步：等待 GitHub Actions 編譯

1. 取得剛 push 的 commit：`git rev-parse HEAD`。
2. 每 10 秒執行一次 `gh run list --workflow "Build iOS App" --branch main --limit 5 --json databaseId,headSha,status,conclusion`，找到 `headSha` 等於剛才 commit 的那一筆（最多等 3 分鐘）。
3. 執行 `gh run watch <databaseId> --exit-status` 等它完成（通常 5~10 分鐘）。等待期間告訴使用者目前狀態即可。

## 第 7 步：結果處理

### 編譯成功

1. 建立資料夾 `%USERPROFILE%\Downloads\obj3d-builds\<YYYYMMDD-HHMM>\`。
2. `gh run download <databaseId> -n Obj3D-ipa -D <上面的資料夾>`，確認裡面有 `Obj3D.ipa`。
3. 用檔案總管開啟並選取這個檔案：`explorer /select,"<完整路徑>\Obj3D.ipa"`。
4. 嘗試開啟 Sideloadly（常見位置：`C:\Program Files\Sideloadly\sideloadly.exe`、`%LOCALAPPDATA%\Programs\Sideloadly\sideloadly.exe`）。找不到就略過。
5. 告訴使用者接下來手動完成：
   - 用 USB 連接 iPhone
   - 在 Sideloadly 選 iPhone、填**同一個 Apple ID**、把 `Obj3D.ipa` 拖進去、按 **Start**
   - Kaggle 那邊不用做任何事

### 編譯失敗

1. `gh run view <databaseId> --log-failed > %USERPROFILE%\Downloads\obj3d-builds\build-error-<時間>.txt`
2. 從記錄中找出含有 `error:` 的行（Swift 編譯錯誤會標示檔案與行號），整理成清單給使用者看。
3. 說明你判斷的原因與打算怎麼修，**問使用者是否同意修改**。
   - 使用者同意：只做讓編譯通過的最小修改，commit 訊息 `修正編譯錯誤：<摘要>`，push 後回到第 6 步。最多自動重試 3 次。
   - 使用者不同意或修不好：告訴使用者錯誤記錄檔的位置，請他把內容提供給原本協助開發的 Claude。

## 最後

用一個簡短的清單總結：這次更新了什麼、commit 編號、IPA 的位置、使用者接下來要做的事。
