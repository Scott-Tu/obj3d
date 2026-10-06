import SwiftUI

struct ContentView: View {
    @EnvironmentObject var job: JobManager
    @State private var showCapture = false
    @State private var showLibrary = false
    @State private var showRuler = true
    @State private var viewMode = 0          // 0 = 3DGS 擬真，1 = 網格（尺寸）

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    previewArea
                    statusCard
                    buttons
                }
                .padding()
            }
            .navigationTitle("3D 掃描")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink {
                        SettingsView()
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $job.showMarkScale) {
                if let dir = job.captureURL {
                    MarkScaleView(captureURL: dir) { job.marksSaved() }
                }
            }
            .sheet(isPresented: $showLibrary) {
                CaptureLibraryView().environmentObject(job)
            }
            .fullScreenCover(isPresented: $showCapture) {
                CaptureView(mode: "orbit") { url, jobId, count in
                    job.setCapture(url: url, jobId: jobId, frames: count)
                }
            }
        }
    }

    @ViewBuilder
    private var previewArea: some View {
        if let r = job.result {
            VStack(spacing: 10) {
                if r.splatURL != nil {
                    Picker("檢視", selection: $viewMode) {
                        Text("擬真（3DGS）").tag(0)
                        Text("網格（尺寸）").tag(1)
                    }
                    .pickerStyle(.segmented)
                }
                if r.splatURL != nil && viewMode == 0 {
                    let maxCM = (r.meta.size_cm ?? [10, 10, 10]).max() ?? 10
                    SplatViewer(resultDir: r.dir, cameraDistance: max(0.15, maxCM / 100 * 2.2),
                                lookHeight: ((r.meta.size_cm?.last ?? 10) / 100) * 0.45)
                        .id(r.jobId + "-gs")
                        .frame(height: 380)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    Text("單指旋轉、雙指縮放平移；第一次開啟需要網路載入檢視器")
                        .font(.caption2).foregroundColor(.secondary)
                } else {
                    ModelViewer(url: r.previewURL, showRuler: showRuler)
                        .id(r.jobId)
                        .frame(height: 380)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    Toggle("顯示公分尺規", isOn: $showRuler)
                        .padding(.horizontal, 4)
                }
                if !r.shareFiles.isEmpty {
                    ShareLink(items: r.shareFiles) {
                        Label("分享／儲存模型檔（GLB、STL、PLY）", systemImage: "square.and.arrow.up")
                    }
                }
                if !r.diagnosticImages.isEmpty {
                    DisclosureGroup("診斷圖（結果不理想時查看）") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("遮罩檢查：只有物體應該是亮的；綠圈＝程式自動找到的物體位置；標 BAD 的影格已略過。")
                                .font(.footnote).foregroundColor(.secondary)
                            ForEach(r.diagnosticImages, id: \.self) { url in
                                if let img = UIImage(contentsOfFile: url.path) {
                                    Image(uiImage: img)
                                        .resizable()
                                        .scaledToFit()
                                        .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            }
                            Text("相機軌跡：藍線＝拍攝路徑（應繞物體一圈），紅色＝物體，側視圖中物體應在 0 的水平線上方。")
                                .font(.footnote).foregroundColor(.secondary)
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 72, weight: .thin))
                    .foregroundColor(.secondary)
                Text("1. 錄影：讓物體在畫面中央，繞一圈\n2. 生成：上傳到 Kaggle 自動建模（約 20~60 分鐘）")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.leading)
            }
            .frame(maxWidth: .infinity, minHeight: 260)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
        }
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                if job.busy { ProgressView() }
                Text(job.status).font(.headline)
            }
            if !job.detail.isEmpty {
                Text(job.detail)
                    .font(.footnote)
                    .foregroundColor(job.phase == .failed ? .red : .secondary)
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private var buttons: some View {
        VStack(spacing: 12) {
            buttonStack
            Text(AppInfo.summary)
                .font(.caption2)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.top, 4)
        }
    }

    private var buttonStack: some View {
        VStack(spacing: 12) {
            Text("物體不動，拿著手機繞物體慢慢走一圈（再從斜上方一圈）。可在畫面中放一支已知長度的比例尺，錄完標記兩端讓尺寸更準。")
                .font(.footnote).foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                showCapture = true
            } label: {
                Label("錄影", systemImage: "video.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(job.busy)

            Button {
                job.generate()
            } label: {
                Label("生成 3D 模型", systemImage: "cube.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .controlSize(.large)
            .disabled(job.busy || job.captureURL == nil)

            if job.captureURL != nil {
                Button {
                    job.showMarkScale = true
                } label: {
                    Label(job.captureHasMarks ? "重新標記比例尺" : "標記比例尺（建議）", systemImage: "ruler")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(job.busy)
            }

            Button {
                showLibrary = true
            } label: {
                Label("從錄影存檔載入", systemImage: "folder").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(job.busy)

            if job.busy {
                Button("暫停查詢（Kaggle 會繼續運算）") { job.stopWaiting() }
                    .buttonStyle(.bordered)
            } else if job.canResume {
                Button {
                    job.resume()
                } label: {
                    Label("繼續查詢上一次的工作", systemImage: "arrow.clockwise").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }
    }
}
