import SwiftUI
import PhotosUI

struct CaptureLibraryView: View {
    @EnvironmentObject var job: JobManager
    @Environment(\.dismiss) private var dismiss
    @State private var items: [CaptureInfo] = []
    @State private var pickerItem: PhotosPickerItem?
    @State private var importing = false
    @State private var importProgress = 0.0
    @State private var importError = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    PhotosPicker(selection: $pickerItem, matching: .videos) {
                        Label("從「照片」匯入影片", systemImage: "photo.on.rectangle")
                    }
                    .disabled(importing)
                    if importing {
                        ProgressView(value: importProgress) { Text("擷取影格中…") }
                    }
                    if !importError.isEmpty {
                        Text(importError).font(.footnote).foregroundColor(.red)
                    }
                } footer: {
                    Text("匯入的影片沒有手機動作資料，會用「物體旋轉」的方法計算（手機繞物體或物體旋轉拍的影片都可以）。畫面中需要有已知長度的比例尺，匯入後要標記兩端。")
                }
                if items.isEmpty {
                    Text("還沒有存檔的錄影").foregroundColor(.secondary)
                }
                ForEach(items) { c in
                    Button {
                        job.setCapture(url: c.url, jobId: c.url.lastPathComponent, frames: c.frames)
                        dismiss()
                    } label: {
                        HStack(spacing: 14) {
                            if let img = CaptureStore.thumbnail(c) {
                                Image(uiImage: img).resizable().scaledToFill()
                                    .frame(width: 54, height: 72).clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text(c.date.formatted(date: .abbreviated, time: .shortened)).font(.headline)
                                Text((c.mode == "turntable" ? "物體旋轉・" : (c.mode == "video" ? "匯入影片・" : "手機繞物體・")) + "\(c.frames) 張影格" + (job.captureURL == c.url ? "（目前使用中）" : ""))
                                    .font(.footnote).foregroundColor(.secondary)
                            }
                        }
                    }
                    .foregroundColor(.primary)
                }
                .onDelete { idx in
                    for i in idx { CaptureStore.delete(items[i]) }
                    items = CaptureStore.list()
                }
            }
            .navigationTitle("錄影存檔")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("關閉") { dismiss() } }
            }
            .onAppear { items = CaptureStore.list() }
            .onChange(of: pickerItem) { item in
                guard let item = item else { return }
                Task { await doImport(item) }
            }
        }
    }

    @MainActor
    private func doImport(_ item: PhotosPickerItem) async {
        importing = true; importProgress = 0; importError = ""
        defer { importing = false; pickerItem = nil }
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                importError = "無法讀取這段影片"; return
            }
            let fps = DevParams.value(forKey: "capture_fps")
            let (dir, n) = try await VideoImporter.importVideo(movie.url, fps: fps > 0 ? fps : 3) { p in
                Task { @MainActor in importProgress = p }
            }
            items = CaptureStore.list()
            job.setCapture(url: dir, jobId: dir.lastPathComponent, frames: n)
            dismiss()
        } catch {
            importError = "匯入失敗：\(error.localizedDescription)"
        }
    }
}
