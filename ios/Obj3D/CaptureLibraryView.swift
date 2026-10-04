import SwiftUI

struct CaptureLibraryView: View {
    @EnvironmentObject var job: JobManager
    @Environment(\.dismiss) private var dismiss
    @State private var items: [CaptureInfo] = []

    var body: some View {
        NavigationStack {
            List {
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
                                Text((c.mode == "turntable" ? "物體旋轉・" : "手機繞物體・") + "\(c.frames) 張影格" + (job.captureURL == c.url ? "（目前使用中）" : ""))
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
        }
    }
}
