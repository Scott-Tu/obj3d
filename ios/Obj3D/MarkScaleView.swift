import SwiftUI

/// 物體旋轉模式：在影格上標記比例尺兩端（必要）與物體位置（選填），寫回 meta.json
struct MarkScaleView: View {
    let captureURL: URL
    var onDone: () -> Void
    @Environment(\.dismiss) private var dismiss
    @AppStorage("scaleLengthCM") private var scaleLengthCM = 15.0

    @State private var meta: CaptureMeta?
    @State private var candidates: [Int] = []          // 可選的影格索引
    @State private var page = 0
    @State private var marks: [Int: [CGPoint]] = [:]    // 影格索引 → 比例尺端點（像素）
    @State private var objectTap: (Int, CGPoint)?
    @State private var tapMode = 0                      // 0 = 比例尺兩端，1 = 物體
    @State private var errorText = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 10) {
                Picker("", selection: $tapMode) {
                    Text("標記比例尺兩端").tag(0)
                    Text("點一下物體（選填）").tag(1)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                Text(tapMode == 0
                     ? "在比例尺（\(String(format: "%.0f", scaleLengthCM)) cm）的兩個端點各點一下。建議在 2 張不同角度的畫面上都標記。"
                     : "在物體上點一下，幫助程式找到要建模的物體。")
                    .font(.footnote).foregroundColor(.secondary)
                    .padding(.horizontal)

                if let meta = meta, !candidates.isEmpty {
                    TabView(selection: $page) {
                        ForEach(Array(candidates.enumerated()), id: \.offset) { idx, fi in
                            frameView(meta: meta, frameIndex: fi).tag(idx)
                        }
                    }
                    .tabViewStyle(.page(indexDisplayMode: .always))
                    .indexViewStyle(.page(backgroundDisplayMode: .always))
                } else {
                    Spacer()
                    Text(errorText.isEmpty ? "讀取中…" : errorText).foregroundColor(.secondary)
                    Spacer()
                }

                HStack {
                    Button("清除這張") {
                        if page < candidates.count {
                            let fi = candidates[page]
                            marks[fi] = nil
                            if objectTap?.0 == fi { objectTap = nil }
                        }
                    }
                    .buttonStyle(.bordered)
                    Spacer()
                    Text("已標記 \(completeCount) 張").font(.footnote).foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }
            .navigationTitle("標記比例尺")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("稍後") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { save() }.disabled(completeCount == 0)
                }
            }
            .onAppear(perform: load)
        }
    }

    private var completeCount: Int { marks.values.filter { $0.count == 2 }.count }

    @ViewBuilder
    private func frameView(meta: CaptureMeta, frameIndex fi: Int) -> some View {
        let url = captureURL.appendingPathComponent(meta.frames[fi].file)
        if let img = UIImage(contentsOfFile: url.path) {
            GeometryReader { geo in
                let rect = fitRect(imageSize: img.size, in: geo.size)
                ZStack(alignment: .topLeading) {
                    Image(uiImage: img).resizable().scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                    if let pts = marks[fi] {
                        ForEach(Array(pts.enumerated()), id: \.offset) { _, p in
                            Circle().stroke(Color.yellow, lineWidth: 3).frame(width: 22, height: 22)
                                .position(toView(p, img.size, rect))
                        }
                        if pts.count == 2 {
                            Path { path in
                                path.move(to: toView(pts[0], img.size, rect))
                                path.addLine(to: toView(pts[1], img.size, rect))
                            }
                            .stroke(Color.yellow, lineWidth: 2)
                        }
                    }
                    if let o = objectTap, o.0 == fi {
                        Circle().fill(Color.green).frame(width: 18, height: 18)
                            .overlay(Circle().stroke(Color.black, lineWidth: 2))
                            .position(toView(o.1, img.size, rect))
                    }
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onEnded { v in
                    guard rect.contains(v.location) else { return }
                    let p = CGPoint(x: (v.location.x - rect.minX) / rect.width * img.size.width,
                                    y: (v.location.y - rect.minY) / rect.height * img.size.height)
                    if tapMode == 1 {
                        objectTap = (fi, p)
                    } else {
                        var pts = marks[fi] ?? []
                        if pts.count >= 2 { pts = [] }
                        pts.append(p)
                        marks[fi] = pts
                    }
                })
            }
        } else {
            Text("無法讀取影格").foregroundColor(.secondary)
        }
    }

    private func fitRect(imageSize: CGSize, in size: CGSize) -> CGRect {
        let s = min(size.width / imageSize.width, size.height / imageSize.height)
        let w = imageSize.width * s, h = imageSize.height * s
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func toView(_ p: CGPoint, _ imageSize: CGSize, _ rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + p.x / imageSize.width * rect.width,
                y: rect.minY + p.y / imageSize.height * rect.height)
    }

    private func load() {
        let url = captureURL.appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: url),
              let m = try? JSONDecoder().decode(CaptureMeta.self, from: data), !m.frames.isEmpty else {
            errorText = "讀取錄影失敗"; return
        }
        meta = m
        let n = m.frames.count
        candidates = [0, n / 4, n / 2, (3 * n) / 4].map { min($0, n - 1) }
        var seen = Set<Int>()
        candidates = candidates.filter { seen.insert($0).inserted }
        for mk in m.scale_marks ?? [] {
            if let fi = m.frames.firstIndex(where: { $0.file == mk.frame }), mk.p1.count == 2, mk.p2.count == 2 {
                marks[fi] = [CGPoint(x: mk.p1[0], y: mk.p1[1]), CGPoint(x: mk.p2[0], y: mk.p2[1])]
                if !candidates.contains(fi) { candidates.append(fi) }
            }
        }
        if let op = m.object_point, let fi = m.frames.firstIndex(where: { $0.file == op.frame }), op.p.count == 2 {
            objectTap = (fi, CGPoint(x: op.p[0], y: op.p[1]))
        }
    }

    private func save() {
        guard var m = meta else { return }
        m.mode = "turntable"
        m.scale_length_cm = scaleLengthCM
        m.scale_marks = marks.compactMap { fi, pts in
            pts.count == 2 ? ScaleMark(frame: m.frames[fi].file, p1: [pts[0].x, pts[0].y], p2: [pts[1].x, pts[1].y]) : nil
        }
        if let o = objectTap {
            m.object_point = ObjectPoint(frame: m.frames[o.0].file, p: [o.1.x, o.1.y])
        }
        do {
            try JSONEncoder().encode(m).write(to: captureURL.appendingPathComponent("meta.json"))
            onDone()
            dismiss()
        } catch {
            errorText = "儲存失敗：\(error.localizedDescription)"
        }
    }
}
