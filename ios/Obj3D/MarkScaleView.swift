import SwiftUI

/// 在影格上標記比例尺兩端與物體位置，寫回 meta.json。
/// 按住拖曳時顯示放大鏡＋十字準心，放開才放下點；拖曳已有的點可以微調；滑桿可選任一張畫面。
struct MarkScaleView: View {
    let captureURL: URL
    var onDone: () -> Void
    @Environment(\.dismiss) private var dismiss
    @AppStorage("scaleLengthCM") private var scaleLengthCM = 15.0

    @State private var meta: CaptureMeta?
    @State private var frameIndex = 0
    @State private var marks: [Int: [CGPoint]] = [:]    // 影格 → 比例尺兩端（原始影像像素）
    @State private var objectTap: (Int, CGPoint)?
    @State private var tapMode = 0                      // 0 = 比例尺兩端，1 = 物體
    @State private var errorText = ""
    @State private var image: UIImage?
    // 拖曳中的狀態
    @State private var dragLoc: CGPoint?
    @State private var editIndex: Int?                  // 正在拖曳的既有點（nil = 新增）

    private let loupeSize: CGFloat = 140
    private let zoom: CGFloat = 3.5

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                Picker("", selection: $tapMode) {
                    Text("比例尺兩端").tag(0)
                    Text("物體（選填）").tag(1)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                Text(tapMode == 0
                     ? "按住畫面拖曳，用放大鏡的十字對準比例尺（\(String(format: "%.1f", scaleLengthCM)) cm）端點後放開；拖曳已有的點可以微調。建議在 2~4 張不同角度的畫面標記。"
                     : "按住拖曳，十字對準物體後放開。")
                    .font(.footnote).foregroundColor(.secondary)
                    .padding(.horizontal)

                if let meta = meta {
                    frameView(meta: meta)
                    HStack {
                        Button { step(-1) } label: { Image(systemName: "chevron.left") }
                            .buttonStyle(.bordered).disabled(frameIndex == 0)
                        Slider(value: Binding(get: { Double(frameIndex) },
                                              set: { frameIndex = Int($0.rounded()); loadImage() }),
                               in: 0...Double(max(meta.frames.count - 1, 1)), step: 1)
                        Button { step(1) } label: { Image(systemName: "chevron.right") }
                            .buttonStyle(.bordered).disabled(frameIndex >= meta.frames.count - 1)
                    }
                    .padding(.horizontal)
                    HStack {
                        Text("第 \(frameIndex + 1) / \(meta.frames.count) 張")
                        Spacer()
                        Text("已標記 \(completeCount) 張" + (markedFrames.isEmpty ? "" : "（\(markedFrames.map { "\($0 + 1)" }.joined(separator: "、"))）"))
                    }
                    .font(.footnote).foregroundColor(.secondary).padding(.horizontal)
                    HStack {
                        Button("清除這張") {
                            marks[frameIndex] = nil
                            if objectTap?.0 == frameIndex { objectTap = nil }
                        }
                        .buttonStyle(.bordered)
                        Spacer()
                        if !markedFrames.isEmpty {
                            Menu("跳到已標記") {
                                ForEach(markedFrames, id: \.self) { f in
                                    Button("第 \(f + 1) 張") { frameIndex = f; loadImage() }
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                } else {
                    Spacer()
                    Text(errorText.isEmpty ? "讀取中…" : errorText).foregroundColor(.secondary)
                    Spacer()
                }
            }
            .navigationTitle("標記比例尺")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("稍後") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { save() }.disabled(completeCount == 0 && objectTap == nil)
                }
            }
            .onAppear(perform: load)
        }
    }

    private var completeCount: Int { marks.values.filter { $0.count == 2 }.count }
    private var markedFrames: [Int] { marks.filter { $0.value.count == 2 }.keys.sorted() }

    private func step(_ d: Int) {
        guard let meta = meta else { return }
        frameIndex = min(max(frameIndex + d, 0), meta.frames.count - 1)
        loadImage()
    }

    private func loadImage() {
        guard let meta = meta, frameIndex < meta.frames.count else { return }
        image = UIImage(contentsOfFile: captureURL.appendingPathComponent(meta.frames[frameIndex].file).path)
    }

    @ViewBuilder
    private func frameView(meta: CaptureMeta) -> some View {
        if let img = image {
            GeometryReader { geo in
                let rect = fitRect(imageSize: img.size, in: geo.size)
                let fi = frameIndex
                ZStack(alignment: .topLeading) {
                    Image(uiImage: img).resizable().scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                    if let pts = marks[fi] {
                        if pts.count == 2 {
                            Path { p in
                                p.move(to: toView(pts[0], img.size, rect)); p.addLine(to: toView(pts[1], img.size, rect))
                            }
                            .stroke(Color.yellow, lineWidth: 2)
                        }
                        ForEach(Array(pts.enumerated()), id: \.offset) { k, p in
                            Circle().stroke(editIndex == k && dragLoc != nil ? Color.orange : Color.yellow, lineWidth: 3)
                                .frame(width: 22, height: 22)
                                .position(toView(p, img.size, rect))
                        }
                    }
                    if let o = objectTap, o.0 == fi {
                        Circle().fill(Color.green).frame(width: 16, height: 16)
                            .overlay(Circle().stroke(Color.black, lineWidth: 2))
                            .position(toView(o.1, img.size, rect))
                    }
                    if let loc = dragLoc {
                        loupe(img: img, rect: rect, at: loc, container: geo.size)
                    }
                }
                .contentShape(Rectangle())
                .highPriorityGesture(DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        if dragLoc == nil {                     // 剛按下：靠近既有的點就是「移動那個點」
                            editIndex = nil
                            if tapMode == 0, let pts = marks[fi] {
                                for (k, p) in pts.enumerated() where dist(toView(p, img.size, rect), v.startLocation) < 30 {
                                    editIndex = k
                                }
                            } else if tapMode == 1, let o = objectTap, o.0 == fi,
                                      dist(toView(o.1, img.size, rect), v.startLocation) < 30 {
                                editIndex = 0
                            }
                        }
                        dragLoc = clamp(v.location, rect)
                    }
                    .onEnded { v in
                        let p = toImage(clamp(v.location, rect), img.size, rect)
                        if tapMode == 1 {
                            objectTap = (fi, p)
                        } else {
                            var pts = marks[fi] ?? []
                            if let k = editIndex, k < pts.count {
                                pts[k] = p
                            } else {
                                if pts.count >= 2 { pts = [] }
                                pts.append(p)
                            }
                            marks[fi] = pts
                        }
                        dragLoc = nil; editIndex = nil
                    })
            }
        } else {
            Text("無法讀取影格").foregroundColor(.secondary).frame(maxHeight: .infinity)
        }
    }

    /// 放大鏡：以手指位置為中心放大，十字準心＝放開時的位置；顯示在手指上方（太靠上時改到下方）
    private func loupe(img: UIImage, rect: CGRect, at loc: CGPoint, container: CGSize) -> some View {
        let L = loupeSize
        let above = loc.y - L * 0.95 - L / 2 > 0
        let cx = min(max(loc.x, L / 2 + 4), container.width - L / 2 - 4)
        let cy = above ? loc.y - L * 0.95 : loc.y + L * 0.95
        return ZStack(alignment: .topLeading) {
            Image(uiImage: img).resizable()
                .frame(width: rect.width * zoom, height: rect.height * zoom)
                .offset(x: L / 2 - (loc.x - rect.minX) * zoom, y: L / 2 - (loc.y - rect.minY) * zoom)
            Path { p in                                    // 十字準心（中間留空，不擋住目標）
                p.move(to: CGPoint(x: L / 2, y: 8)); p.addLine(to: CGPoint(x: L / 2, y: L / 2 - 7))
                p.move(to: CGPoint(x: L / 2, y: L / 2 + 7)); p.addLine(to: CGPoint(x: L / 2, y: L - 8))
                p.move(to: CGPoint(x: 8, y: L / 2)); p.addLine(to: CGPoint(x: L / 2 - 7, y: L / 2))
                p.move(to: CGPoint(x: L / 2 + 7, y: L / 2)); p.addLine(to: CGPoint(x: L - 8, y: L / 2))
            }
            .stroke(Color.red, lineWidth: 1.5)
            Circle().stroke(Color.red, lineWidth: 1).frame(width: 6, height: 6)
                .position(x: L / 2, y: L / 2)
        }
        .frame(width: L, height: L)
        .clipShape(Circle())
        .overlay(Circle().stroke(Color.white, lineWidth: 3))
        .shadow(radius: 4)
        .position(x: cx, y: cy)
        .allowsHitTesting(false)
    }

    private func dist(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(a.x - b.x, a.y - b.y) }

    private func clamp(_ p: CGPoint, _ r: CGRect) -> CGPoint {
        CGPoint(x: min(max(p.x, r.minX), r.maxX), y: min(max(p.y, r.minY), r.maxY))
    }

    private func fitRect(imageSize: CGSize, in size: CGSize) -> CGRect {
        let s = min(size.width / imageSize.width, size.height / imageSize.height)
        let w = imageSize.width * s, h = imageSize.height * s
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func toView(_ p: CGPoint, _ imageSize: CGSize, _ rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + p.x / imageSize.width * rect.width, y: rect.minY + p.y / imageSize.height * rect.height)
    }

    private func toImage(_ v: CGPoint, _ imageSize: CGSize, _ rect: CGRect) -> CGPoint {
        CGPoint(x: (v.x - rect.minX) / rect.width * imageSize.width, y: (v.y - rect.minY) / rect.height * imageSize.height)
    }

    private func load() {
        let url = captureURL.appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: url),
              let m = try? JSONDecoder().decode(CaptureMeta.self, from: data), !m.frames.isEmpty else {
            errorText = "讀取錄影失敗"; return
        }
        meta = m
        for mk in m.scale_marks ?? [] {
            if let fi = m.frames.firstIndex(where: { $0.file == mk.frame }), mk.p1.count == 2, mk.p2.count == 2 {
                marks[fi] = [CGPoint(x: mk.p1[0], y: mk.p1[1]), CGPoint(x: mk.p2[0], y: mk.p2[1])]
            }
        }
        if let op = m.object_point, let fi = m.frames.firstIndex(where: { $0.file == op.frame }), op.p.count == 2 {
            objectTap = (fi, CGPoint(x: op.p[0], y: op.p[1]))
        }
        frameIndex = markedFrames.first ?? m.frames.count / 4
        loadImage()
    }

    private func save() {
        guard var m = meta else { return }
        if m.mode == nil { m.mode = "orbit" }
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
