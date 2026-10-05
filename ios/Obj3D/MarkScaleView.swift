import SwiftUI

/// 物體旋轉模式：在影格上標記比例尺兩端（必要）與物體位置（選填），寫回 meta.json
struct MarkScaleView: View {
    let captureURL: URL
    var onDone: () -> Void
    @Environment(\.dismiss) private var dismiss
    @AppStorage("scaleLengthCM") private var scaleLengthCM = 15.0

    @State private var meta: CaptureMeta?
    @State private var current = 0                      // 目前顯示的影格索引（可用滑桿選任一張）
    @State private var marks: [Int: [CGPoint]] = [:]    // 影格索引 → 比例尺端點（像素）
    @State private var objectTap: (Int, CGPoint)?
    @State private var tapMode = 0                      // 0 = 比例尺兩端，1 = 物體
    @State private var errorText = ""
    @State private var drag: DragState?                 // 手指按住時的位置（顯示放大鏡）
    @State private var imageCache = NSCache<NSNumber, UIImage>()

    /// 拖曳中的點：editIndex 不是 nil 時代表正在移動既有的端點
    private struct DragState {
        var frame: Int
        var editIndex: Int?
        var point: CGPoint          // 影像像素座標
        var touch: CGPoint          // 畫面座標（決定放大鏡位置）
    }

    private let loupeSize: CGFloat = 130
    private let loupeZoom: CGFloat = 4
    private let grabRadius: CGFloat = 32               // 手指落在既有端點附近多少點以內就改成拖動它

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
                     ? "在比例尺（\(String(format: "%.0f", scaleLengthCM)) cm）的兩個端點各點一下；按住會出現放大鏡，拖到準確位置再放開。已標的端點可以直接拖動微調。建議在 2～3 張不同角度的畫面上都標記。"
                     : "在物體上點一下，幫助程式找到要建模的物體。建議選物體清楚、角度和大部分影格相近的畫面。")
                    .font(.footnote).foregroundColor(.secondary)
                    .padding(.horizontal)

                if let meta = meta {
                    frameView(meta: meta, frameIndex: current)
                    frameSelector(count: meta.frames.count)
                } else {
                    Spacer()
                    Text(errorText.isEmpty ? "讀取中…" : errorText).foregroundColor(.secondary)
                    Spacer()
                }

                HStack {
                    Button("清除這張") {
                        marks[current] = nil
                        if objectTap?.0 == current { objectTap = nil }
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

    /// 已標記（比例尺或物體）的影格，方便跳回去檢查
    private var markedFrames: [Int] {
        var s = Set(marks.filter { !$0.value.isEmpty }.keys)
        if let o = objectTap { s.insert(o.0) }
        return s.sorted()
    }

    // MARK: - 影格選擇（滑桿＋上一張／下一張＋已標記影格）

    @ViewBuilder
    private func frameSelector(count n: Int) -> some View {
        VStack(spacing: 6) {
            HStack(spacing: 12) {
                Button { current = max(0, current - 1) } label: { Image(systemName: "chevron.left") }
                    .disabled(current == 0)
                if n > 1 {
                    Slider(value: Binding(get: { Double(current) },
                                          set: { current = min(n - 1, max(0, Int($0.rounded()))) }),
                           in: 0...Double(n - 1), step: 1)
                } else {
                    Spacer()
                }
                Button { current = min(n - 1, current + 1) } label: { Image(systemName: "chevron.right") }
                    .disabled(current >= n - 1)
            }
            HStack {
                Text("第 \(current + 1) / \(n) 張").font(.footnote.monospacedDigit())
                Spacer()
                if !markedFrames.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(markedFrames, id: \.self) { fi in
                                Button("\(fi + 1)") { current = fi }
                                    .font(.caption)
                                    .buttonStyle(.bordered)
                                    .tint(fi == current ? Color.accentColor : Color.gray)
                            }
                        }
                    }
                }
            }
        }
        .padding(.horizontal)
    }

    // MARK: - 影格與標記

    private func image(meta: CaptureMeta, frameIndex fi: Int) -> UIImage? {
        if let img = imageCache.object(forKey: NSNumber(value: fi)) { return img }
        let url = captureURL.appendingPathComponent(meta.frames[fi].file)
        guard let img = UIImage(contentsOfFile: url.path) else { return nil }
        imageCache.setObject(img, forKey: NSNumber(value: fi))
        return img
    }

    @ViewBuilder
    private func frameView(meta: CaptureMeta, frameIndex fi: Int) -> some View {
        if let img = image(meta: meta, frameIndex: fi) {
            GeometryReader { geo in
                let rect = fitRect(imageSize: img.size, in: geo.size)
                ZStack(alignment: .topLeading) {
                    Image(uiImage: img).resizable().scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                    let pts = displayedMarks(fi)
                    ForEach(Array(pts.enumerated()), id: \.offset) { _, p in
                        Circle().stroke(Color.yellow, lineWidth: 2).frame(width: 18, height: 18)
                            .overlay(crosshair(size: 18, color: .yellow))
                            .position(toView(p, img.size, rect))
                    }
                    if pts.count == 2 {
                        Path { path in
                            path.move(to: toView(pts[0], img.size, rect))
                            path.addLine(to: toView(pts[1], img.size, rect))
                        }
                        .stroke(Color.yellow, lineWidth: 1.5)
                    }
                    if let o = displayedObject(fi) {
                        Circle().fill(Color.green).frame(width: 14, height: 14)
                            .overlay(Circle().stroke(Color.black, lineWidth: 2))
                            .position(toView(o, img.size, rect))
                    }
                    if let d = drag, d.frame == fi {
                        loupe(img: img, rect: rect, point: d.point)
                            .position(loupePosition(touch: d.touch, in: geo.size))
                    }
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        let loc = clamp(v.location, to: rect)
                        let p = toImage(loc, img.size, rect)
                        if var d = drag, d.frame == fi {
                            d.point = p; d.touch = loc
                            drag = d
                        } else {
                            guard rect.contains(v.startLocation) else { return }
                            drag = DragState(frame: fi, editIndex: grabbedIndex(at: v.startLocation, frame: fi, img.size, rect),
                                             point: p, touch: loc)
                        }
                    }
                    .onEnded { _ in
                        guard let d = drag, d.frame == fi else { drag = nil; return }
                        commit(d)
                        drag = nil
                    })
            }
        } else {
            Text("無法讀取影格").foregroundColor(.secondary)
        }
    }

    /// 拖曳中也即時顯示新位置
    private func displayedMarks(_ fi: Int) -> [CGPoint] {
        var pts = marks[fi] ?? []
        guard tapMode == 0, let d = drag, d.frame == fi else { return pts }
        if let k = d.editIndex, k < pts.count {
            pts[k] = d.point
        } else {
            if pts.count >= 2 { pts = [] }
            pts.append(d.point)
        }
        return pts
    }

    private func displayedObject(_ fi: Int) -> CGPoint? {
        if tapMode == 1, let d = drag, d.frame == fi { return d.point }
        if let o = objectTap, o.0 == fi { return o.1 }
        return nil
    }

    private func grabbedIndex(at loc: CGPoint, frame fi: Int, _ imageSize: CGSize, _ rect: CGRect) -> Int? {
        guard tapMode == 0, let pts = marks[fi] else { return nil }
        var best: (Int, CGFloat)?
        for (k, p) in pts.enumerated() {
            let v = toView(p, imageSize, rect)
            let dist = hypot(v.x - loc.x, v.y - loc.y)
            if dist <= grabRadius, dist < (best?.1 ?? .infinity) { best = (k, dist) }
        }
        return best?.0
    }

    private func commit(_ d: DragState) {
        if tapMode == 1 {
            objectTap = (d.frame, d.point)
            return
        }
        var pts = marks[d.frame] ?? []
        if let k = d.editIndex, k < pts.count {
            pts[k] = d.point
        } else {
            if pts.count >= 2 { pts = [] }
            pts.append(d.point)
        }
        marks[d.frame] = pts
    }

    // MARK: - 放大鏡

    private func loupe(img: UIImage, rect: CGRect, point p: CGPoint) -> some View {
        let w = rect.width * loupeZoom, h = rect.height * loupeZoom
        let px = p.x / img.size.width * w, py = p.y / img.size.height * h
        return Image(uiImage: img).resizable()
            .frame(width: w, height: h)
            .offset(x: loupeSize / 2 - px, y: loupeSize / 2 - py)
            .frame(width: loupeSize, height: loupeSize, alignment: .topLeading)
            .clipShape(Circle())
            .overlay(crosshair(size: loupeSize, color: tapMode == 0 ? .yellow : .green))
            .overlay(Circle().stroke(Color.white, lineWidth: 3))
            .shadow(radius: 4)
            .allowsHitTesting(false)
    }

    private func crosshair(size: CGFloat, color: Color) -> some View {
        Path { path in
            path.move(to: CGPoint(x: size / 2, y: 0)); path.addLine(to: CGPoint(x: size / 2, y: size))
            path.move(to: CGPoint(x: 0, y: size / 2)); path.addLine(to: CGPoint(x: size, y: size / 2))
        }
        .stroke(color, lineWidth: 1)
        .frame(width: size, height: size)
    }

    /// 放大鏡放在手指上方，避免被手指擋住；太靠上時改放在下方
    private func loupePosition(touch: CGPoint, in size: CGSize) -> CGPoint {
        let gap = loupeSize * 0.5 + 50
        var y = touch.y - gap
        if y < loupeSize / 2 { y = touch.y + gap }
        let x = min(max(touch.x, loupeSize / 2), size.width - loupeSize / 2)
        return CGPoint(x: x, y: y)
    }

    // MARK: - 座標換算

    private func fitRect(imageSize: CGSize, in size: CGSize) -> CGRect {
        let s = min(size.width / imageSize.width, size.height / imageSize.height)
        let w = imageSize.width * s, h = imageSize.height * s
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func toView(_ p: CGPoint, _ imageSize: CGSize, _ rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + p.x / imageSize.width * rect.width,
                y: rect.minY + p.y / imageSize.height * rect.height)
    }

    private func toImage(_ v: CGPoint, _ imageSize: CGSize, _ rect: CGRect) -> CGPoint {
        CGPoint(x: (v.x - rect.minX) / rect.width * imageSize.width,
                y: (v.y - rect.minY) / rect.height * imageSize.height)
    }

    private func clamp(_ v: CGPoint, to rect: CGRect) -> CGPoint {
        CGPoint(x: min(max(v.x, rect.minX), rect.maxX), y: min(max(v.y, rect.minY), rect.maxY))
    }

    // MARK: - 讀寫 meta.json

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
        current = markedFrames.first ?? 0
    }

    private func save() {
        guard var m = meta else { return }
        if m.mode != "video" { m.mode = "turntable" }
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
