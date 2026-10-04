import SwiftUI

/// 開發者參數清單：在這裡加一行，設定畫面就會自動出現，並在「生成」時送到 Kaggle 的 runner.py。
/// 數值 0 代表「自動」（使用 runner.py 裡的預設值）。
struct DevOption: Hashable {
    let label: String
    let value: Double
}

struct DevParam: Identifiable {
    let key: String            // 與 runner.py 讀取的名稱相同
    let title: String
    let section: String
    let help: String
    let isToggle: Bool
    let defaultValue: Double
    let options: [DevOption]
    var id: String { key }

    static func choice(_ key: String, _ title: String, _ section: String, _ help: String,
                       _ opts: [(String, Double)], default def: Double = 0) -> DevParam {
        DevParam(key: key, title: title, section: section, help: help, isToggle: false, defaultValue: def,
                 options: opts.map { DevOption(label: $0.0, value: $0.1) })
    }

    static func toggle(_ key: String, _ title: String, _ section: String, _ help: String, default on: Bool = true) -> DevParam {
        DevParam(key: key, title: title, section: section, help: help, isToggle: true, defaultValue: on ? 1 : 0, options: [])
    }
}

enum DevParams {
    static let all: [DevParam] = [
        // 錄影（手機端）
        .choice("capture_fps", "錄影每秒張數", "錄影", "建模用影格的擷取速度；越多越完整，但上傳檔案越大",
                [("2 張", 2), ("3 張（預設）", 3), ("4 張", 4)], default: 3),

        // 影格與 AI 推論
        .choice("batches", "VGGT 分批數", "影格與推論", "每批約 50 張，分批後用 ARKit＋ICP 合併",
                [("自動（3）", 0), ("1", 1), ("2", 2), ("3", 3)]),
        .choice("max_frames_per_batch", "每批張數", "影格與推論", "T4 記憶體約可 50 張；不足時調低",
                [("自動", 0), ("30", 30), ("40", 40), ("50", 50)]),
        .toggle("use_hires", "高解析第二次推論", "影格與推論", "裁切物體附近再推論一次，細節較多但可能帶來變形"),
        .choice("conf_drop_percent", "丟棄低信心點", "影格與推論", "丟掉 VGGT 信心度最低的百分比",
                [("自動（40%）", 0), ("20%", 20), ("30%", 30), ("50%", 50), ("60%", 60)]),
        .choice("mask_erode_px", "遮罩內縮", "影格與推論", "物體邊緣內縮像素，避免沾到背景",
                [("自動（4）", 0), ("2", 2), ("6", 6), ("8", 8)]),

        // 深度融合
        .toggle("use_tsdf", "TSDF 融合", "深度融合", "把所有畫面的深度平均後再建表面"),
        .toggle("depth_refine", "逐張深度校正", "深度融合", "把每張深度圖對齊到共同表面後再融合一次"),
        .choice("tsdf_trunc_mm", "TSDF 截斷距離", "深度融合", "越大越能平均掉各畫面的誤差，但太大會讓薄壁互相干擾",
                [("自動", 0), ("6 mm", 6), ("8 mm", 8), ("12 mm", 12), ("16 mm", 16), ("20 mm", 20)]),
        .choice("tsdf_max_res", "TSDF 格子數上限", "深度融合", "每邊的格子數；越大越細、越吃記憶體",
                [("自動（384）", 0), ("256", 256), ("512", 512)]),
        .choice("consistency_tol_pct", "多視角一致性容許誤差", "深度融合", "點在其他視角的深度誤差小於這個比例才算一致",
                [("自動（2.5%）", 0), ("1.5%", 1.5), ("4%", 4), ("6%", 6)]),
        .choice("hr_tol_pct", "高解析深度容許誤差", "深度融合", "第二次推論與第一次深度差超過此比例就丟棄",
                [("自動（4%）", 0), ("2%", 2), ("6%", 6), ("10%", 10)]),

        // 表面
        .choice("poisson_depth", "Poisson 深度", "表面", "曲面重建的細緻度",
                [("自動（9）", 0), ("8", 8), ("10", 10)]),
        .choice("solid_voxel_mm", "實體解析度", "表面", "封閉實體的格子大小；越大越平滑、細節越少",
                [("自動", 0), ("0.75 mm", 0.75), ("1.0 mm", 1.0), ("1.5 mm", 1.5), ("2.0 mm", 2.0)]),
        .choice("extra_smooth_iters", "額外平滑次數", "表面", "在「表面平滑度」之外再做的平滑次數",
                [("自動", 0), ("0 次", 0.5), ("5 次", 5), ("10 次", 10), ("20 次", 20), ("40 次", 40)]),

        // 底座與貼圖
        .toggle("use_base", "加底座", "底座與貼圖", "關閉時只輸出物體本身（底部切平封閉）"),
        .choice("base_margin_cm", "底座外擴", "底座與貼圖", "底座比物體外框多出的距離",
                [("自動", 0), ("0.5 cm", 0.5), ("1 cm", 1), ("2 cm", 2), ("3 cm", 3)]),
        .toggle("use_texture", "貼圖", "底座與貼圖", "關閉時改用頂點顏色"),
        .choice("texture_size", "貼圖解析度", "底座與貼圖", "越大越清晰，檔案也越大",
                [("自動（2048）", 0), ("1024", 1024), ("4096", 4096)]),
    ]

    static var sections: [String] {
        var seen: [String] = []
        for p in all where !seen.contains(p.section) { seen.append(p.section) }
        return seen
    }

    private static func storageKey(_ p: DevParam) -> String { "dev_" + p.key }

    static func value(_ p: DevParam) -> Double {
        UserDefaults.standard.object(forKey: storageKey(p)) as? Double ?? p.defaultValue
    }

    static func set(_ p: DevParam, _ v: Double) {
        UserDefaults.standard.set(v, forKey: storageKey(p))
    }

    static func value(forKey key: String) -> Double {
        guard let p = all.first(where: { $0.key == key }) else { return 0 }
        return value(p)
    }

    static func resetAll() {
        for p in all { UserDefaults.standard.removeObject(forKey: storageKey(p)) }
    }

    /// 送到 runner.py 的 JSON
    static func json() -> String {
        var dict: [String: Any] = [:]
        for p in all { dict[p.key] = p.isToggle ? (value(p) != 0) : value(p) }
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let s = String(data: data, encoding: .utf8) else { return "{}" }
        return s
    }

    /// 和預設值不同的項目數
    static var changedCount: Int {
        all.filter { value($0) != $0.defaultValue }.count
    }
}

struct DevParamsView: View {
    @State private var refresh = 0

    var body: some View {
        Form {
            Section(footer: Text("下次按「生成」時送到 Kaggle，不用重新編譯 App。用「從錄影存檔載入」對同一段錄影比較不同參數；結果卡片會列出這次改過的參數。")) {
                Button("全部恢復預設") {
                    DevParams.resetAll()
                    refresh += 1
                }
            }
            ForEach(DevParams.sections, id: \.self) { section in
                Section(header: Text(section)) {
                    ForEach(DevParams.all.filter { $0.section == section }) { p in
                        row(p)
                    }
                }
            }
        }
        .id(refresh)
        .navigationTitle("開發者參數")
    }

    @ViewBuilder
    private func row(_ p: DevParam) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if p.isToggle {
                Toggle(p.title, isOn: Binding(
                    get: { DevParams.value(p) != 0 },
                    set: { DevParams.set(p, $0 ? 1 : 0); refresh += 1 }))
            } else {
                Picker(p.title, selection: Binding(
                    get: { DevParams.value(p) },
                    set: { DevParams.set(p, $0); refresh += 1 })) {
                    ForEach(p.options, id: \.self) { o in
                        Text(o.label).tag(o.value)
                    }
                }
            }
            Text(p.help).font(.caption).foregroundColor(.secondary)
        }
    }
}
