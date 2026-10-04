import SwiftUI
import SceneKit

/// 顯示 Kaggle 產生的 preview.bin（可用手指旋轉、縮放），可疊加公分尺規
struct ModelViewer: UIViewRepresentable {
    let url: URL
    var showRuler: Bool = true

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = UIColor.secondarySystemBackground
        view.antialiasingMode = .multisampling4X
        let (scene, camera) = PreviewLoader.load(url)
        view.scene = scene
        if let camera = camera { view.pointOfView = camera }
        scene.rootNode.childNode(withName: "ruler", recursively: false)?.isHidden = !showRuler
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        uiView.scene?.rootNode.childNode(withName: "ruler", recursively: false)?.isHidden = !showRuler
    }
}

enum PreviewLoader {
    /// 兩種格式（單位：公尺，y 軸朝上，桌面 y = 0）：
    /// 'O3DP'：u32 頂點數、u32 面數、f32 位置、f32 法向量、f32 顏色、u32 索引（頂點顏色）
    /// 'O3DT'：u32 頂點數、u32 面數、f32 位置、f32 法向量、f32 uv、u32 索引；貼圖為同資料夾的 preview_tex.jpg
    static func load(_ url: URL) -> (SCNScene, SCNNode?) {
        let scene = SCNScene()
        guard let data = try? Data(contentsOf: url), data.count > 12,
              let magic = String(data: data.prefix(4), encoding: .ascii),
              magic == "O3DP" || magic == "O3DT" else { return (scene, nil) }
        let textured = magic == "O3DT"
        let nV = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) })
        let nF = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self) })
        let vBytes = nV * 12
        let posOff = 12
        let nrmOff = posOff + vBytes
        let thirdOff = nrmOff + vBytes                       // 顏色（O3DP）或 uv（O3DT）
        let thirdBytes = textured ? nV * 8 : vBytes
        let idxOff = thirdOff + thirdBytes
        let end = idxOff + nF * 12
        guard nV > 0, nF > 0, data.count >= end else { return (scene, nil) }

        func source(_ off: Int, _ semantic: SCNGeometrySource.Semantic, _ comps: Int) -> SCNGeometrySource {
            SCNGeometrySource(data: data.subdata(in: off..<(off + nV * comps * 4)), semantic: semantic,
                              vectorCount: nV, usesFloatComponents: true, componentsPerVector: comps,
                              bytesPerComponent: 4, dataOffset: 0, dataStride: comps * 4)
        }
        let element = SCNGeometryElement(data: data.subdata(in: idxOff..<end), primitiveType: .triangles,
                                         primitiveCount: nF, bytesPerIndex: 4)
        let third = textured ? source(thirdOff, .texcoord, 2) : source(thirdOff, .color, 3)
        let geometry = SCNGeometry(sources: [source(posOff, .vertex, 3), source(nrmOff, .normal, 3), third],
                                   elements: [element])
        let material = SCNMaterial()
        material.isDoubleSided = true
        if textured,
           let img = UIImage(contentsOfFile: url.deletingLastPathComponent().appendingPathComponent("preview_tex.jpg").path) {
            material.diffuse.contents = img
            material.diffuse.mipFilter = .linear
            material.lightingModel = .lambert
        } else {
            material.diffuse.contents = UIColor.white
            material.lightingModel = .blinn
        }
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        let (minV, maxV) = node.boundingBox
        let cx = (minV.x + maxV.x) / 2, cz = (minV.z + maxV.z) / 2
        node.position = SCNVector3(-cx, 0, -cz)            // 水平置中，桌面維持在 y = 0
        scene.rootNode.addChildNode(node)

        let objMin = SCNVector3(minV.x - cx, minV.y, minV.z - cz)
        let objMax = SCNVector3(maxV.x - cx, maxV.y, maxV.z - cz)
        scene.rootNode.addChildNode(RulerBuilder.make(min: objMin, max: objMax))

        let size = max(maxV.x - minV.x, maxV.y - minV.y, maxV.z - minV.z)
        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.zNear = Double(size) * 0.01
        camera.zFar = Double(size) * 50
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(size * 0.9, size * 1.0, size * 1.6)
        cameraNode.look(at: SCNVector3(0, maxV.y * 0.35, 0))
        scene.rootNode.addChildNode(cameraNode)
        return (scene, cameraNode)
    }
}

/// 公分尺規：桌面上的方格（每 1 cm 細線、每 5 cm 粗線並標示數字），角落一支垂直尺
enum RulerBuilder {
    static func make(min mn: SCNVector3, max mx: SCNVector3) -> SCNNode {
        let root = SCNNode()
        root.name = "ruler"
        let cm: Float = 0.01
        let ext = Swift.max(Swift.max(abs(mn.x), abs(mx.x)), Swift.max(abs(mn.z), abs(mx.z)))
        let halfCM = Int((ext / cm / 5).rounded(.up)) * 5 + 5          // 方格半寬（公分，5 的倍數）
        let half = Float(halfCM) * cm
        let y: Float = 0.0012                                           // 稍微高於桌面，避免閃爍
        let labelSize = Swift.max(Float(halfCM) * cm * 0.07, 0.006)

        var minor: [(SCNVector3, SCNVector3)] = []
        var major: [(SCNVector3, SCNVector3)] = []
        for i in -halfCM...halfCM {
            let p = Float(i) * cm
            let a1 = SCNVector3(p, y, -half), b1 = SCNVector3(p, y, half)
            let a2 = SCNVector3(-half, y, p), b2 = SCNVector3(half, y, p)
            if i % 5 == 0 { major += [(a1, b1), (a2, b2)] } else { minor += [(a1, b1), (a2, b2)] }
        }
        root.addChildNode(lines(minor, color: UIColor(white: 0.55, alpha: 0.35)))
        root.addChildNode(lines(major, color: UIColor(white: 0.25, alpha: 0.8)))
        root.addChildNode(lines([(SCNVector3(-half, y, 0), SCNVector3(half, y, 0))], color: .systemRed))   // x 軸
        root.addChildNode(lines([(SCNVector3(0, y, -half), SCNVector3(0, y, half))], color: .systemBlue))  // 深度軸

        for i in stride(from: -halfCM, through: halfCM, by: 5) {
            let p = Float(i) * cm
            root.addChildNode(label("\(i)", at: SCNVector3(p, y, half + labelSize * 1.4), size: labelSize))
            root.addChildNode(label("\(i)", at: SCNVector3(half + labelSize * 1.4, y, p), size: labelSize))
        }
        root.addChildNode(label("cm", at: SCNVector3(half + labelSize * 1.6, y, half + labelSize * 1.6), size: labelSize))

        // 垂直尺（放在方格的左後角）
        let topCM = Swift.max(Int((mx.y / cm).rounded(.up)), 1)
        let step = topCM <= 6 ? 1 : (topCM <= 15 ? 2 : 5)
        let x0 = -half, z0 = -half
        var vlines: [(SCNVector3, SCNVector3)] = [(SCNVector3(x0, 0, z0), SCNVector3(x0, Float(topCM) * cm, z0))]
        for k in 0...topCM {
            let h = Float(k) * cm
            let tick: Float = (k % step == 0) ? labelSize * 0.9 : labelSize * 0.45
            vlines.append((SCNVector3(x0, h, z0), SCNVector3(x0 + tick, h, z0)))
            if k % step == 0 && k > 0 {
                root.addChildNode(label("\(k)", at: SCNVector3(x0 - labelSize * 1.2, h, z0), size: labelSize))
            }
        }
        root.addChildNode(lines(vlines, color: UIColor(white: 0.2, alpha: 0.9)))
        root.addChildNode(label("cm", at: SCNVector3(x0, Float(topCM) * cm + labelSize * 1.5, z0), size: labelSize))
        return root
    }

    static func lines(_ segs: [(SCNVector3, SCNVector3)], color: UIColor) -> SCNNode {
        var verts: [SCNVector3] = []
        var idx: [Int32] = []
        for (a, b) in segs {
            idx.append(Int32(verts.count)); verts.append(a)
            idx.append(Int32(verts.count)); verts.append(b)
        }
        let geo = SCNGeometry(sources: [SCNGeometrySource(vertices: verts)],
                              elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let m = SCNMaterial()
        m.diffuse.contents = color
        m.lightingModel = .constant
        geo.materials = [m]
        return SCNNode(geometry: geo)
    }

    static func label(_ text: String, at p: SCNVector3, size: Float) -> SCNNode {
        let t = SCNText(string: text, extrusionDepth: 0)
        t.font = UIFont.systemFont(ofSize: 10, weight: .semibold)
        t.flatness = 0.2
        let m = SCNMaterial()
        m.diffuse.contents = UIColor.darkGray
        m.lightingModel = .constant
        m.isDoubleSided = true
        t.materials = [m]
        let n = SCNNode(geometry: t)
        let (a, b) = n.boundingBox
        n.pivot = SCNMatrix4MakeTranslation((a.x + b.x) / 2, (a.y + b.y) / 2, 0)
        let s = size / 10
        n.scale = SCNVector3(s, s, s)
        n.position = p
        n.constraints = [SCNBillboardConstraint()]
        return n
    }
}
