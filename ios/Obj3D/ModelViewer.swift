import SwiftUI
import SceneKit

/// 顯示 Kaggle 產生的 preview.bin（可用手指旋轉、縮放）
struct ModelViewer: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = UIColor.secondarySystemBackground
        view.antialiasingMode = .multisampling4X
        let (scene, camera) = PreviewLoader.load(url)
        view.scene = scene
        if let camera = camera { view.pointOfView = camera }
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}

enum PreviewLoader {
    /// 格式：'O3DP' + u32 頂點數 + u32 三角面數 + f32 位置 + f32 法向量 + f32 顏色 + u32 索引
    static func load(_ url: URL) -> (SCNScene, SCNNode?) {
        let scene = SCNScene()
        guard let data = try? Data(contentsOf: url), data.count > 12,
              String(data: data.prefix(4), encoding: .ascii) == "O3DP" else { return (scene, nil) }
        let nV = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) })
        let nF = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self) })
        let vBytes = nV * 12
        let posOff = 12
        let nrmOff = posOff + vBytes
        let colOff = nrmOff + vBytes
        let idxOff = colOff + vBytes
        let end = idxOff + nF * 12
        guard nV > 0, nF > 0, data.count >= end else { return (scene, nil) }

        func source(_ off: Int, _ semantic: SCNGeometrySource.Semantic) -> SCNGeometrySource {
            SCNGeometrySource(data: data.subdata(in: off..<(off + vBytes)), semantic: semantic,
                              vectorCount: nV, usesFloatComponents: true, componentsPerVector: 3,
                              bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        }
        let element = SCNGeometryElement(data: data.subdata(in: idxOff..<end), primitiveType: .triangles,
                                         primitiveCount: nF, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [source(posOff, .vertex), source(nrmOff, .normal), source(colOff, .color)],
                                   elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.white
        material.isDoubleSided = true
        material.lightingModel = .blinn
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        let (minV, maxV) = node.boundingBox
        node.position = SCNVector3(-(minV.x + maxV.x) / 2, -(minV.y + maxV.y) / 2, -(minV.z + maxV.z) / 2)
        scene.rootNode.addChildNode(node)

        let size = max(maxV.x - minV.x, maxV.y - minV.y, maxV.z - minV.z)
        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.zNear = Double(size) * 0.01
        camera.zFar = Double(size) * 50
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, size * 0.5, size * 1.8)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)
        return (scene, cameraNode)
    }
}
