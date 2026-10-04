import SwiftUI
import ARKit

struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = false
        view.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct CaptureView: View {
    @StateObject private var capture = CaptureManager()
    @Environment(\.dismiss) private var dismiss
    @State private var finishing = false
    var mode: String
    var onFinish: (URL, String, Int) -> Void

    init(mode: String, onFinish: @escaping (URL, String, Int) -> Void) {
        self.mode = mode
        self.onFinish = onFinish
    }

    var body: some View {
        ZStack {
            ARViewContainer(session: capture.session)
                .ignoresSafeArea()

            Image(systemName: "plus")
                .font(.system(size: 48, weight: .light))
                .foregroundColor(.white)
                .shadow(color: .black.opacity(0.6), radius: 2)

            VStack(spacing: 12) {
                HStack {
                    Button("取消") {
                        capture.pauseSession()
                        dismiss()
                    }
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    Text("\(capture.frameCount) 張")
                        .monospacedDigit()
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                Text(capture.trackingText)
                    .font(.callout)
                    .foregroundColor(capture.trackingGood ? .primary : .orange)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                Spacer()
                Text(mode == "orbit"
                     ? "讓物體一直在十字中央，慢慢繞一整圈，\n再從斜上方繞一圈（約 40~90 秒）"
                     : "比例尺放在轉盤上、物體旁邊，跟著一起轉；\n手機盡量不動，慢慢轉一圈以上（約 40~90 秒）")
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                Button(action: toggle) {
                    ZStack {
                        Circle().stroke(Color.white, lineWidth: 4).frame(width: 78, height: 78)
                        if capture.isRecording {
                            RoundedRectangle(cornerRadius: 6).fill(Color.red).frame(width: 32, height: 32)
                        } else {
                            Circle().fill(Color.red).frame(width: 64, height: 64)
                        }
                    }
                }
                .disabled(finishing)
                .padding(.bottom, 20)
            }
            .padding()
        }
        .onAppear { capture.startSession() }
        .onDisappear { capture.pauseSession() }
    }

    private func toggle() {
        if capture.isRecording {
            finishing = true
            capture.endRecording { url, jobId, count in
                capture.pauseSession()
                if let url = url, let jobId = jobId {
                    onFinish(url, jobId, count)
                }
                dismiss()
            }
        } else {
            capture.beginRecording(mode: mode)
        }
    }
}
