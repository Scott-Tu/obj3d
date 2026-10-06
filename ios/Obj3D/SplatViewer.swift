import SwiftUI
import WebKit

/// 3DGS 檢視：WKWebView 載入 App 內附的 viewer.html，透過自訂網址 app:// 讀取這次結果的 model.splat
struct SplatViewer: UIViewRepresentable {
    let resultDir: URL
    var cameraDistance: Double = 0.3     // 公尺
    var lookHeight: Double = 0.05        // 公尺

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(context.coordinator, forURLScheme: "app")
        let web = WKWebView(frame: .zero, configuration: config)
        web.isOpaque = false
        web.backgroundColor = .secondarySystemBackground
        web.scrollView.isScrollEnabled = false
        context.coordinator.resultDir = resultDir
        let q = String(format: "r=%.3f&cy=%.3f", cameraDistance, lookHeight)
        if let url = URL(string: "app://viewer/index.html?\(q)") {
            web.load(URLRequest(url: url))
        }
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKURLSchemeHandler {
        var resultDir: URL?

        func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
            guard let url = task.request.url else { return }
            var fileURL: URL?
            var mime = "application/octet-stream"
            if url.host == "viewer" {
                fileURL = Bundle.main.url(forResource: "viewer", withExtension: "html")
                mime = "text/html"
            } else if url.host == "result", let dir = resultDir {
                fileURL = dir.appendingPathComponent(url.lastPathComponent)
            }
            guard let f = fileURL, let data = try? Data(contentsOf: f),
                  let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                             headerFields: ["Content-Type": mime,
                                                            "Content-Length": "\(data.count)",
                                                            "Access-Control-Allow-Origin": "*"]) else {
                task.didFailWithError(URLError(.fileDoesNotExist))
                return
            }
            task.didReceive(resp)
            task.didReceive(data)
            task.didFinish()
        }

        func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
    }
}
