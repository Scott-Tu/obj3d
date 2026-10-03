import Foundation
import Network

/// 不經過 URLSession、自己用 TLS 連線送出 HTTP/1.1 請求。
/// 用途：iOS 的 URLSession 可能會忽略或改掉 Authorization 標頭（Apple 列為保留標頭），
/// 這裡可以確保驗證標頭原封不動地送到 Kaggle。
enum RawHTTP {
    static func post(url: URL, headers: [String: String], body: Data, timeout: TimeInterval = 90) async throws -> (Int, Data) {
        guard let host = url.host else { throw KaggleError(message: "網址錯誤") }
        var path = url.path.isEmpty ? "/" : url.path
        if let q = url.query { path += "?" + q }
        var head = "POST \(path) HTTP/1.1\r\nHost: \(host)\r\nConnection: close\r\nContent-Length: \(body.count)\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        head += "\r\n"
        var request = Data(head.utf8)
        request.append(body)

        let connection = NWConnection(host: NWEndpoint.Host(host), port: 443, using: .tls)
        let queue = DispatchQueue(label: "obj3d.rawhttp")
        let raw: Data = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            var buffer = Data()
            var finished = false
            func finish(_ result: Result<Data, Error>) {
                if finished { return }
                finished = true
                connection.cancel()
                cont.resume(with: result)
            }
            func receive() {
                connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, isComplete, error in
                    if let d = data { buffer.append(d) }
                    if let e = error {
                        if buffer.isEmpty { finish(.failure(e)) } else { finish(.success(buffer)) }
                        return
                    }
                    if isComplete { finish(.success(buffer)) } else { receive() }
                }
            }
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    connection.send(content: request, completion: .contentProcessed { error in
                        if let e = error { finish(.failure(e)) } else { receive() }
                    })
                case .failed(let e):
                    finish(.failure(e))
                default:
                    break
                }
            }
            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + timeout) {
                finish(.failure(KaggleError(message: "連線逾時")))
            }
        }
        return try parse(raw)
    }

    private static func parse(_ raw: Data) throws -> (Int, Data) {
        guard let sep = raw.range(of: Data("\r\n\r\n".utf8)) else {
            throw KaggleError(message: "Kaggle 回應格式錯誤")
        }
        let headText = String(decoding: raw[raw.startIndex..<sep.lowerBound], as: UTF8.self)
        var body = Data(raw[sep.upperBound...])
        let lines = headText.components(separatedBy: "\r\n")
        let parts = (lines.first ?? "").split(separator: " ")
        let code = parts.count > 1 ? (Int(parts[1]) ?? 0) : 0
        let chunked = lines.contains { line in
            let l = line.lowercased()
            return l.hasPrefix("transfer-encoding:") && l.contains("chunked")
        }
        if chunked { body = dechunk(body) }
        return (code, body)
    }

    private static func dechunk(_ d: Data) -> Data {
        var out = Data()
        var i = d.startIndex
        let crlf = Data("\r\n".utf8)
        while i < d.endIndex {
            guard let r = d.range(of: crlf, in: i..<d.endIndex) else { break }
            let line = String(decoding: d[i..<r.lowerBound], as: UTF8.self)
            let sizeText = line.split(separator: ";").first.map(String.init) ?? "0"
            guard let size = Int(sizeText.trimmingCharacters(in: .whitespaces), radix: 16), size > 0 else { break }
            let start = r.upperBound
            let end = d.index(start, offsetBy: size, limitedBy: d.endIndex) ?? d.endIndex
            out.append(d[start..<end])
            i = d.index(end, offsetBy: 2, limitedBy: d.endIndex) ?? d.endIndex
        }
        return out
    }
}
