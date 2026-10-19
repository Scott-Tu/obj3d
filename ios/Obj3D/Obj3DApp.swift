import SwiftUI

@main
struct Obj3DApp: App {
    @StateObject private var job = JobManager()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(job)
        }
        .onChange(of: scenePhase) { phase in
            if phase == .active { job.appBecameActive() }   // 回到前景：查詢卡住就重新開始
        }
    }
}
