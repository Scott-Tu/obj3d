import SwiftUI

@main
struct Obj3DApp: App {
    @StateObject private var job = JobManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(job)
        }
    }
}
