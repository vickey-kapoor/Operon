use tauri::AppHandle;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

/// In dev mode the Python server is started manually (`uvicorn operon.api.server:app --port 8080`).
/// In release builds Tauri bundles `operon-api` as an external binary and launches it here.
pub fn launch(app: AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(debug_assertions)]
    {
        log::info!("sidecar: dev mode — start Python backend manually on port 8080");
        let _ = app; // suppress unused warning
        return Ok(());
    }

    #[cfg(not(debug_assertions))]
    {
        let (mut rx, _child) = app
            .shell()
            .sidecar("operon-api")
            .expect("operon-api sidecar not declared in tauri.conf.json")
            .spawn()
            .expect("failed to spawn operon-api sidecar");

        tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(line) => {
                        log::info!("[sidecar] {}", String::from_utf8_lossy(&line));
                    }
                    CommandEvent::Stderr(line) => {
                        log::warn!("[sidecar] {}", String::from_utf8_lossy(&line));
                    }
                    CommandEvent::Error(e) => {
                        log::error!("[sidecar] error: {e}");
                    }
                    CommandEvent::Terminated(status) => {
                        log::error!("[sidecar] terminated with status {:?}", status);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }
}
