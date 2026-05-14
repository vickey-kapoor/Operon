mod browser;
mod sidecar;
mod ws_bridge;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let handle = app.handle().clone();

            sidecar::launch(handle.clone())?;

            tauri::async_runtime::spawn(async move {
                ws_bridge::run(handle).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error running Operon");
}
