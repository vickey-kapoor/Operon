// Browser launch helpers for the Tauri desktop wrapper.
// The public CDP attach API (POST /connect-cdp) was removed; browser
// sessions are now managed internally by the Python backend.

use std::process::Command;

/// Launch Chrome with a remote debugging port (internal use only).
pub fn launch_chrome_debug(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    _launch_chrome_impl(port)
}

async fn _is_port_open(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

#[cfg(target_os = "windows")]
fn _launch_chrome_impl(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ];
    let exe = candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .ok_or("Google Chrome not found in standard locations")?;

    Command::new(exe)
        .args([
            &format!("--remote-debugging-port={port}"),
            "--no-first-run",
            "--no-default-browser-check",
        ])
        .spawn()?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn _launch_chrome_impl(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    Command::new("open")
        .args([
            "-a",
            "Google Chrome",
            "--args",
            &format!("--remote-debugging-port={port}"),
            "--no-first-run",
        ])
        .spawn()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn _launch_chrome_impl(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    Command::new("google-chrome")
        .args([
            &format!("--remote-debugging-port={port}"),
            "--no-first-run",
        ])
        .spawn()?;
    Ok(())
}
