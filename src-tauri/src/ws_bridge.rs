/// Bi-directional WebSocket bridge.
///
/// Listens on BRIDGE_PORT for React frontend connections and proxies each
/// connection to the Python backend's `/ws/stream` endpoint.
///
/// - Text frames  → JSON step events, rule traces, confidence, subgoal updates
/// - Binary frames → raw JPEG screenshot data (zero-copy passthrough)
/// - Frontend → Python → control messages: pause / resume / override hint
use futures_util::{SinkExt, StreamExt};
use tauri::AppHandle;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message};

pub const BRIDGE_PORT: u16 = 9001;
const PYTHON_WS_URL: &str = "ws://127.0.0.1:8080/ws/stream";

/// Retry connecting to the Python backend with exponential backoff.
/// Operon's sidecar takes a moment to warm up; this prevents a hard fail.
async fn connect_to_python() -> tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
> {
    let mut delay = std::time::Duration::from_millis(250);
    loop {
        match connect_async(PYTHON_WS_URL).await {
            Ok((ws, _)) => return ws,
            Err(e) => {
                log::warn!("WS bridge: cannot reach Python backend ({e}), retrying in {delay:?}");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(4));
            }
        }
    }
}

pub async fn run(_app: AppHandle) {
    let listener = TcpListener::bind(format!("127.0.0.1:{BRIDGE_PORT}"))
        .await
        .unwrap_or_else(|e| panic!("WS bridge failed to bind port {BRIDGE_PORT}: {e}"));

    log::info!("WS bridge listening on ws://127.0.0.1:{BRIDGE_PORT}");

    while let Ok((stream, addr)) = listener.accept().await {
        log::info!("WS bridge: frontend connected from {addr}");
        tokio::spawn(handle_connection(stream));
    }
}

async fn handle_connection(raw: tokio::net::TcpStream) {
    let frontend_ws = match accept_async(raw).await {
        Ok(ws) => ws,
        Err(e) => {
            log::error!("WS bridge: frontend handshake failed: {e}");
            return;
        }
    };

    let python_ws = connect_to_python().await;

    let (mut fe_tx, mut fe_rx) = frontend_ws.split();
    let (mut py_tx, mut py_rx) = python_ws.split();

    // Python → Frontend: step events (text) and screenshot frames (binary)
    let py_to_fe = async {
        while let Some(msg) = py_rx.next().await {
            match msg {
                Ok(msg) => {
                    if fe_tx.send(msg).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    log::warn!("WS bridge: Python side error: {e}");
                    break;
                }
            }
        }
    };

    // Frontend → Python: control messages (pause, resume, override hint)
    let fe_to_py = async {
        while let Some(msg) = fe_rx.next().await {
            match msg {
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(msg) => {
                    if py_tx.send(msg).await.is_err() {
                        break;
                    }
                }
            }
        }
    };

    tokio::select! {
        _ = py_to_fe => {}
        _ = fe_to_py => {}
    }

    log::info!("WS bridge: connection closed");
}
