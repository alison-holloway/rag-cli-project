use crate::backend::{self, Backend};
use crate::notifications;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

#[derive(Debug, Serialize, Deserialize)]
pub struct BackendInfo {
    pub port: u16,
    pub running: bool,
    pub data_dir: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NotificationRequest {
    pub notification_type: String,
    pub title: Option<String>,
    pub message: String,
}

/// Get information about the backend
#[tauri::command]
pub fn get_backend_info(backend: State<'_, Backend>) -> BackendInfo {
    let port = backend
        .0
        .lock()
        .map(|b| b.port())
        .unwrap_or(8000);

    BackendInfo {
        port,
        running: backend::is_backend_running(&backend),
        data_dir: backend::get_data_dir().to_string_lossy().to_string(),
    }
}

/// Check if the backend is healthy
#[tauri::command]
pub async fn check_backend_health(backend: State<'_, Backend>) -> Result<bool, String> {
    let port = backend
        .0
        .lock()
        .map(|b| b.port())
        .unwrap_or(8000);

    Ok(backend::check_health(port).await)
}

/// Get the API base URL for the frontend
#[tauri::command]
pub fn get_api_url(backend: State<'_, Backend>) -> String {
    let port = backend
        .0
        .lock()
        .map(|b| b.port())
        .unwrap_or(8000);

    format!("http://127.0.0.1:{}", port)
}

/// Restart the backend process
#[tauri::command]
pub async fn restart_backend(backend: State<'_, Backend>) -> Result<u16, String> {
    // Shutdown existing backend
    backend::shutdown_backend(&backend)?;

    // Start new backend
    let port = backend::spawn_backend(&backend)?;

    // Wait for it to be healthy
    backend::wait_for_healthy(port, 30).await?;

    Ok(port)
}

/// Send a system notification from the frontend
#[tauri::command]
pub fn send_notification(app: AppHandle, request: NotificationRequest) {
    match request.notification_type.as_str() {
        "query_complete" => {
            notifications::notify_query_complete(&app, &request.message);
        }
        "document_indexed" => {
            notifications::notify_document_indexed(&app, &request.message);
        }
        "document_deleted" => {
            notifications::notify_document_deleted(&app, &request.message);
        }
        "error" => {
            let title = request.title.unwrap_or_else(|| "Error".to_string());
            notifications::notify_error(&app, &title, &request.message);
        }
        _ => {
            log::warn!("Unknown notification type: {}", request.notification_type);
        }
    }
}

/// Open a file dialog for document selection
#[tauri::command]
pub async fn open_file_dialog() -> Result<Vec<String>, String> {
    // For now, just return an empty list - file dialog would need additional plugin
    // The actual file upload will be handled by the drag-and-drop or file input in the frontend
    Ok(vec![])
}
