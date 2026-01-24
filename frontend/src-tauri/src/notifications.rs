use tauri::{AppHandle, Wry};
use tauri_plugin_notification::NotificationExt;

/// Send a notification when a query is complete
pub fn notify_query_complete(app: &AppHandle<Wry>, query: &str) {
    let truncated_query = if query.len() > 50 {
        format!("{}...", &query[..50])
    } else {
        query.to_string()
    };

    if let Err(e) = app
        .notification()
        .builder()
        .title("Query Complete")
        .body(format!("Answer ready for: {}", truncated_query))
        .show()
    {
        log::error!("Failed to show notification: {}", e);
    }
}

/// Send a notification when a document is indexed
pub fn notify_document_indexed(app: &AppHandle<Wry>, filename: &str) {
    if let Err(e) = app
        .notification()
        .builder()
        .title("Document Indexed")
        .body(format!("{} has been added to your knowledge base", filename))
        .show()
    {
        log::error!("Failed to show notification: {}", e);
    }
}

/// Send a notification when a document is deleted
pub fn notify_document_deleted(app: &AppHandle<Wry>, filename: &str) {
    if let Err(e) = app
        .notification()
        .builder()
        .title("Document Removed")
        .body(format!("{} has been removed from your knowledge base", filename))
        .show()
    {
        log::error!("Failed to show notification: {}", e);
    }
}

/// Send an error notification
pub fn notify_error(app: &AppHandle<Wry>, title: &str, message: &str) {
    if let Err(e) = app
        .notification()
        .builder()
        .title(title)
        .body(message)
        .show()
    {
        log::error!("Failed to show error notification: {}", e);
    }
}

/// Send a notification when the backend starts successfully
pub fn notify_backend_ready(app: &AppHandle<Wry>) {
    if let Err(e) = app
        .notification()
        .builder()
        .title("RAG Assistant Ready")
        .body("Backend service started successfully")
        .show()
    {
        log::error!("Failed to show notification: {}", e);
    }
}

/// Send a notification when the backend fails to start
pub fn notify_backend_error(app: &AppHandle<Wry>, error: &str) {
    if let Err(e) = app
        .notification()
        .builder()
        .title("Backend Error")
        .body(format!("Failed to start backend: {}", error))
        .show()
    {
        log::error!("Failed to show error notification: {}", e);
    }
}
