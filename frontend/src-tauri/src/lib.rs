mod backend;
mod commands;
mod menu;
mod notifications;

use backend::Backend;
use log::{error, info};
use tauri::{Emitter, Manager};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .manage(Backend::new())
        .invoke_handler(tauri::generate_handler![
            commands::get_backend_info,
            commands::check_backend_health,
            commands::get_api_url,
            commands::restart_backend,
            commands::send_notification,
            commands::open_file_dialog,
        ])
        .setup(|app| {
            info!("Setting up RAG Assistant...");

            // Check webview window
            if let Some(window) = app.get_webview_window("main") {
                info!("Found main webview window");
                if let Ok(url) = window.url() {
                    info!("Current webview URL: {}", url);
                }
                // Only open devtools in debug builds
                #[cfg(debug_assertions)]
                {
                    window.open_devtools();
                    info!("DevTools opened (debug build)");
                }
            } else {
                error!("Could not find main webview window!");
                for (label, _) in app.webview_windows() {
                    info!("Found window with label: {}", label);
                }
            }

            // Build and set the application menu
            let handle = app.handle();
            match menu::build_menu(handle) {
                Ok(menu) => {
                    if let Err(e) = app.set_menu(menu) {
                        error!("Failed to set menu: {}", e);
                    }
                }
                Err(e) => {
                    error!("Failed to build menu: {}", e);
                }
            }

            // Get handle for async operations
            let handle = app.handle().clone();

            // Start backend in background
            let backend_state = app.state::<Backend>();
            match backend::spawn_backend(&backend_state) {
                Ok(port) => {
                    info!("Backend starting on port {}", port);

                    // Spawn async task to wait for backend to be ready
                    let port_copy = port;
                    let notification_handle = handle.clone();
                    tauri::async_runtime::spawn(async move {
                        match backend::wait_for_healthy(port_copy, 30).await {
                            Ok(_) => {
                                info!("Backend is ready!");
                                // Emit event to frontend
                                let _ = handle.emit("backend-ready", port_copy);
                                // Show notification
                                notifications::notify_backend_ready(&notification_handle);
                            }
                            Err(e) => {
                                error!("Backend failed to start: {}", e);
                                let _ = handle.emit("backend-error", e.clone());
                                // Show error notification
                                notifications::notify_backend_error(&notification_handle, &e);
                            }
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to spawn backend: {}", e);
                    notifications::notify_backend_error(&handle, &e);
                }
            }

            Ok(())
        })
        .on_menu_event(|app, event| {
            menu::handle_menu_event(app, event.id().as_ref());
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                info!("Window close requested, shutting down backend...");
                let backend = window.state::<Backend>();
                if let Err(e) = backend::shutdown_backend(&backend) {
                    error!("Error shutting down backend: {}", e);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
