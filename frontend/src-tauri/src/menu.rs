use tauri::{
    menu::{Menu, MenuBuilder, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder},
    AppHandle, Emitter, Wry,
};

/// Build the application menu
pub fn build_menu(app: &AppHandle<Wry>) -> Result<Menu<Wry>, tauri::Error> {
    // File menu
    let new_chat = MenuItemBuilder::with_id("new_chat", "New Chat")
        .accelerator("CmdOrCtrl+N")
        .build(app)?;

    let export_chat = MenuItemBuilder::with_id("export_chat", "Export Chat...")
        .accelerator("CmdOrCtrl+Shift+E")
        .build(app)?;

    let file_menu = SubmenuBuilder::new(app, "File")
        .item(&new_chat)
        .separator()
        .item(&export_chat)
        .separator()
        .item(&PredefinedMenuItem::close_window(app, Some("Close Window"))?)
        .build()?;

    // Edit menu (standard macOS edit menu)
    let edit_menu = SubmenuBuilder::new(app, "Edit")
        .item(&PredefinedMenuItem::undo(app, Some("Undo"))?)
        .item(&PredefinedMenuItem::redo(app, Some("Redo"))?)
        .separator()
        .item(&PredefinedMenuItem::cut(app, Some("Cut"))?)
        .item(&PredefinedMenuItem::copy(app, Some("Copy"))?)
        .item(&PredefinedMenuItem::paste(app, Some("Paste"))?)
        .item(&PredefinedMenuItem::select_all(app, Some("Select All"))?)
        .build()?;

    // View menu
    let toggle_sidebar = MenuItemBuilder::with_id("toggle_sidebar", "Toggle Sidebar")
        .accelerator("CmdOrCtrl+\\")
        .build(app)?;

    let toggle_dark_mode = MenuItemBuilder::with_id("toggle_dark_mode", "Toggle Dark Mode")
        .accelerator("CmdOrCtrl+Shift+D")
        .build(app)?;

    let focus_search = MenuItemBuilder::with_id("focus_search", "Focus Search")
        .accelerator("CmdOrCtrl+K")
        .build(app)?;

    let view_menu = SubmenuBuilder::new(app, "View")
        .item(&toggle_sidebar)
        .item(&toggle_dark_mode)
        .separator()
        .item(&focus_search)
        .separator()
        .item(&PredefinedMenuItem::fullscreen(app, Some("Enter Full Screen"))?)
        .build()?;

    // Window menu (standard macOS window menu)
    let window_menu = SubmenuBuilder::new(app, "Window")
        .item(&PredefinedMenuItem::minimize(app, Some("Minimize"))?)
        .item(&PredefinedMenuItem::maximize(app, Some("Zoom"))?)
        .separator()
        .item(&PredefinedMenuItem::close_window(app, Some("Close"))?)
        .build()?;

    // Help menu
    let documentation = MenuItemBuilder::with_id("documentation", "Documentation")
        .build(app)?;

    let about = MenuItemBuilder::with_id("about", "About RAG Assistant")
        .build(app)?;

    let help_menu = SubmenuBuilder::new(app, "Help")
        .item(&documentation)
        .separator()
        .item(&about)
        .build()?;

    // Build the complete menu
    let menu = MenuBuilder::new(app)
        .item(&file_menu)
        .item(&edit_menu)
        .item(&view_menu)
        .item(&window_menu)
        .item(&help_menu)
        .build()?;

    Ok(menu)
}

/// Handle menu events
pub fn handle_menu_event(app: &AppHandle<Wry>, event_id: &str) {
    match event_id {
        "new_chat" => {
            log::info!("Menu: New Chat");
            let _ = app.emit("menu-new-chat", ());
        }
        "export_chat" => {
            log::info!("Menu: Export Chat");
            let _ = app.emit("menu-export-chat", ());
        }
        "toggle_sidebar" => {
            log::info!("Menu: Toggle Sidebar");
            let _ = app.emit("menu-toggle-sidebar", ());
        }
        "toggle_dark_mode" => {
            log::info!("Menu: Toggle Dark Mode");
            let _ = app.emit("menu-toggle-dark-mode", ());
        }
        "focus_search" => {
            log::info!("Menu: Focus Search");
            let _ = app.emit("menu-focus-search", ());
        }
        "documentation" => {
            log::info!("Menu: Documentation");
            // Open documentation URL
            let _ = open::that("https://github.com/student/rag-assistant");
        }
        "about" => {
            log::info!("Menu: About");
            let _ = app.emit("menu-about", ());
        }
        _ => {}
    }
}
