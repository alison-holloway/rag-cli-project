use log::{error, info, warn};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tokio::time::sleep;

/// Global state for the backend process
pub struct BackendState {
    process: Option<Child>,
    port: u16,
    project_dir: PathBuf,
}

impl BackendState {
    pub fn new() -> Self {
        Self {
            process: None,
            port: 8000,
            project_dir: PathBuf::new(),
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Wrapper for thread-safe access to backend state
pub struct Backend(pub Mutex<BackendState>);

impl Backend {
    pub fn new() -> Self {
        Self(Mutex::new(BackendState::new()))
    }
}

/// Find the project root directory (where backend/ folder is)
fn find_project_dir() -> Option<PathBuf> {
    // Try to find the project directory relative to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        // In development: exe is in target/debug or target/release
        // In production (app bundle): exe is in RAG Assistant.app/Contents/MacOS/

        let mut current = exe_path.parent().map(|p| p.to_path_buf());

        // Walk up the directory tree looking for the backend folder
        while let Some(dir) = current {
            let backend_dir = dir.join("backend");
            if backend_dir.exists() && backend_dir.is_dir() {
                return Some(dir);
            }

            // Also check for the typical development structure
            // frontend/src-tauri/target/release -> ../../.. = frontend -> .. = project root
            let potential_project = dir.join("..").join("..").join("..").join("backend");
            if potential_project.exists() {
                if let Ok(canonical) = dir.join("..").join("..").join("..").canonicalize() {
                    return Some(canonical);
                }
            }

            current = dir.parent().map(|p| p.to_path_buf());
        }
    }

    // Try current working directory
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("backend").exists() {
            return Some(cwd);
        }
        // Check parent (in case running from frontend/)
        if let Some(parent) = cwd.parent() {
            if parent.join("backend").exists() {
                return Some(parent.to_path_buf());
            }
        }
    }

    None
}

/// Find an available port starting from the given port
fn find_available_port(start_port: u16) -> u16 {
    portpicker::pick_unused_port().unwrap_or(start_port)
}

/// Get the data directory for the application
pub fn get_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("RAG Assistant")
}

/// Get the log directory for the application
pub fn get_log_dir() -> PathBuf {
    get_data_dir().join("logs")
}

/// Spawn the backend process
pub fn spawn_backend(state: &Backend) -> Result<u16, String> {
    let mut backend = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    if backend.process.is_some() {
        return Ok(backend.port);
    }

    // Find project directory
    let project_dir = find_project_dir()
        .ok_or_else(|| "Could not find project directory with backend folder".to_string())?;

    info!("Project directory: {:?}", project_dir);
    backend.project_dir = project_dir.clone();

    // Find an available port
    let port = find_available_port(8000);
    backend.port = port;
    info!("Using port: {}", port);

    // Create data and log directories
    let data_dir = get_data_dir();
    let log_dir = get_log_dir();
    std::fs::create_dir_all(&data_dir).map_err(|e| format!("Failed to create data dir: {}", e))?;
    std::fs::create_dir_all(&log_dir).map_err(|e| format!("Failed to create log dir: {}", e))?;

    // Find Python executable (try venv first, then system)
    let venv_python = project_dir.join("venv").join("bin").join("python");
    let python_path = if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python3")
    };

    info!("Python path: {:?}", python_path);

    // Verify Python executable exists and is correct version
    if !python_path.exists() {
        return Err(format!("Python executable not found at {:?}", python_path));
    }

    // Check Python version
    let version_output = Command::new(&python_path)
        .args(["--version"])
        .output()
        .map_err(|e| format!("Failed to get Python version: {}", e))?;

    let version_str = String::from_utf8_lossy(&version_output.stdout);
    info!("Python version: {}", version_str.trim());

    if version_output.stderr.len() > 0 {
        let stderr_str = String::from_utf8_lossy(&version_output.stderr);
        info!("Python version stderr: {}", stderr_str.trim());
    }

    // Test that we can import the backend module
    info!("Testing backend module import...");
    let test_import = Command::new(&python_path)
        .args(["-c", "import backend.main; print('Backend module OK')"])
        .current_dir(&project_dir)
        .output()
        .map_err(|e| format!("Failed to test backend import: {}", e))?;

    if !test_import.status.success() {
        let stderr = String::from_utf8_lossy(&test_import.stderr);
        error!("Backend module import failed: {}", stderr);
        return Err(format!("Backend module import failed: {}", stderr));
    }
    info!("Backend module import successful");

    // Start uvicorn
    let mut child = Command::new(&python_path)
        .args([
            "-m", "uvicorn",
            "backend.main:app",
            "--port", &port.to_string(),
            "--host", "127.0.0.1",
        ])
        .current_dir(&project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;

    info!("Backend process started with PID: {:?}", child.id());

    // Spawn threads to capture and log stdout/stderr
    if let Some(stdout) = child.stdout.take() {
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    info!("[Backend stdout] {}", line);
                }
            }
        });
    }

    if let Some(stderr) = child.stderr.take() {
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    // Uvicorn logs to stderr, so use warn for actual errors
                    if line.contains("ERROR") || line.contains("Traceback") || line.contains("Exception") {
                        error!("[Backend stderr] {}", line);
                    } else {
                        info!("[Backend stderr] {}", line);
                    }
                }
            }
        });
    }

    backend.process = Some(child);

    Ok(port)
}

/// Check if the backend is healthy
pub async fn check_health(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/api/health", port);

    match reqwest::get(&url).await {
        Ok(response) => {
            let success = response.status().is_success();
            if success {
                info!("Health check succeeded: {}", url);
            } else {
                warn!("Health check failed with status: {}", response.status());
            }
            success
        },
        Err(e) => {
            warn!("Health check error: {}", e);
            false
        }
    }
}

/// Wait for the backend to become healthy
pub async fn wait_for_healthy(port: u16, timeout_secs: u64) -> Result<(), String> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    while start.elapsed() < timeout {
        if check_health(port).await {
            info!("Backend is healthy on port {}", port);
            return Ok(());
        }
        sleep(Duration::from_millis(500)).await;
    }

    Err(format!("Backend did not become healthy within {} seconds", timeout_secs))
}

/// Shutdown the backend process gracefully
pub fn shutdown_backend(state: &Backend) -> Result<(), String> {
    let mut backend = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    if let Some(mut child) = backend.process.take() {
        info!("Shutting down backend process...");

        // Try graceful shutdown first (SIGTERM on Unix)
        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as i32, libc::SIGTERM);
        }

        // Wait a bit for graceful shutdown
        std::thread::sleep(Duration::from_secs(2));

        // Force kill if still running
        match child.try_wait() {
            Ok(Some(_)) => {
                info!("Backend process exited gracefully");
            }
            Ok(None) => {
                warn!("Backend process did not exit, forcing kill");
                let _ = child.kill();
            }
            Err(e) => {
                error!("Error checking backend process: {}", e);
                let _ = child.kill();
            }
        }
    }

    Ok(())
}

/// Check if the backend process is still running
pub fn is_backend_running(state: &Backend) -> bool {
    if let Ok(mut backend) = state.0.lock() {
        if let Some(ref mut child) = backend.process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    // Process has exited
                    backend.process = None;
                    false
                }
                Ok(None) => true, // Still running
                Err(_) => false,
            }
        } else {
            false
        }
    } else {
        false
    }
}
