import { useState, useCallback, useEffect, useRef } from 'react';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import AboutDialog from './components/AboutDialog';
import WelcomeScreen from './components/WelcomeScreen';
import ErrorBoundary from './components/ErrorBoundary';
import { NotificationContainer } from './components/Notification';
import { setApiBaseUrl } from './services/api';
import './App.css';

const WELCOME_COMPLETED_KEY = 'rag-assistant-welcome-completed';

function AppContent() {
  const [notifications, setNotifications] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendReady, setBackendReady] = useState(!window.__TAURI__);
  const [showAbout, setShowAbout] = useState(false);
  const [showWelcome, setShowWelcome] = useState(() => {
    return localStorage.getItem(WELCOME_COMPLETED_KEY) !== 'true';
  });
  const [backendError, setBackendError] = useState(null);
  const { toggleTheme } = useTheme();
  const chatInterfaceRef = useRef(null);

  const handleRestartBackend = useCallback(async () => {
    if (!window.__TAURI__) return;

    try {
      setBackendError(null);
      setBackendReady(false);
      const { invoke } = await import('@tauri-apps/api/core');
      const port = await invoke('restart_backend');
      setApiBaseUrl(`http://127.0.0.1:${port}`);
      setBackendReady(true);
      addNotification({
        type: 'success',
        message: 'Backend restarted successfully',
      });
    } catch (error) {
      console.error('Failed to restart backend:', error);
      setBackendError(error.toString());
      addNotification({
        type: 'error',
        message: `Failed to restart backend: ${error}`,
        duration: 0,
      });
    }
  }, [addNotification]);

  const addNotification = useCallback((notification) => {
    const id = Date.now().toString();
    setNotifications((prev) => [...prev, { ...notification, id }]);
  }, []);

  const removeNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const handleNewChat = useCallback(() => {
    // Emit a custom event that ChatInterface can listen to
    window.dispatchEvent(new CustomEvent('app-new-chat'));
    addNotification({
      type: 'info',
      message: 'Started a new chat',
    });
  }, [addNotification]);

  const handleExportChat = useCallback(() => {
    // Emit a custom event that ChatInterface can listen to
    window.dispatchEvent(new CustomEvent('app-export-chat'));
  }, []);

  const handleFocusSearch = useCallback(() => {
    // Focus the chat input
    window.dispatchEvent(new CustomEvent('app-focus-search'));
  }, []);

  const handleWelcomeComplete = useCallback(() => {
    localStorage.setItem(WELCOME_COMPLETED_KEY, 'true');
    setShowWelcome(false);
  }, []);

  // Listen for Tauri backend and menu events
  useEffect(() => {
    if (!window.__TAURI__) return;

    const unlisteners = [];

    const setupTauriListeners = async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');

        // Backend events
        unlisteners.push(await listen('backend-ready', (event) => {
          console.log('Backend ready on port:', event.payload);
          setApiBaseUrl(`http://127.0.0.1:${event.payload}`);
          setBackendReady(true);
          addNotification({
            type: 'success',
            message: 'Backend service started successfully',
          });
        }));

        unlisteners.push(await listen('backend-error', (event) => {
          console.error('Backend error:', event.payload);
          setBackendError(event.payload);
          addNotification({
            type: 'error',
            message: `Backend error: ${event.payload}`,
            duration: 0,
          });
        }));

        // Menu events
        unlisteners.push(await listen('menu-new-chat', () => {
          handleNewChat();
        }));

        unlisteners.push(await listen('menu-export-chat', () => {
          handleExportChat();
        }));

        unlisteners.push(await listen('menu-toggle-sidebar', () => {
          toggleSidebar();
        }));

        unlisteners.push(await listen('menu-toggle-dark-mode', () => {
          toggleTheme();
        }));

        unlisteners.push(await listen('menu-focus-search', () => {
          handleFocusSearch();
        }));

        unlisteners.push(await listen('menu-about', () => {
          setShowAbout(true);
        }));

      } catch (error) {
        console.error('Failed to setup Tauri listeners:', error);
        setBackendReady(true);
      }
    };

    setupTauriListeners();

    return () => {
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, [addNotification, handleNewChat, handleExportChat, toggleSidebar, toggleTheme, handleFocusSearch]);

  // Keyboard shortcuts (for browser mode and as backup)
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Cmd/Ctrl + N: New chat
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault();
        handleNewChat();
      }
      // Cmd/Ctrl + K: Focus search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        handleFocusSearch();
      }
      // Cmd/Ctrl + Shift + E: Export chat
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'e') {
        e.preventDefault();
        handleExportChat();
      }
      // Cmd/Ctrl + Shift + D: Toggle dark mode
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'd') {
        e.preventDefault();
        toggleTheme();
      }
      // Cmd/Ctrl + \: Toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault();
        toggleSidebar();
      }
      // Escape: Close about dialog
      if (e.key === 'Escape' && showAbout) {
        setShowAbout(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNewChat, handleExportChat, handleFocusSearch, toggleTheme, toggleSidebar, showAbout]);

  return (
    <ErrorBoundary>
      <div className="app">
        {showWelcome && backendReady && !backendError && (
          <WelcomeScreen onComplete={handleWelcomeComplete} />
        )}
        {backendError ? (
          <div className="backend-error">
            <div className="backend-error-icon">
              <svg viewBox="0 0 100 100" width="60" height="60" fill="none" stroke="currentColor" strokeWidth="4">
                <circle cx="50" cy="50" r="40" />
                <path d="M50 30 L50 55" strokeLinecap="round" />
                <circle cx="50" cy="68" r="3" fill="currentColor" />
              </svg>
            </div>
            <h2>Backend Connection Error</h2>
            <p>Unable to connect to the RAG Assistant backend.</p>
            <p className="backend-error-detail">{backendError}</p>
            <div className="backend-error-actions">
              <button className="backend-error-button primary" onClick={handleRestartBackend}>
                Restart Backend
              </button>
              <button className="backend-error-button secondary" onClick={() => setBackendError(null)}>
                Dismiss
              </button>
            </div>
          </div>
        ) : !backendReady ? (
          <div className="backend-loading">
            <div className="loading-spinner" />
            <p>Starting RAG Assistant...</p>
          </div>
        ) : (
          <>
            <main className="main-content">
              <ChatInterface ref={chatInterfaceRef} />
            </main>
            <Sidebar
              isOpen={sidebarOpen}
              onToggle={toggleSidebar}
              onNotification={addNotification}
            />
          </>
        )}
        <NotificationContainer
          notifications={notifications}
          onDismiss={removeNotification}
        />
        {showAbout && (
          <AboutDialog onClose={() => setShowAbout(false)} />
        )}
      </div>
    </ErrorBoundary>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
