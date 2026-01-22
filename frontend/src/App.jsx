import { useState, useCallback } from 'react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import { NotificationContainer } from './components/Notification';
import './App.css';

function App() {
  const [notifications, setNotifications] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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

  return (
    <div className="app">
      <main className="main-content">
        <ChatInterface />
      </main>
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        onNotification={addNotification}
      />
      <NotificationContainer
        notifications={notifications}
        onDismiss={removeNotification}
      />
    </div>
  );
}

export default App;
