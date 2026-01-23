import { useState, useCallback } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import SettingsPanel from './SettingsPanel';
import ThemeToggle from './ThemeToggle';
import { queryKnowledgeBase } from '../services/api';
import './ChatInterface.css';

/**
 * Format date for export
 */
function formatDate(date) {
  if (!date) return '';
  return new Date(date).toLocaleString();
}

/**
 * Export chat as JSON file
 */
function exportAsJSON(messages) {
  const exportData = {
    exportedAt: new Date().toISOString(),
    messageCount: messages.length,
    messages: messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
      ...(msg.sources && { sources: msg.sources }),
      ...(msg.metadata && { metadata: msg.metadata }),
    })),
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rag-chat-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export chat as text file
 */
function exportAsText(messages) {
  let text = `RAG CLI Chat Export\n`;
  text += `Exported: ${formatDate(new Date())}\n`;
  text += `${'='.repeat(50)}\n\n`;

  messages.forEach((msg) => {
    const role = msg.role === 'user' ? 'You' : 'Assistant';
    const time = msg.timestamp ? ` (${formatDate(msg.timestamp)})` : '';
    text += `${role}${time}:\n`;
    text += `${msg.content}\n`;

    if (msg.sources && msg.sources.length > 0) {
      text += `\nSources:\n`;
      msg.sources.forEach((source, i) => {
        text += `  ${i + 1}. ${source.file} (${(source.similarity * 100).toFixed(0)}% match)\n`;
      });
    }

    if (msg.metadata) {
      text += `\n[Model: ${msg.metadata.model}, Time: ${msg.metadata.processing_time_ms.toFixed(0)}ms]\n`;
    }

    text += `\n${'-'.repeat(50)}\n\n`;
  });

  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rag-chat-${new Date().toISOString().slice(0, 10)}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Main chat interface component
 */
function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [settings, setSettings] = useState({
    llmProvider: 'ollama',
    topK: 5,
    temperature: 0.7,
  });

  // Generate unique message ID
  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  const handleSend = useCallback(async (question) => {
    // Clear any previous error
    setError(null);

    // Add user message
    const userMessage = {
      id: generateId(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add loading placeholder for assistant
    const assistantId = generateId();
    const loadingMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      isLoading: true,
    };
    setMessages((prev) => [...prev, loadingMessage]);
    setIsLoading(true);

    try {
      const result = await queryKnowledgeBase(question, {
        llmProvider: settings.llmProvider,
        topK: settings.topK,
        temperature: settings.temperature,
      });

      // Update the loading message with the actual response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: result.answer,
                sources: result.sources,
                metadata: result.metadata,
                isLoading: false,
                timestamp: new Date(),
              }
            : msg
        )
      );
    } catch (err) {
      // Remove loading message and show error
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantId));
      setError(err.message || 'Failed to get response');
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  const handleDismissError = () => {
    setError(null);
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
  };

  const handleSettingsChange = (newSettings) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  };

  const handleExport = (format) => {
    if (format === 'json') {
      exportAsJSON(messages);
    } else {
      exportAsText(messages);
    }
    setShowExportMenu(false);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-content">
          <h1>RAG CLI</h1>
          <p>Ask questions about your indexed documents</p>
        </div>
        <div className="header-actions">
          {messages.length > 0 && (
            <>
              <div className="export-dropdown">
                <button
                  className={`header-btn export-btn ${showExportMenu ? 'active' : ''}`}
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  title="Export chat"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Export</span>
                </button>
                {showExportMenu && (
                  <div className="export-menu">
                    <button onClick={() => handleExport('text')}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                      </svg>
                      Export as Text
                    </button>
                    <button onClick={() => handleExport('json')}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="16 18 22 12 16 6" />
                        <polyline points="8 6 2 12 8 18" />
                      </svg>
                      Export as JSON
                    </button>
                  </div>
                )}
              </div>
              <button
                className="header-btn clear-btn"
                onClick={handleClearChat}
                title="Clear chat history"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                <span>Clear</span>
              </button>
            </>
          )}
          <button
            className={`header-btn settings-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(!showSettings)}
            title="Settings"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span>Settings</span>
          </button>
          <ThemeToggle />
        </div>
      </div>

      {showSettings && (
        <SettingsPanel
          settings={settings}
          onChange={handleSettingsChange}
          onClose={() => setShowSettings(false)}
        />
      )}

      {error && (
        <div className="error-banner">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{error}</span>
          <button onClick={handleDismissError} className="dismiss-btn">
            &times;
          </button>
        </div>
      )}

      <MessageList messages={messages} />

      <MessageInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}

export default ChatInterface;
