import { useState, useCallback } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import SettingsPanel from './SettingsPanel';
import { queryKnowledgeBase } from '../services/api';
import './ChatInterface.css';

/**
 * Main chat interface component
 */
function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
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

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-content">
          <h1>RAG CLI</h1>
          <p>Ask questions about your indexed documents</p>
        </div>
        <div className="header-actions">
          {messages.length > 0 && (
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
