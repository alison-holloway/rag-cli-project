import { useState, useCallback } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { queryKnowledgeBase } from '../services/api';
import './ChatInterface.css';

/**
 * Main chat interface component
 */
function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

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
      const result = await queryKnowledgeBase(question);

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
  }, []);

  const handleDismissError = () => {
    setError(null);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>RAG CLI</h1>
        <p>Ask questions about your indexed documents</p>
      </div>

      {error && (
        <div className="error-banner">
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
