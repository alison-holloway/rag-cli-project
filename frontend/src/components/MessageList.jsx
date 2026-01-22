import { useEffect, useRef } from 'react';
import './MessageList.css';

/**
 * Display a list of chat messages
 */
function MessageList({ messages }) {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="message-list empty">
        <div className="empty-state">
          <h3>Ask a question</h3>
          <p>Type a question below to query your knowledge base.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.role}`}>
          <div className="message-content">
            {message.role === 'assistant' && message.isLoading ? (
              <div className="loading-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            ) : (
              <>
                <p className="message-text">{message.content}</p>
                {message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <details>
                      <summary>Sources ({message.sources.length})</summary>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>
                            <strong>{source.file}</strong>
                            <span className="similarity">
                              {(source.similarity * 100).toFixed(1)}% match
                            </span>
                            <p className="source-preview">{source.content}</p>
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
                {message.metadata && (
                  <div className="metadata">
                    <span>Model: {message.metadata.model}</span>
                    <span>{message.metadata.processing_time_ms.toFixed(0)}ms</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
