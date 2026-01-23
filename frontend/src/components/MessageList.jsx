import { useEffect, useRef, useState } from 'react';
import './MessageList.css';

/**
 * Copy button component with feedback
 */
function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <button
      className={`copy-btn ${copied ? 'copied' : ''}`}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

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
          <div className="empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <h3>Ask a question</h3>
          <p>Type a question below to query your knowledge base.</p>
          <div className="empty-hints">
            <span>Try asking:</span>
            <ul>
              <li>"What is this document about?"</li>
              <li>"Summarize the main points"</li>
              <li>"How do I..."</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.role}`}>
          <div className="message-avatar">
            {message.role === 'user' ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 16v-4" />
                <path d="M12 8h.01" />
              </svg>
            )}
          </div>
          <div className="message-content">
            {message.role === 'assistant' && message.isLoading ? (
              <div className="loading-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="loading-text">Thinking...</span>
              </div>
            ) : (
              <>
                <div className="message-header">
                  <span className="message-role">
                    {message.role === 'user' ? 'You' : 'Assistant'}
                  </span>
                  {message.role === 'assistant' && message.content && (
                    <CopyButton text={message.content} />
                  )}
                </div>
                <div className="message-text">{message.content}</div>
                {message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <details>
                      <summary>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                        {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
                      </summary>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>
                            <div className="source-header">
                              <strong>{source.file}</strong>
                              <span className="similarity">
                                {(source.similarity * 100).toFixed(0)}%
                              </span>
                            </div>
                            <p className="source-preview">{source.content}</p>
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
                {message.metadata && (
                  <div className="metadata">
                    <span className="meta-item">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                        <line x1="8" y1="21" x2="16" y2="21" />
                        <line x1="12" y1="17" x2="12" y2="21" />
                      </svg>
                      {message.metadata.model}
                    </span>
                    <span className="meta-item">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <polyline points="12 6 12 12 16 14" />
                      </svg>
                      {message.metadata.processing_time_ms.toFixed(0)}ms
                    </span>
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
