import { useEffect, useRef, useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import sql from 'highlight.js/lib/languages/sql';
import yaml from 'highlight.js/lib/languages/yaml';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import typescript from 'highlight.js/lib/languages/typescript';
import 'highlight.js/styles/github.css';
import './MessageList.css';

// Register languages
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('js', javascript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('py', python);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('sh', bash);
hljs.registerLanguage('shell', bash);
hljs.registerLanguage('json', json);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('yaml', yaml);
hljs.registerLanguage('yml', yaml);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('html', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('ts', typescript);

/**
 * Copy button component with feedback
 */
function CopyButton({ text, className = '' }) {
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
      className={`copy-btn ${copied ? 'copied' : ''} ${className}`}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span className="copy-label">Copied!</span>
        </>
      ) : (
        <>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          <span className="copy-label">Copy</span>
        </>
      )}
    </button>
  );
}

/**
 * Code block component with syntax highlighting
 */
function CodeBlock({ children, className }) {
  const codeRef = useRef(null);
  const language = className?.replace('language-', '') || '';
  const code = String(children).replace(/\n$/, '');

  const highlightedCode = useMemo(() => {
    if (language && hljs.getLanguage(language)) {
      try {
        return hljs.highlight(code, { language }).value;
      } catch (e) {
        console.error('Highlight error:', e);
      }
    }
    // Auto-detect language
    try {
      return hljs.highlightAuto(code).value;
    } catch (e) {
      return code;
    }
  }, [code, language]);

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-language">{language || 'code'}</span>
        <CopyButton text={code} className="code-copy-btn" />
      </div>
      <pre className="code-block">
        <code
          ref={codeRef}
          className={className}
          dangerouslySetInnerHTML={{ __html: highlightedCode }}
        />
      </pre>
    </div>
  );
}

/**
 * Inline code component
 */
function InlineCode({ children }) {
  return <code className="inline-code">{children}</code>;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(date) {
  if (!date) return '';
  const d = new Date(date);
  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

/**
 * Markdown renderer with custom components
 */
function MarkdownContent({ content }) {
  return (
    <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }) {
          if (inline) {
            return <InlineCode {...props}>{children}</InlineCode>;
          }
          return <CodeBlock className={className}>{children}</CodeBlock>;
        },
        // Style links
        a({ node, children, ...props }) {
          return (
            <a {...props} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          );
        },
        // Style lists
        ul({ node, children, ...props }) {
          return <ul className="markdown-list" {...props}>{children}</ul>;
        },
        ol({ node, children, ...props }) {
          return <ol className="markdown-list ordered" {...props}>{children}</ol>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
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
                  <div className="message-header-left">
                    <span className="message-role">
                      {message.role === 'user' ? 'You' : 'Assistant'}
                    </span>
                    {message.timestamp && (
                      <span className="message-time">
                        {formatTimestamp(message.timestamp)}
                      </span>
                    )}
                  </div>
                  {message.role === 'assistant' && message.content && (
                    <CopyButton text={message.content} />
                  )}
                </div>
                <div className="message-text">
                  {message.role === 'assistant' ? (
                    <MarkdownContent content={message.content} />
                  ) : (
                    message.content
                  )}
                </div>
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
