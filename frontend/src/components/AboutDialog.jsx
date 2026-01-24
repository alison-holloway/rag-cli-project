import { useState } from 'react';
import './AboutDialog.css';

const LICENSES = [
  { name: 'React', license: 'MIT', url: 'https://github.com/facebook/react' },
  { name: 'Tauri', license: 'MIT/Apache-2.0', url: 'https://github.com/tauri-apps/tauri' },
  { name: 'FastAPI', license: 'MIT', url: 'https://github.com/tiangolo/fastapi' },
  { name: 'ChromaDB', license: 'Apache-2.0', url: 'https://github.com/chroma-core/chroma' },
  { name: 'Ollama', license: 'MIT', url: 'https://github.com/ollama/ollama' },
  { name: 'highlight.js', license: 'BSD-3-Clause', url: 'https://github.com/highlightjs/highlight.js' },
  { name: 'react-markdown', license: 'MIT', url: 'https://github.com/remarkjs/react-markdown' },
];

function AboutDialog({ onClose }) {
  const [showLicenses, setShowLicenses] = useState(false);

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const openUrl = async (url) => {
    if (window.__TAURI__) {
      try {
        const { open } = await import('@tauri-apps/plugin-opener');
        await open(url);
      } catch (error) {
        console.error('Failed to open URL:', error);
        window.open(url, '_blank');
      }
    } else {
      window.open(url, '_blank');
    }
  };

  const handleOpenGitHub = () => openUrl('https://github.com/student/rag-assistant');

  return (
    <div className="about-dialog-backdrop" onClick={handleBackdropClick}>
      <div className="about-dialog">
        <button className="about-close-button" onClick={onClose} aria-label="Close">
          &times;
        </button>

        <div className="about-icon">
          <svg viewBox="0 0 100 100" width="80" height="80">
            <defs>
              <linearGradient id="iconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#667eea" />
                <stop offset="100%" stopColor="#764ba2" />
              </linearGradient>
            </defs>
            <rect x="10" y="10" width="80" height="80" rx="16" fill="url(#iconGradient)" />
            <text x="50" y="62" textAnchor="middle" fill="white" fontSize="36" fontWeight="bold" fontFamily="system-ui, -apple-system, sans-serif">
              R
            </text>
          </svg>
        </div>

        <h1 className="about-title">RAG Assistant</h1>
        <p className="about-version">Version 0.1.0</p>

        <p className="about-description">
          Local Document Intelligence powered by Retrieval-Augmented Generation.
          Query your documents using AI without sending data to the cloud.
        </p>

        <div className="about-features">
          <div className="about-feature">
            <span className="about-feature-icon">&#128196;</span>
            <span>PDF, Markdown, HTML support</span>
          </div>
          <div className="about-feature">
            <span className="about-feature-icon">&#128274;</span>
            <span>100% local processing</span>
          </div>
          <div className="about-feature">
            <span className="about-feature-icon">&#9889;</span>
            <span>Powered by Ollama</span>
          </div>
        </div>

        <div className="about-links">
          <button className="about-link-button" onClick={handleOpenGitHub}>
            View on GitHub
          </button>
        </div>

        <button
          className="about-licenses-toggle"
          onClick={() => setShowLicenses(!showLicenses)}
        >
          {showLicenses ? 'Hide' : 'Show'} Open Source Licenses
          <span className={`about-licenses-arrow ${showLicenses ? 'open' : ''}`}>&#9662;</span>
        </button>

        {showLicenses && (
          <div className="about-licenses">
            {LICENSES.map((lib) => (
              <button
                key={lib.name}
                className="about-license-item"
                onClick={() => openUrl(lib.url)}
              >
                <span className="about-license-name">{lib.name}</span>
                <span className="about-license-type">{lib.license}</span>
              </button>
            ))}
          </div>
        )}

        <p className="about-copyright">
          Built with Tauri, React, and FastAPI
        </p>
      </div>
    </div>
  );
}

export default AboutDialog;
