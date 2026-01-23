import './SettingsPanel.css';

/**
 * Settings panel for configuring query options
 */
function SettingsPanel({ settings, onChange, onClose }) {
  const handleProviderChange = (provider) => {
    onChange({ llmProvider: provider });
  };

  const handleTopKChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (value >= 1 && value <= 20) {
      onChange({ topK: value });
    }
  };

  const handleTemperatureChange = (e) => {
    const value = parseFloat(e.target.value);
    onChange({ temperature: value });
  };

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Query Settings</h3>
        <button className="close-btn" onClick={onClose} title="Close settings">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div className="settings-content">
        <div className="setting-group">
          <label className="setting-label">LLM Provider</label>
          <div className="provider-options">
            <button
              className={`provider-btn ${settings.llmProvider === 'ollama' ? 'active' : ''}`}
              onClick={() => handleProviderChange('ollama')}
            >
              <span className="provider-name">Ollama</span>
              <span className="provider-tag free">Free</span>
            </button>
            <button
              className={`provider-btn ${settings.llmProvider === 'claude' ? 'active' : ''}`}
              onClick={() => handleProviderChange('claude')}
            >
              <span className="provider-name">Claude</span>
              <span className="provider-tag paid">Paid</span>
            </button>
          </div>
          <p className="setting-hint">
            {settings.llmProvider === 'ollama'
              ? 'Using local Llama model (free, runs on your machine)'
              : 'Using Claude API (requires ANTHROPIC_API_KEY)'}
          </p>
        </div>

        <div className="setting-group">
          <label className="setting-label" htmlFor="topK">
            Context Chunks: {settings.topK}
          </label>
          <input
            type="range"
            id="topK"
            min="1"
            max="20"
            value={settings.topK}
            onChange={handleTopKChange}
            className="range-input"
          />
          <p className="setting-hint">
            Number of document chunks to retrieve for context (1-20)
          </p>
        </div>

        <div className="setting-group">
          <label className="setting-label" htmlFor="temperature">
            Temperature: {settings.temperature.toFixed(1)}
          </label>
          <input
            type="range"
            id="temperature"
            min="0"
            max="1"
            step="0.1"
            value={settings.temperature}
            onChange={handleTemperatureChange}
            className="range-input"
          />
          <p className="setting-hint">
            Lower = more focused, Higher = more creative
          </p>
        </div>
      </div>

      <div className="settings-footer">
        <div className="current-settings">
          <span className="settings-badge">{settings.llmProvider}</span>
          <span className="settings-badge">top_k: {settings.topK}</span>
          <span className="settings-badge">temp: {settings.temperature}</span>
        </div>
      </div>
    </div>
  );
}

export default SettingsPanel;
