import { Component } from 'react';
import './ErrorBoundary.css';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleClearData = () => {
    if (window.confirm('This will clear all local data and reload the app. Continue?')) {
      localStorage.clear();
      window.location.reload();
    }
  };

  handleDismiss = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary-content">
            <div className="error-boundary-icon">
              <svg viewBox="0 0 100 100" width="80" height="80" fill="none" stroke="currentColor" strokeWidth="4">
                <circle cx="50" cy="50" r="40" />
                <path d="M50 30 L50 55" strokeLinecap="round" />
                <circle cx="50" cy="68" r="3" fill="currentColor" />
              </svg>
            </div>

            <h1 className="error-boundary-title">Something went wrong</h1>
            <p className="error-boundary-message">
              The application encountered an unexpected error. This might be a temporary issue.
            </p>

            {this.state.error && (
              <details className="error-boundary-details">
                <summary>Technical Details</summary>
                <pre>
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}

            <div className="error-boundary-actions">
              <button className="error-boundary-button primary" onClick={this.handleReload}>
                Reload App
              </button>
              <button className="error-boundary-button secondary" onClick={this.handleDismiss}>
                Try to Continue
              </button>
              <button className="error-boundary-button danger" onClick={this.handleClearData}>
                Clear Data & Reload
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
