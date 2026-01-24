import { useState } from 'react';
import './WelcomeScreen.css';

const WELCOME_STEPS = [
  {
    title: 'Welcome to RAG Assistant',
    description: 'Your local AI-powered document intelligence tool. Query your documents using natural language without sending data to the cloud.',
    icon: (
      <svg viewBox="0 0 100 100" width="120" height="120">
        <defs>
          <linearGradient id="welcomeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#667eea" />
            <stop offset="100%" stopColor="#764ba2" />
          </linearGradient>
        </defs>
        <rect x="10" y="10" width="80" height="80" rx="16" fill="url(#welcomeGradient)" />
        <text x="50" y="62" textAnchor="middle" fill="white" fontSize="36" fontWeight="bold" fontFamily="system-ui, -apple-system, sans-serif">
          R
        </text>
      </svg>
    ),
  },
  {
    title: 'Upload Your Documents',
    description: 'Add PDF, Markdown, HTML, or text files. Drag and drop or use the upload button. Your documents are stored locally and never leave your computer.',
    icon: (
      <svg viewBox="0 0 100 100" width="100" height="100" fill="none" stroke="currentColor" strokeWidth="3">
        <rect x="20" y="15" width="60" height="70" rx="4" />
        <path d="M35 50 L50 35 L65 50" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M50 35 L50 70" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: 'Ask Questions',
    description: 'Type natural language questions about your documents. The AI will search through your content and provide relevant answers with source citations.',
    icon: (
      <svg viewBox="0 0 100 100" width="100" height="100" fill="none" stroke="currentColor" strokeWidth="3">
        <circle cx="50" cy="45" r="25" />
        <path d="M50 35 L50 45" strokeLinecap="round" />
        <circle cx="50" cy="52" r="2" fill="currentColor" />
        <path d="M68 63 L85 80" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: 'Ready to Start!',
    description: 'You\'re all set! Start by uploading a document or try asking a question. Use Cmd+K to quickly focus the search, and Cmd+N for a new chat.',
    icon: (
      <svg viewBox="0 0 100 100" width="100" height="100" fill="none" stroke="currentColor" strokeWidth="3">
        <circle cx="50" cy="50" r="35" />
        <path d="M35 50 L45 60 L65 40" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

function WelcomeScreen({ onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);

  const handleNext = () => {
    if (currentStep < WELCOME_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handleSkip = () => {
    onComplete();
  };

  const step = WELCOME_STEPS[currentStep];
  const isLastStep = currentStep === WELCOME_STEPS.length - 1;

  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-icon">{step.icon}</div>
        <h1 className="welcome-title">{step.title}</h1>
        <p className="welcome-description">{step.description}</p>

        <div className="welcome-progress">
          {WELCOME_STEPS.map((_, index) => (
            <button
              key={index}
              className={`welcome-dot ${index === currentStep ? 'active' : ''} ${index < currentStep ? 'completed' : ''}`}
              onClick={() => setCurrentStep(index)}
              aria-label={`Go to step ${index + 1}`}
            />
          ))}
        </div>

        <div className="welcome-actions">
          {!isLastStep && (
            <button className="welcome-skip" onClick={handleSkip}>
              Skip
            </button>
          )}
          <button className="welcome-next" onClick={handleNext}>
            {isLastStep ? 'Get Started' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default WelcomeScreen;
