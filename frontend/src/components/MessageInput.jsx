import { useState, forwardRef, useRef, useImperativeHandle } from 'react';
import './MessageInput.css';

/**
 * Input box for sending messages
 */
const MessageInput = forwardRef(function MessageInput({ onSend, disabled }, ref) {
  const [input, setInput] = useState('');
  const inputRef = useRef(null);

  // Expose focus method to parent
  useImperativeHandle(ref, () => ({
    focus: () => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    },
  }));

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className="message-input" onSubmit={handleSubmit}>
      <input
        ref={inputRef}
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your documents..."
        disabled={disabled}
        autoFocus
      />
      <button type="submit" disabled={disabled || !input.trim()}>
        Send
      </button>
    </form>
  );
});

export default MessageInput;
