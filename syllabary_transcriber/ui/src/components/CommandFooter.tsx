import React from 'react';

interface CommandFooterProps {
  copiedStatus: boolean;
  hasText: boolean;
  onCopy: () => void;
  onClear: () => void;
  onDeleteLastWord: () => void;
}

export const CommandFooter: React.FC<CommandFooterProps> = ({
  copiedStatus,
  hasText,
  onCopy,
  onClear,
  onDeleteLastWord,
}) => {
  return (
    <footer className="command-footer">
      <div className="action-buttons-group">
        <button
          className={`action-btn copy-btn ${copiedStatus ? 'active' : ''}`}
          onClick={onCopy}
          disabled={!hasText}
          title="📋 (Copy)"
          aria-label="Copy text to clipboard"
        >
          <span className="btn-icon">📋</span>
          <span className="btn-label">Copy to clipboard {copiedStatus ? '✓' : ''}</span>
        </button>

        <button
          className="action-btn delete-word-btn"
          onClick={onDeleteLastWord}
          disabled={!hasText}
          title="⌫ Delete Last Word"
          aria-label="Delete last word"
        >
          <span className="btn-icon">⌫</span>
          <span className="btn-label">Delete last word</span>
        </button>

        <button
          className="action-btn clear-btn"
          onClick={onClear}
          disabled={!hasText}
          title="🗑️ Clear All"
          aria-label="Clear document"
        >
          <span className="btn-icon">🗑️</span>
          <span className="btn-label">Clear all</span>
        </button>
      </div>
    </footer>
  );
};

export default CommandFooter;
