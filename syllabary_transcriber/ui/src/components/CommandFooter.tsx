import React from 'react';

interface CommandFooterProps {
  copiedStatus: boolean;
}

export const CommandFooter: React.FC<CommandFooterProps> = ({ copiedStatus }) => {
  return (
    <footer className="command-footer">
      <div className="command-instructions">
        <span className="command-tag">Spoken Commands:</span>
        <span className="command-item">
          Say <strong>"Delete"</strong> to remove last word
        </span>
        <span className="command-divider">•</span>
        <span className="command-item">
          Say <strong>"Clear All"</strong> to erase document
        </span>
      </div>
      <div className="clipboard-status">
        <span className={`clipboard-badge ${copiedStatus ? 'active' : ''}`}>
          {copiedStatus ? '✓ Copied to Clipboard' : 'Auto-Clipboard Ready'}
        </span>
      </div>
    </footer>
  );
};

export default CommandFooter;
