import React from 'react';

interface StatusBannerProps {
  listening: boolean;
  userSpeaking: boolean;
  isTranscribing: boolean;
  commandFeedback?: string | null;
  onTogglePause: () => void;
}

export const StatusBanner: React.FC<StatusBannerProps> = ({
  listening,
  userSpeaking,
  isTranscribing,
  commandFeedback,
  onTogglePause,
}) => {
  let statusText = 'PAUSED';
  let indicatorClass = 'status-dot dot-paused';

  if (commandFeedback) {
    statusText = commandFeedback;
    indicatorClass = 'status-dot dot-transcribing';
  } else if (listening) {
    if (userSpeaking) {
      statusText = 'LISTENING — User speaking...';
      indicatorClass = 'status-dot dot-speaking';
    } else if (isTranscribing) {
      statusText = 'LISTENING — Transcribing audio...';
      indicatorClass = 'status-dot dot-transcribing';
    } else {
      statusText = 'LISTENING — Speak a word syllable-by-syllable...';
      indicatorClass = 'status-dot dot-listening';
    }
  }

  return (
    <div className="status-banner">
      <div className="status-indicator">
        <span className={indicatorClass} />
        <span className="status-text">{statusText}</span>
      </div>
      <button className="toggle-pause-btn" onClick={onTogglePause}>
        {listening ? 'Pause' : 'Resume'}
      </button>
    </div>
  );
};

export default StatusBanner;
