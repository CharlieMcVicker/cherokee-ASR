import React, { useState, useCallback, useEffect, useRef } from 'react';
import Header from './components/Header';
import StatusBanner from './components/StatusBanner';
import DocumentCanvas from './components/DocumentCanvas';
import CommandFooter from './components/CommandFooter';
import { useVad } from './hooks/useVad';
import type { TranscribeResult } from './types/pywebview';
import './index.css';

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error: any }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20, color: 'red', background: '#fff', fontFamily: 'monospace' }}>
          <h2>Application Error</h2>
          <pre>{String(this.state.error?.stack || this.state.error)}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  );
};

const MainApp: React.FC = () => {

  const [documentText, setDocumentText] = useState<string>('');
  const [copiedStatus, setCopiedStatus] = useState<boolean>(false);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [commandFeedback, setCommandFeedback] = useState<string | null>(null);
  const [isSetupComplete, setIsSetupComplete] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [isFetchingMic, setIsFetchingMic] = useState<boolean>(true);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const copyTimeoutRef = useRef<number | null>(null);
  const feedbackTimeoutRef = useRef<number | null>(null);

  const triggerFeedback = useCallback((msg: string) => {
    setCommandFeedback(msg);
    if (feedbackTimeoutRef.current) {
      window.clearTimeout(feedbackTimeoutRef.current);
    }
    feedbackTimeoutRef.current = window.setTimeout(() => {
      setCommandFeedback(null);
    }, 2500);
  }, []);

  const fetchDevices = useCallback(async () => {
    try {
      setIsFetchingMic(true);
      setPermissionError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices.filter((d) => d.kind === 'audioinput');
      setDevices(audioInputs);
      if (audioInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(audioInputs[0].deviceId);
      }
    } catch (err: any) {
      console.error('Failed to get microphone permissions:', err);
      setPermissionError(err.message || 'Microphone access denied');
    } finally {
      setIsFetchingMic(false);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    fetchDevices();
  }, [fetchDevices]);

  const handleTranscriptionComplete = useCallback((result: TranscribeResult) => {
    // Check confidence score if present (scale 0..1 or 0..100)
    if (typeof result.confidence === 'number') {
      const confValue = result.confidence <= 1.0 ? result.confidence * 100 : result.confidence;
      if (confValue < 90) {
        triggerFeedback('⚠️ Sorry, please say that again');
        return;
      }
    }

    const transcribedText = result.syllabary || result.transcription;
    if (!transcribedText) return;
    const transcribed = transcribedText.trim();
    if (!transcribed) return;

    const lower = transcribed.toLowerCase();

    if (lower === 'clear all' || lower === 'clear' || transcribed === 'Ꭷ' || lower === 'Ꭷ') {
      setDocumentText('');
      triggerFeedback(`⚡ Command Executed: Canvas Cleared (${transcribed})`);
      return;
    }

    if (lower === 'delete' || lower === 'backspace' || lower === 'erase' || transcribed === 'ᎼᏏ' || lower === 'ᎼᏏ') {
      setDocumentText((prev) => {
        const words = prev.trim().split(/\s+/);
        if (words.length <= 1) return '';
        words.pop();
        return words.join(' ');
      });
      triggerFeedback(`⚡ Command Executed: Word Deleted (${transcribed})`);
      return;
    }

    setDocumentText((prev) => {
      const trimmed = prev.trim();
      return trimmed ? `${trimmed} ${transcribed}` : transcribed;
    });
  }, [triggerFeedback]);

  const handleTranscriptionError = useCallback((err: string) => {
    console.error('Transcription error:', err);
  }, []);

  const { listening, userSpeaking, isTranscribing, toggle, loading: vadLoading } = useVad({
    enabled: isSetupComplete,
    deviceId: selectedDeviceId,
    onTranscriptionComplete: handleTranscriptionComplete,
    onTranscriptionError: handleTranscriptionError,
  });

  // Auto-clipboard sync
  useEffect(() => {
    if (!documentText) return;

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(documentText)
        .then(() => {
          setCopiedStatus(true);
          if (copyTimeoutRef.current) {
            window.clearTimeout(copyTimeoutRef.current);
          }
          copyTimeoutRef.current = window.setTimeout(() => {
            setCopiedStatus(false);
          }, 2000);
        })
        .catch((err) => {
          console.error('Failed to copy text to clipboard:', err);
        });
    }
  }, [documentText]);

  if (!isSetupComplete) {
    if (isFetchingMic) {
      return (
        <div className="setup-container" style={{ padding: '60px 40px', maxWidth: '600px', margin: '40px auto', background: '#fff', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)', textAlign: 'center' }}>
          <div className="spinner" style={{ fontSize: '32px', marginBottom: '16px' }}>🎙️⏳</div>
          <h2 style={{ fontSize: '22px', color: '#111', marginBottom: '8px' }}>Connecting to your microphone...</h2>
          <p style={{ color: '#666', fontSize: '14px' }}>Please grant permission if prompted by your browser.</p>
        </div>
      );
    }

    return (
      <div className="setup-container" style={{ padding: '40px', maxWidth: '600px', margin: '40px auto', background: '#fff', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
        <h2 style={{ fontSize: '24px', marginBottom: '16px', color: '#111' }}>🎙️ Microphone Setup</h2>
        <p style={{ marginBottom: '24px', color: '#555', lineHeight: '1.5' }}>
          Select your primary microphone below to begin hands-free Cherokee Syllabary transcription.
        </p>

        {permissionError && (
          <div style={{ padding: '12px', background: '#fce8e6', color: '#c5221f', borderRadius: '6px', marginBottom: '20px' }}>
            ⚠️ {permissionError}. Please allow microphone access in system preferences.
          </div>
        )}

        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>Microphone Source:</label>
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            style={{ width: '100%', padding: '10px', fontSize: '16px', borderRadius: '6px', border: '1px solid #ccc' }}
          >
            {devices.map((device, index) => (
              <option key={device.deviceId || index} value={device.deviceId}>
                {device.label || `Microphone ${index + 1}`}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={fetchDevices}
            style={{ padding: '10px 16px', background: '#f1f3f4', border: '1px solid #ccc', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            🔄 Refresh Devices
          </button>
          <button
            onClick={() => setIsSetupComplete(true)}
            style={{ padding: '10px 24px', background: '#1a73e8', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold', flex: 1 }}
          >
            Start Transcribing →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Header
        selectedDeviceId={selectedDeviceId}
        onSelectDevice={(deviceId) => setSelectedDeviceId(deviceId)}
      />
      <StatusBanner
        listening={listening}
        userSpeaking={userSpeaking}
        isTranscribing={isTranscribing || vadLoading}
        commandFeedback={commandFeedback}
        onTogglePause={toggle}
      />
      <DocumentCanvas text={documentText} onChangeText={(newText) => setDocumentText(newText)} />
      <CommandFooter copiedStatus={copiedStatus} />
    </div>
  );
};

export default App;

