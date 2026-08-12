import React, { useEffect, useState } from 'react';

interface HeaderProps {
  selectedDeviceId?: string;
  onSelectDevice?: (deviceId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ selectedDeviceId, onSelectDevice }) => {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    async function getDevices() {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
          const allDevices = await navigator.mediaDevices.enumerateDevices();
          const audioInputs = allDevices.filter((d) => d.kind === 'audioinput');
          setDevices(audioInputs);
        }
      } catch (err) {
        console.error('Failed to enumerate audio devices:', err);
      }
    }

    getDevices();
  }, []);

  const handleRefreshDevices = async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = allDevices.filter((d) => d.kind === 'audioinput');
        setDevices(audioInputs);
      }
    } catch (err) {
      console.error('Failed to enumerate audio devices:', err);
    }
  };

  return (
    <header className="app-header">
      <div className="header-title">
        <h1>Cherokee Syllabary Transcriber</h1>
      </div>
      <div className="header-device-selector">
        <label htmlFor="mic-select">Microphone: </label>
        <select
          id="mic-select"
          value={selectedDeviceId || ''}
          onFocus={handleRefreshDevices}
          onChange={(e) => onSelectDevice?.(e.target.value)}
        >
          {devices.length === 0 && <option value="">Default Microphone</option>}
          {devices.map((device, index) => (
            <option key={device.deviceId || index} value={device.deviceId}>
              {device.label || `Microphone ${index + 1}`}
            </option>
          ))}
        </select>
      </div>
    </header>
  );

};

export default Header;
