import { useCallback, useEffect, useState, useRef } from 'react';
import { usePyWebView } from './usePyWebView';
import type { TranscribeResult } from '../types/pywebview';

export interface UseVadOptions {
  enabled?: boolean;
  deviceId?: string;
  onTranscriptionComplete?: (result: TranscribeResult) => void;
  onTranscriptionError?: (error: string) => void;
  startOnLoad?: boolean;
}

export function useVad(options: UseVadOptions = {}) {
  const { transcribePcm, isTranscribing, error: transcribeError } = usePyWebView();
  const [latestResult, setLatestResult] = useState<TranscribeResult | null>(null);
  const [listening, setListening] = useState<boolean>(false);
  const [userSpeaking, setUserSpeaking] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [errored, setErrored] = useState<boolean>(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  
  // Use refs for callbacks to avoid re-initializing audio stream when functions change
  const onCompleteRef = useRef(options.onTranscriptionComplete);
  const onErrorRef = useRef(options.onTranscriptionError);

  useEffect(() => {
    onCompleteRef.current = options.onTranscriptionComplete;
    onErrorRef.current = options.onTranscriptionError;
  }, [options.onTranscriptionComplete, options.onTranscriptionError]);

  const handleSpeechEnd = useCallback(
    async (audio: Float32Array) => {
      setUserSpeaking(false);
      const result = await transcribePcm(audio, 16000);
      setLatestResult(result);
      if (result.error) {
        onErrorRef.current?.(result.error);
      } else {
        onCompleteRef.current?.(result);
      }
    },
    [transcribePcm]
  );

  const cleanupAudio = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.onmessage = null;
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      if (audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      audioContextRef.current = null;
    }
    setListening(false);
    setUserSpeaking(false);
  }, []);

  const initAudio = useCallback(async () => {
    if (streamRef.current || audioContextRef.current) {
      return;
    }

    try {
      setLoading(true);
      setErrored(false);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: options.deviceId
          ? { deviceId: { exact: options.deviceId }, sampleRate: 16000, channelCount: 1 }
          : { sampleRate: 16000, channelCount: 1 },
      });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule('/vad-processor.js');

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, 'vad-audio-processor');

      workletNode.port.onmessage = (event) => {
        const { type, audio, samples } = event.data;
        if (type === 'SPEECH_START') {
          console.log('[VAD] Speech started');
          setUserSpeaking(true);
        } else if (type === 'SPEECH_END') {
          console.log(`[VAD] Speech ended (samples: ${samples ?? 0}, hasAudio: ${!!audio})`);
          if (audio) {
            handleSpeechEnd(audio);
          } else {
            setUserSpeaking(false);
          }
        }
      };

      source.connect(workletNode);
      workletNodeRef.current = workletNode;

      setLoading(false);
      setListening(true);
    } catch (err) {
      console.error('Failed to initialize AudioWorklet VAD:', err);
      setLoading(false);
      setErrored(true);
    }
  }, [options.deviceId, handleSpeechEnd]);

  useEffect(() => {
    if (!options.enabled) {
      cleanupAudio();
      return;
    }

    initAudio();

    return () => {
      cleanupAudio();
    };
  }, [options.enabled, options.deviceId, initAudio, cleanupAudio]);

  const start = useCallback(async () => {
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
      setListening(true);
    } else if (!streamRef.current) {
      await initAudio();
    }
  }, [initAudio]);

  const pause = useCallback(async () => {
    if (audioContextRef.current && audioContextRef.current.state === 'running') {
      await audioContextRef.current.suspend();
      setListening(false);
      setUserSpeaking(false);
    }
  }, []);

  const toggle = useCallback(async () => {
    if (listening) {
      await pause();
    } else {
      await start();
    }
  }, [listening, start, pause]);

  return {
    listening,
    userSpeaking,
    loading,
    errored,
    isTranscribing,
    error: transcribeError,
    latestResult,
    start,
    pause,
    toggle,
  };
}
