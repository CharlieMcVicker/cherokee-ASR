import { useCallback, useEffect, useState, useRef } from 'react';
import { MicVAD } from '@ricky0123/vad-web';
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
  const vadRef = useRef<any>(null);

  const handleSpeechEnd = useCallback(
    async (audio: Float32Array) => {
      setUserSpeaking(false);
      const result = await transcribePcm(audio, 16000);
      setLatestResult(result);
      if (result.error) {
        options.onTranscriptionError?.(result.error);
      } else {
        options.onTranscriptionComplete?.(result);
      }
    },
    [transcribePcm, options]
  );

  useEffect(() => {
    if (!options.enabled) {
      setListening(false);
      setLoading(false);
      return;
    }

    let isMounted = true;

    async function initVad() {
      try {
        setLoading(true);

        const vad = await MicVAD.new({
          model: 'legacy',
          positiveSpeechThreshold: 0.8,
          redemptionMs: 700,
          baseAssetPath: '/',
          onnxWASMBasePath: '/',
          onSpeechStart: () => {
            if (isMounted) setUserSpeaking(true);
          },
          onSpeechEnd: (audio) => {
            if (isMounted) handleSpeechEnd(audio);
          },
        });







        if (isMounted) {
          vadRef.current = vad;
          setLoading(false);
          vad.start();
          setListening(true);
        }
      } catch (err) {
        console.error('Failed to initialize MicVAD:', err);
        if (isMounted) {
          setLoading(false);
          setErrored(true);
        }
      }
    }

    initVad();

    return () => {
      isMounted = false;
      if (vadRef.current) {
        vadRef.current.destroy();
        vadRef.current = null;
      }
    };
  }, [options.enabled, options.deviceId]);


  const start = useCallback(async () => {
    if (vadRef.current) {
      try {
        await vadRef.current.start();
        setListening(true);
      } catch (err) {
        console.error('Failed to start VAD:', err);
      }
    }
  }, []);

  const pause = useCallback(async () => {
    if (vadRef.current) {
      try {
        await vadRef.current.pause();
        setListening(false);
      } catch (err) {
        console.error('Failed to pause VAD:', err);
      }
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

