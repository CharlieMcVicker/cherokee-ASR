import { useCallback, useState } from 'react';
import type { TranscribeResult } from '../types/pywebview';

export function usePyWebView() {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const transcribePcm = useCallback(
    async (
      pcmData: Float32Array | number[],
      sampleRate: number = 16000
    ): Promise<TranscribeResult> => {
      setIsTranscribing(true);
      setError(null);

      try {
        const rawArray =
          pcmData instanceof Float32Array ? Array.from(pcmData) : pcmData;

        if (window.pywebview?.api?.transcribe_pcm) {
          const result = await window.pywebview.api.transcribe_pcm(
            rawArray,
            sampleRate
          );
          if (result.error) {
            setError(result.error);
          }
          return result;
        } else {
          const response = await fetch('/api/transcribe-pcm', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              pcmData: rawArray,
              sampleRate,
            }),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result: TranscribeResult = await response.json();
          if (result.error) {
            setError(result.error);
          }
          return result;
        }
      } catch (err: any) {
        const errMsg = err?.message || 'Failed to transcribe PCM audio';
        setError(errMsg);
        return { transcription: '', syllabary: '', error: errMsg };
      } finally {
        setIsTranscribing(false);
      }
    },
    []
  );

  return {
    transcribePcm,
    isTranscribing,
    error,
  };
}
