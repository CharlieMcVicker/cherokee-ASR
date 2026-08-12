export interface TranscribeResult {
  transcription: string;
  syllabary: string;
  confidence?: number;
  error?: string;
}

export interface PyWebViewApi {
  transcribe_pcm(pcmData: number[] | Float32Array, sampleRate?: number): Promise<TranscribeResult>;
}

declare global {
  interface Window {
    pywebview?: {
      api: PyWebViewApi;
    };
  }
}
