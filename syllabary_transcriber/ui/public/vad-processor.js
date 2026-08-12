class VadAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.minSpeechFrames = 3;       // minimum consecutive active frames to trigger speech start (~60ms)
    this.redemptionFrames = 35;     // silence frames before speech end (~700ms)
    this.energyThreshold = 0.015;    // RMS threshold for speech detection
    
    this.speechFramesCount = 0;
    this.silenceFramesCount = 0;
    this.isSpeaking = false;
    this.speechBuffer = [];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    let sumSquares = 0;
    let zeroCrossings = 0;

    for (let i = 0; i < channelData.length; i++) {
      const sample = channelData[i];
      sumSquares += sample * sample;
      if (i > 0 && ((channelData[i - 1] >= 0 && sample < 0) || (channelData[i - 1] < 0 && sample >= 0))) {
        zeroCrossings++;
      }
    }

    const rms = Math.sqrt(sumSquares / channelData.length);
    const isFrameActive = rms >= this.energyThreshold;

    if (isFrameActive) {
      this.speechFramesCount++;
      this.silenceFramesCount = 0;
    } else {
      this.silenceFramesCount++;
    }

    if (!this.isSpeaking) {
      if (this.speechFramesCount >= this.minSpeechFrames) {
        this.isSpeaking = true;
        this.port.postMessage({ type: 'SPEECH_START' });
      }
    }

    if (this.isSpeaking) {
      // Collect 16kHz PCM data (assuming AudioContext at 16kHz or downstream resampling)
      for (let i = 0; i < channelData.length; i++) {
        this.speechBuffer.push(channelData[i]);
      }

      if (this.silenceFramesCount >= this.redemptionFrames) {
        this.isSpeaking = false;
        this.speechFramesCount = 0;
        this.silenceFramesCount = 0;

        const audioData = new Float32Array(this.speechBuffer);
        this.port.postMessage({ type: 'SPEECH_END', audio: audioData });
        this.speechBuffer = [];
      }
    }

    return true;
  }
}

registerProcessor('vad-audio-processor', VadAudioProcessor);
