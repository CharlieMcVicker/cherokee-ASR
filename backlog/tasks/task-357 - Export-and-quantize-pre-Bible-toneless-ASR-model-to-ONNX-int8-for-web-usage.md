---
id: TASK-357
title: Export and quantize pre-Bible toneless ASR model to ONNX int8 for web usage
status: Done
assignee:
  - '@myself'
created_date: '2026-09-21 17:25'
updated_date: '2026-09-21 17:27'
labels: []
dependencies: []
ordinal: 383000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Download the pre-Bible toneless Cherokee ASR model checkpoint (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2) and export an int8 quantized ONNX model for automatic-speech-recognition suitable for web usage using optimum-cli.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Download pre-Bible toneless model checkpoint revision 76e62140955f4738abdab345ea34068b02d8d2a2 to a local directory
- [x] #2 Export and quantize model to int8 ONNX format for automatic-speech-recognition task via optimum-cli
- [x] #3 Verify exported ONNX model artifacts and tokenizer/preprocessor configuration files exist and are ready for web deployment
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Install optimum and onnx/onnxruntime in conda environment if needed.
2. Download model charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2 to local directory using huggingface-cli.
3. Run optimum-cli export onnx --model <local_dir> --task automatic-speech-recognition --quantize int8 <export_dir>.
4. Validate the exported ONNX model and tokenizer/feature extractor files.
5. Check acceptance criteria, write final summary, and mark task Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Downloaded checkpoint charliemcvicker/length-only-20260704-155307-asr-cherokee-colon (revision 76e62140955f4738abdab345ea34068b02d8d2a2) to pretrained_models/prebible-toneless-asr. Exported ONNX graph via optimum-cli and performed dynamic int8 quantization on MatMul layers to pretrained_models/prebible-toneless-asr-onnx/model_quantized.onnx (340 MB vs 1.2 GB original, 72% reduction). Structured for Transformers.js / ONNX Runtime Web. Verified live audio inference against PyTorch reference yielding 97.99% frame token agreement and identical transcription.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exported and quantized pre-Bible toneless Cherokee Wav2Vec2 ASR model to ONNX int8 for web deployment. Downloaded revision 76e62140955f4738abdab345ea34068b02d8d2a2 to pretrained_models/prebible-toneless-asr, exported ONNX graph to pretrained_models/prebible-toneless-asr-onnx/model.onnx, and quantized MatMul weights to model_quantized.onnx (340 MB). Validated inference against PyTorch reference with identical CTC decoded phonetic output ('itata hato utsutsathvki').
<!-- SECTION:FINAL_SUMMARY:END -->
