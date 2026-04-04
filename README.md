# ComfyUI-VibeVoice-HF-ASR

> ⚠️ **Notice**: This project is currently under active development. This README is still incomplete, and the program is **not yet recommended for direct use**.

A ComfyUI custom node for Automatic Speech Recognition (ASR) utilizing the Microsoft VibeVoice-ASR model. 

This project is a modified fork of [ComfyUI-kaola-VibeVoice-ASR](https://github.com/kana112233/ComfyUI-kaola-VibeVoice-ASR).

## Key Differences & Features

- **Focused on pure ASR**: All non-ASR related functionalities, such as Text-To-Speech (TTS), have been completely removed to keep the project lightweight and focused solely on speech recognition.
- **Hugging Face Transformers Integration**: Utilizes the [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) model, taking advantage of its recent merger into the official `transformers` library, instead of relying on custom standalone inference scripts.
- **Real-time Quantization**: Newly introduced support for on-the-fly, real-time model quantization during runtime to reduce VRAM consumption and improve efficiency. The default quantization method uses `bitsandbytes` (requires `pip install bitsandbytes`). If you opt for the Quanto methods, it requires `optimum-quanto` (`pip install optimum-quanto`).
- **Holistic Audio Chunking Strategy**: Unlike the original repo which hard-splits long audio into independent segments and runs separate LLM inferences (causing loss of context across chunk boundaries), this project leverages the native `transformers` implementation. It chunks the audio solely at the acoustic/semantic feature extraction level to save memory, but concatenates all features back together for a single, unified LLM inference pass. While this theoretically consumes slightly more VRAM for very long clips, it yields significantly better and more contextually coherent transcription results.

## Installation
*(To be added)*

## Usage
*(To be added)*
