# ComfyUI-VibeVoice-HF-ASR

> ⚠️ **Notice**: This project is currently under active development. This README is still incomplete, and the program is **not yet recommended for direct use**.

A ComfyUI custom node for Automatic Speech Recognition (ASR) utilizing the Microsoft VibeVoice-ASR model. 

This project is a modified fork of [ComfyUI-kaola-VibeVoice-ASR](https://github.com/kana112233/ComfyUI-kaola-VibeVoice-ASR).

## Key Differences & Features

- **Focused on pure ASR**: All non-ASR related functionalities, such as Text-To-Speech (TTS), have been completely removed to keep the project lightweight and focused solely on speech recognition.
- **Hugging Face Transformers Integration**: Utilizes the [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) model, taking advantage of its recent merger into the official `transformers` library, instead of relying on custom standalone inference scripts.
- **Real-time Quantization**: Newly introduced support for on-the-fly, real-time model quantization during runtime to reduce VRAM consumption and improve efficiency.

## Installation
*(To be added)*

## Usage
*(To be added)*
