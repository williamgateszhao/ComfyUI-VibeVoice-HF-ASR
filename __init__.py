from .nodes import VibeVoiceHFLoader, VibeVoiceHFTranscribe, VibeVoiceHFShowText, VibeVoiceHFSaveFile

NODE_CLASS_MAPPINGS = {
    "VibeVoiceHFLoader": VibeVoiceHFLoader,
    "VibeVoiceHFTranscribe": VibeVoiceHFTranscribe,
    "VibeVoiceHFShowText": VibeVoiceHFShowText,
    "VibeVoiceHFSaveFile": VibeVoiceHFSaveFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceHFLoader": "[HF Native] VibeVoice ASR Loader",
    "VibeVoiceHFTranscribe": "[HF Native] VibeVoice Transcribe (ASR)",
    "VibeVoiceHFShowText": "[HF Native] VibeVoice Show String",
    "VibeVoiceHFSaveFile": "[HF Native] VibeVoice Save File",
}

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
