import os
import torch
import folder_paths
import comfy.model_management

# Use Native Transformers rather than custom vibevoice modular code
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

def get_quantization_config(quantization_str, compute_dtype=torch.bfloat16):
    if quantization_str == "none":
        return None

    if quantization_str.startswith("bnb_"):
        try:
            from transformers import BitsAndBytesConfig
            if quantization_str == "bnb_int8":
                # Note: bnb_int8 is fundamentally broken for VibeVoice due to audio encoder outliers
                # causing CUDA kernel segfaults or infinite hangs in bitsandbytes.
                # It is highly recommended to use bnb_nf4 or quanto_int8 instead.
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["lm_head"]
                )
            elif quantization_str == "bnb_nf4":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                raise ValueError(f"Unknown bitsandbytes quantization method: {quantization_str}")
        except ImportError:
            raise ImportError(f"To use '{quantization_str}', please install bitsandbytes: pip install bitsandbytes")

    elif quantization_str.startswith("quanto_"):
        try:
            from transformers import QuantoConfig
            import torch

            # Monkey-patch Quanto's QLinear to ensure inputs are contiguous.
            # This prevents the "A is not contiguous" RuntimeError (specifically seen in Marlin kernel operations)
            # when transformers internally passes strided views into linear layers.
            try:
                import optimum.quanto.nn.qlinear as quanto_qlinear
                if not hasattr(quanto_qlinear.QLinear, "_original_forward"):
                    quanto_qlinear.QLinear._original_forward = quanto_qlinear.QLinear.forward
                    def patched_forward(self, input: torch.Tensor) -> torch.Tensor:
                        if not input.is_contiguous():
                            input = input.contiguous()
                        return self._original_forward(input)
                    quanto_qlinear.QLinear.forward = patched_forward
                    print("Applied contiguous patch to optimum.quanto QLinear")
            except ImportError:
                pass # optimum.quanto not present or version mismatch, skip patch

            quanto_weight = quantization_str.replace("quanto_", "")
            return QuantoConfig(weights=quanto_weight, modules_to_not_convert=["lm_head"])
        except ImportError:
            raise ImportError(f"To use '{quantization_str}' quantization, please install optimum-quanto: pip install optimum-quanto")
            
    else:
        raise ValueError(f"Unknown quantization method: {quantization_str}")

class VibeVoiceHFLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "microsoft/VibeVoice-ASR-HF", "tooltip": "HuggingFace repo ID or local path to model"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Model precision. Use bf16 for stability if supported, fp16 otherwise"}),
                "quantization": (["none", "bnb_nf4", "quanto_int8", "quanto_int4"], {"default": "bnb_nf4", "tooltip": "Quantization method. bnb_* -> bitsandbytes, quanto_* -> optimum-quanto."}),
                "device": (["cuda", "cpu", "mps", "xpu", "auto"], {"default": "auto", "tooltip": "Device to run the model on"}),
            },
        }

    RETURN_TYPES = ("VIBEVOICE_HF_MODEL",)
    RETURN_NAMES = ("vibevoice_hf_model",)
    FUNCTION = "load_model"
    CATEGORY = "VibeVoice HF ASR"

    def load_model(self, model_name, precision, device, quantization="bnb_nf4"):
        print(f"Loading Native VibeVoice HF ASR model: {model_name}")

        model_path = model_name
        if not os.path.exists(model_path):
            comfy_models_dir = folder_paths.models_dir
            search_paths = [
                os.path.join(comfy_models_dir, model_name),
                os.path.join(comfy_models_dir, "vibevoice", model_name),
            ]
            
            if "/" in model_name:
                model_basename = model_name.split("/")[-1]
                search_paths.append(os.path.join(comfy_models_dir, model_basename))
                search_paths.append(os.path.join(comfy_models_dir, "vibevoice", model_basename))
            
            for potential_path in search_paths:
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
        
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
            print(f"Resolved model path to: {model_path}")
        else:
            print(f"Model path not found locally, assuming HuggingFace repo ID: {model_name}")
            model_path = model_name

        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if device == "cpu":
            if precision != "fp32":
                print(f"Warning: forcing float32 for {device} device stability")
                dtype = torch.float32

        processor = AutoProcessor.from_pretrained(model_path)
        
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        quant_config = get_quantization_config(quantization, dtype)

        if quant_config is not None:
             print(f"Loading HF model with '{quantization}' quantization to {device}...")
             device_map = "auto" if device != "cpu" else "cpu"
             model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                 model_path,
                 torch_dtype=dtype,
                 device_map=device_map,
                 quantization_config=quant_config,
                 low_cpu_mem_usage=True,
             )
        else:
            print(f"Loading HF model to {device}...")
            device_map = "auto" if device != "cpu" else None
            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            
            if device == "cpu":
                model = model.to(device)

        model.eval()
        
        return ({"model": model, "processor": processor, "device": device, "dtype": dtype},)


class VibeVoiceHFTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vibevoice_hf_model": ("VIBEVOICE_HF_MODEL", {"tooltip": "Loaded Native VibeVoice HF model"}),
                "audio": ("AUDIO", {"tooltip": "Input audio to transcribe"}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 65536, "tooltip": "Max tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Randomness (0.0 = deterministic)"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability"}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Penalty for repetition (>1.0 reduces repetition)"}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Beam search width (1 = greedy)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
                "tokenizer_chunk_size": ("INT", {"default": 1440000, "min": 3200, "max": 2**31-1, "step": 3200, "tooltip": "Size of audio chunks for tokenizer to process to save memory. 1440000 = 60s at 24kHz"}),
            },
            "optional": {
                 "context_info": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter hotwords or context here...", "tooltip": "Context prompt / hotwords"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt_content", "json_content", "raw_text", "speaker_log")
    FUNCTION = "transcribe"
    CATEGORY = "VibeVoice HF ASR"

    def transcribe(self, vibevoice_hf_model, audio, max_new_tokens, temperature, top_p, repetition_penalty, num_beams, seed, tokenizer_chunk_size=1440000, context_info=""):
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(seed)
                 
        model = vibevoice_hf_model["model"]
        processor = vibevoice_hf_model["processor"]
        device = vibevoice_hf_model["device"]
        dtype = vibevoice_hf_model.get("dtype", torch.float32)
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3:
             waveform = waveform[0]
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform_np = waveform.squeeze().cpu().numpy()
        
        target_sr = 24000
        if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
            target_sr = processor.feature_extractor.sampling_rate
        elif hasattr(processor, "target_sample_rate"):
            target_sr = processor.target_sample_rate
        elif hasattr(processor, "sampling_rate"):
            target_sr = processor.sampling_rate
            
        if sample_rate != target_sr:
             import librosa
             print(f"Resampling audio from {sample_rate}Hz to target {target_sr}Hz...")
             waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=target_sr)
             sample_rate = target_sr
             
        print(f"Processing audio: {len(waveform_np)} samples at {target_sr}Hz with tokenizer_chunk_size {tokenizer_chunk_size}")

        prompt = context_info if context_info.strip() else None
        
        if hasattr(processor, "apply_transcription_request"):
            inputs = processor.apply_transcription_request(
                audio=waveform_np,
                prompt=prompt,
            ).to(device, dtype)
        else:
            raise NotImplementedError("apply_transcription_request is missing from processor. Ensure transformers >= 5.3.0.")

        do_sample = temperature > 0
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else processor.pad_id,
        }
        if hasattr(processor, "tokenizer"):
            generation_config["eos_token_id"] = processor.tokenizer.eos_token_id
            
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        if hasattr(model.config, "acoustic_tokenizer_chunk_size"):
            generation_config["acoustic_tokenizer_chunk_size"] = tokenizer_chunk_size
        else:
            generation_config["tokenizer_chunk_size"] = tokenizer_chunk_size

        from transformers import TextStreamer
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print(f"\n{'='*50}")
        print("Starting LLM Generation (Watch stream below):")
        with torch.no_grad():
            output_ids = model.generate(**inputs, streamer=streamer, **generation_config)
        print(f"\n{'='*50}\nGeneration Finished.")
            
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        
        try:
            transcription = processor.decode(generated_ids[0], return_format="parsed", skip_special_tokens=True)
            if isinstance(transcription, str):
                raw_text = transcription
                segments = [{"start_time": 0.0, "end_time": 0.0, "speaker_id": 0, "text": raw_text}]
            elif isinstance(transcription, list):
                segments = []
                for seg in transcription:
                    segments.append({
                        "start_time": seg.get("Start", 0.0),
                        "end_time": seg.get("End", 0.0),
                        "speaker_id": seg.get("Speaker", 0),
                        "text": seg.get("Content", "")
                    })
                raw_text = " ".join([seg["text"] for seg in segments])
            else:
                raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                segments = [{"start_time": 0.0, "end_time": 0.0, "speaker_id": 0, "text": raw_text}]
        except Exception as e:
            print(f"Error parsing segments: {e}")
            raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
            segments = [{"start_time": 0.0, "end_time": 0.0, "speaker_id": 0, "text": raw_text}]
            
        srt_output = self.generate_srt(segments)
        speaker_log = self.generate_log(segments)
        
        import json
        json_output = json.dumps({"raw_text": raw_text, "segments": segments}, indent=2, ensure_ascii=False)
        
        return (srt_output, json_output, raw_text, speaker_log)

    def generate_log(self, segments):
        if not isinstance(segments, list):
            return ""

        log_lines = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
                
            start = seg.get('start_time', 0.0)
            end = seg.get('end_time', 0.0)
            text = seg.get('text', '')
            speaker = seg.get('speaker_id', 'Unknown')
            
            def format_time(seconds):
                import datetime
                try:
                    dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=seconds)
                    return dt.strftime('%H:%M:%S,%f')[:-3]
                except:
                    return f"{seconds:.2f}"
            
            line = f"{i+1} {format_time(start)} --> {format_time(end)} speaker{speaker}: {text}"
            log_lines.append(line)
            
        return "\n\n".join(log_lines)

    def generate_srt(self, segments, speaker_prefix=False):
        if not isinstance(segments, list):
            return ""

        srt_lines = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

            start = seg.get('start_time', 0.0)
            end = seg.get('end_time', 0.0)
            text = seg.get('text', '')
            speaker = seg.get('speaker_id', 'Unknown')
            
            def format_time(seconds):
                import datetime
                dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=seconds)
                return dt.strftime('%H:%M:%S,%f')[:-3]
            
            srt_lines.append(f"{i+1}")
            srt_lines.append(f"{format_time(start)} --> {format_time(end)}")
            if speaker_prefix:
                srt_lines.append(f"speaker{speaker}: {text}")
            else:
                srt_lines.append(f"[{speaker}] {text}")
            srt_lines.append("")
            
        return "\n".join(srt_lines)


class VibeVoiceHFShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    CATEGORY = "VibeVoice HF ASR"
    OUTPUT_NODE = True

    def show_text(self, text):
        print(f"####################\n[VibeVoiceHFShowText] Content:\n{text}\n####################")
        return {"ui": {"text": [text]}, "result": (text,)}


class VibeVoiceHFSaveFile:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "vibevoice_hf_output"}),
                "file_extension": (["srt", "txt", "json"], {"default": "srt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_file"
    OUTPUT_NODE = True
    CATEGORY = "VibeVoice HF ASR"

    def save_file(self, text, filename_prefix="vibevoice_hf_output", file_extension="srt"):
        import time
        if "/" in filename_prefix or "\\" in filename_prefix:
            filename_prefix = filename_prefix.replace("\\", "/")
            subfolder = os.path.dirname(filename_prefix)
            base_prefix = os.path.basename(filename_prefix)
            full_output_dir = os.path.join(self.output_dir, subfolder)
            if not os.path.exists(full_output_dir):
                os.makedirs(full_output_dir)
        else:
            base_prefix = filename_prefix
            full_output_dir = self.output_dir
            subfolder = ""
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{base_prefix}_{timestamp}.{file_extension}"
        file_path = os.path.join(full_output_dir, filename)
        
        with open(file_path, "w", encoding="utf-8-sig") as f:
            f.write(text)
            
        print(f"Saved text to: {file_path}")
        return {
            "ui": {
                "text": [f"Saved to: {file_path}"],
                "file_info": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": "output"
                }]
            }, 
            "result": (file_path,)
        }
