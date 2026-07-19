# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2025-9-26

import base64
import json
import librosa
import numpy as np
import os
import random
import re
import requests
import sys
import torch
import types
import uuid
from funasr import AutoModel
from collections import OrderedDict
from dotenv import load_dotenv
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from typing import Union

# Import additional packages for advanced functions
file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
sys.path.append(file_dir)
load_dotenv()
DISABLED_PACKAGES = [item.strip() for item in os.getenv("ASRPROCESSOR_DISABLED_PACKAGES", "").split(",") if item.strip()]
ap_dir = os.path.abspath(f"{file_dir}/../AudioProcessor")
if os.path.isdir(ap_dir):
    sys.path.append(ap_dir)
if "tencent" not in DISABLED_PACKAGES:
    try:
        from TencentASR import FlashRecognizer, FlashRecognitionRequest
    except Exception as e:
        DISABLED_PACKAGES.append("tencent")
        print(f"Failed to load Tencent ASR module and skip: {e}")
if "xunfei" not in DISABLED_PACKAGES:
    try:
        from XunfeiASR import XunfeiASR
    except Exception as e:
        DISABLED_PACKAGES.append("xunfei")
        print(f"Failed to load Xunfei ASR module and skip: {e}")
if "gemini" not in DISABLED_PACKAGES:
    try:
        import mimetypes
        from google import genai
    except Exception as e:
        DISABLED_PACKAGES.append("gemini")
        print(f"Failed to load Google Gemini module and skip: {e}")
if "whisper_v2" not in DISABLED_PACKAGES or "whisper_v3" not in DISABLED_PACKAGES:
    try:
        import whisper
    except Exception as e:
        DISABLED_PACKAGES.append("whisper_v2")
        DISABLED_PACKAGES.append("whisper_v3")
        print(f"Failed to load Whisper module and skip: {e}")
if "whisper_finetune" not in DISABLED_PACKAGES:
    try:
        from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
    except Exception as e:
        DISABLED_PACKAGES.append("whisper_finetune")
        print(f"Failed to load Transformers module and skip: {e}")
if "pyannote" not in DISABLED_PACKAGES:
    try:
        from pyannote.audio import Pipeline as pyannote_pipeline
    except Exception as e:
        DISABLED_PACKAGES.append("pyannote")
        print(f"Failed to load Pyannote module and skip: {e}")


# Main class
class ASRProcessor:
    def __init__(self,
                 is_asr: bool = False, fast_asr: bool = False, asr_model_dir: Union[str, list] = ["iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"],
                 is_vad: bool = False, vad_model_dir: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                 is_punc: bool = False, punc_model_dir: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                 is_timestamp: bool = False, timestamp_model_dir: str = "iic/speech_timestamp_prediction-v1-16k-offline",
                 is_emotion: bool = False, emotion_model_dir: str = "iic/emotion2vec_plus_large",
                 is_diarization: bool = False, diarization_model_dir: str = "pyannote/speaker-diarization-3.1",
                 is_asr_api: bool = False, api_config_path: str = f"{file_dir}/config.json",
                 verbose_log: bool = True, cuda_device: int = 0, ap=None):
        # Load ASR module
        self.is_asr = is_asr
        # ASR model directory
        if isinstance(asr_model_dir, str):
            self.asr_model_dirs = [asr_model_dir]
        else:
            self.asr_model_dirs = asr_model_dir
        # Use lite models
        self.fast_asr = fast_asr
        # Load VAD module
        self.is_vad = is_vad
        # VAD model directory
        self.vad_model_dir = vad_model_dir
        # Load punctuation restore module
        self.is_punc = is_punc
        # VAD model directory
        self.punc_model_dir = punc_model_dir
        # Load timestamp prediction module
        self.is_timestamp = is_timestamp
        # Timestamp prediction model directory
        self.timestamp_model_dir = timestamp_model_dir
        # Load emotion detection module
        self.is_emotion = is_emotion
        # Emotion detection model directory
        self.emotion_model_dir = emotion_model_dir
        # Load speaker diarization module
        self.is_diarization = is_diarization
        # Speaker diarization model directory
        self.diarization_model_dir = diarization_model_dir
        # Use ASR API
        self.is_asr_api = is_asr_api
        # API config file path
        self.api_config_path = api_config_path
        # Print verbose log
        self.verbose_log = verbose_log
        # Single CUDA device (-1 for CPU)
        self.cuda_device = cuda_device

        # Correct paths
        self.file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
        def correct_path(path: str):
            path = path.replace("\\", "/").rstrip("/")
            if not os.path.isabs(path):
                if os.path.exists(f"{self.file_dir}/{path}"):
                    path = f"{self.file_dir}/{path}"
            return path
        
        for i in range(len(self.asr_model_dirs)):
            self.asr_model_dirs[i] = correct_path(self.asr_model_dirs[i])
        self.vad_model_dir = correct_path(self.vad_model_dir)
        self.timestamp_model_dir = correct_path(self.timestamp_model_dir)
        self.emotion_model_dir = correct_path(self.emotion_model_dir)
        self.api_config_path = correct_path(self.api_config_path)
        if self.diarization_model_dir and os.path.isfile(os.path.abspath(f"{self.diarization_model_dir}/config.yaml")):
            self.diarization_model_dir = os.path.abspath(f"{self.diarization_model_dir}/config.yaml")

        # Get CUDA device
        self.get_device()

        # Init models
        self.load_model()

        # Init AudioProcessor
        if not ap:
            from AudioProcessor import AudioProcessor
            self.ap = AudioProcessor(cuda_device=self.cuda_device, verbose_log=False)
        else:
            self.ap = ap
        
        # Init ASR API
        if self.is_asr_api:
            self.api_config = {}
            if os.path.isfile(self.api_config_path):
                try:
                    with open(self.api_config_path, "r", encoding="utf-8") as f:
                        self.api_config = json.load(f)
                except:
                    pass
            if "tencent" not in DISABLED_PACKAGES:
                try:
                    self.tencent_asr = FlashRecognizer(appid=self.api_config['tencent_appid'], secret_id=self.api_config['tencent_secret_id'], secret_key=self.api_config['tencent_secret_key'])
                except Exception as e:
                    DISABLED_PACKAGES.append("tencent")
                    print(f"Failed to init Tencent ASR: {e}")
            if "xunfei" not in DISABLED_PACKAGES:
                try:
                    self.xunfei_asr = XunfeiASR(appid=self.api_config['xunfei_appid'], api_key=self.api_config['xunfei_api_key'], api_secret=self.api_config['xunfei_api_secret'])
                except Exception as e:
                    DISABLED_PACKAGES.append("xunfei")
                    print(f"Failed to init Xunfei ASR: {e}")
            if "gemini" not in DISABLED_PACKAGES:
                try:
                    self.gemini_asr = self.init_gemini(api_key=self.api_config['gemini_api_key'], base_url=self.api_config['gemini_base_url'])
                except Exception as e:
                    DISABLED_PACKAGES.append("gemini")
                    print(f"Failed to init Gemini ASR: {e}")
            self.jzx_asr_endpoint = os.getenv("JZX_ASR_ENDPOINT")
            if not self.jzx_asr_endpoint:
                self.jzx_asr_endpoint = self.api_config.get("jzx_asr_endpoint", "")
            if not self.jzx_asr_endpoint:
                DISABLED_PACKAGES.append("jzx")
        else:
            self.is_asr_api = False

    # Get CUDA device ID (-1=CPU)
    def get_device(self):
        if self.cuda_device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = "cpu"
        else:
            if self.cuda_device == -1:
                self.device = "cpu"
            else:
                self.device = f"cuda:{self.cuda_device}"
                try:
                    torch.cuda.set_device(self.device)
                except Exception as e:
                    print(f"Failed to set CUDA device: {e}")

    # Load local models
    def load_model(self):
        if self.is_asr:
            self.asr = {}
            for asr_model_dir in self.asr_model_dirs:
                print(f"Loading ASR model from: {os.path.abspath(asr_model_dir)}")
                if "sensevoice" in asr_model_dir.lower():
                    try:
                        if not self.fast_asr and self.is_vad:
                            self.asr['sensevoice'] = AutoModel(model=asr_model_dir, vad_model=self.vad_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
                        else:
                            print("Using fast_asr mode.")
                            self.asr['sensevoice'] = AutoModel(model=asr_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
                    except Exception as e:
                        DISABLED_PACKAGES.append("sensevoice")
                        print(f"Failed to load SenseVoice model: {e}")
                elif "paraformer" in asr_model_dir.lower():
                    try:
                        if not self.fast_asr and self.is_vad:
                            self.asr['paraformer'] = AutoModel(model=asr_model_dir, vad_model=self.vad_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
                        else:
                            print("Using fast_asr mode.")
                            self.asr['paraformer'] = AutoModel(model=asr_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
                    except Exception as e:
                        DISABLED_PACKAGES.append("paraformer")
                        print(f"Failed to load Paraformer model: {e}")
                elif "whisper" in asr_model_dir.lower() and "v2" in asr_model_dir.lower() and os.path.isfile(asr_model_dir):
                    try:
                        self.asr['whisper_v2'] = whisper.load_model(asr_model_dir)
                    except Exception as e:
                        DISABLED_PACKAGES.append("whisper_v2")
                        print(f"Failed to load Whisper V2 model: {e}")
                elif "whisper" in asr_model_dir.lower() and "v3" in asr_model_dir.lower() and os.path.isfile(asr_model_dir):
                    try:
                        self.asr['whisper_v3'] = whisper.load_model(asr_model_dir)
                    except Exception as e:
                        DISABLED_PACKAGES.append("whisper_v3")
                        print(f"Failed to load Whisper V3 model: {e}")
                elif "whisper" in asr_model_dir.lower() and "finetune" in asr_model_dir.lower() and os.path.isdir(asr_model_dir) and "whisper_finetune" not in DISABLED_PACKAGES:
                    try:
                        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        self.asr['whisper_finetune'] = types.SimpleNamespace()
                        self.asr['whisper_finetune'].model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model_dir, torch_dtype=torch_dtype, use_safetensors=True).to(self.device)
                        self.asr['whisper_finetune'].processor = WhisperProcessor.from_pretrained(asr_model_dir)
                    except Exception as e:
                        DISABLED_PACKAGES.append("whisper_finetune")
                        print(f"Failed to load Whisper finetune model: {e}")
        if self.is_vad:
            print(f"Loading VAD model from: {os.path.abspath(self.vad_model_dir)}")
            try:
                self.vad = AutoModel(model=self.vad_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
            except Exception as e:
                self.is_vad = False
                DISABLED_PACKAGES.append("funasr_vad")
                print(f"Failed to load FunASR VAD model: {e}")
        if self.is_punc:
            print(f"Loading punctuation restore model from: {os.path.abspath(self.vad_model_dir)}")
            try:
                self.punc = AutoModel(model=self.punc_model_dir, device=self.device, disable_pbar=not self.verbose_log, disable_update=True)
            except Exception as e:
                self.is_punc = False
                DISABLED_PACKAGES.append("funasr_punctuation")
                print(f"Failed to load punctuation restorer model: {e}")
        if self.is_timestamp:
            print(f"Loading timestamp prediction model from: {os.path.abspath(self.timestamp_model_dir)}")
            try:
                self.timestamp = pipeline(task=Tasks.speech_timestamp, model=self.timestamp_model_dir, disable_pbar=not self.verbose_log, disable_update=True)
            except Exception as e:
                self.is_timestamp = False
                DISABLED_PACKAGES.append("modelscope_timestamp")
                print(f"Failed to load timestamp prediction model: {e}")
        if self.is_emotion:
            print(f"Loading emotion detection model from: {os.path.abspath(self.emotion_model_dir)}")
            try:
                self.emotion = pipeline(task=Tasks.emotion_recognition, model=self.emotion_model_dir, disable_pbar=not self.verbose_log, disable_update=True)
            except Exception as e:
                self.is_emotion = False
                DISABLED_PACKAGES.append("modelscope_emotion")
                print(f"Failed to load emotion detection model: {e}")
        if self.is_diarization and "pyannote" not in DISABLED_PACKAGES:
            print(f"Loading Pyannote diarization model from: {os.path.abspath(self.diarization_model_dir)}")
            try:
                os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
                hf_token = os.getenv("HF_TOKEN")
                self.diarization = pyannote_pipeline.from_pretrained(self.diarization_model_dir, use_auth_token=hf_token)
                self.diarization.to(torch.device(self.device))
            except Exception as e:
                self.is_diarization = False
                DISABLED_PACKAGES.append("pyannote")
                print(f"Failed to load Pyannote diarization model: {e}")
    
    # Init Google Gemini LLM API
    def init_gemini(self, api_key: str, base_url: str = ""):
        class Gemini:
            def __init__(self, api_key: str, base_url: str = ""):
                if base_url:
                    http_options = genai.types.HttpOptions(base_url=base_url)
                else:
                    http_options = None
                self.client = genai.Client(
                    api_key=api_key,
                    http_options=http_options
                )
                self.generate_content_config = genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=0
                    ),
                    temperature=0.0
                )
                self.model_id = "gemini-2.5-flash"
            
            def get_result(self, audio: Union[str, bytes], audio_language: str = "unknown", asr_language: str = "zh-CN"):
                result_str = ""
                if isinstance(audio, str):
                    audio_path = os.path.abspath(audio)
                    if not os.path.isfile(audio_path):
                        print(f"Not an audio file: {audio_path}")
                        return ""
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    mime_type = mimetypes.guess_type(audio_path)[0]
                    if not mime_type:
                        mime_type = "audio/wav"
                else:
                    audio_bytes = audio
                    mime_type = "audio/wav"
                if audio_language == "unknown":
                    language_instruction = "You MUST automatically detect the language from the audio."
                else:
                    language_instruction = f"The language spoken in the audio is `{audio_language}`."
                user_prompt = f"""
**Task**: Transcribe the attached audio file.

**Instructions**:
1.  **Audio Language**: {language_instruction}
2.  **Output Language**: The final transcription text must be in `{asr_language}`.
3.  **Output Format**:
    - Provide only the pure, transcribed text.
    - Do NOT include any headers, introductory phrases (e.g., "Here is the transcription:"), or any other extraneous information.
4.  **Handling Uncertainty**:
    - If the audio is inaudible or the content is unintelligible, you MUST output an empty string: `""`.
"""
                user_prompt = user_prompt.strip()
                contents = [
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part.from_text(text=user_prompt),
                            genai.types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                        ]
                    )
                ]
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=self.generate_content_config
                )
                if response.candidates[0].content:
                    part = response.candidates[0].content.parts[0]
                    if hasattr(part, "text") and part.text:
                        result_str = part.text
                        result_str = result_str.strip()
                return result_str

        return Gemini(api_key=api_key, base_url=base_url)

    # Predict spoken text from audio using local model
    def asr_detection(self, wav_file: Union[str, list, bytes, np.ndarray], language: str = "auto", prompt: str = "", asr_engine: str = "paraformer", no_punc: bool = False, output_text_only: bool = False, output_raw_result: bool = False):
        result_list = []
        if asr_engine in DISABLED_PACKAGES:
            print("ASR engine init failed. Return empty result.")
            if output_text_only:
                return ""
            else:
                return result_list
        asr_engine = asr_engine.lower()
        if asr_engine in ["tencent", "xunfei", "gemini", "jzx"]:
            return self.asr_detection_api(wav_file=wav_file, language=language, asr_engine=asr_engine, prompt=prompt, no_punc=no_punc, output_text_only=output_text_only, output_raw_result=output_raw_result)
        if not self.is_asr or not self.asr:
            print("ASR models haven't been loaded. Return empty result.")
            if output_text_only:
                return ""
            else:
                return result_list
        if asr_engine not in ["sensevoice", "paraformer", "whisper_v2", "whisper_v3", "whisper_finetune"]:
            asr_engine = list(self.asr.keys())[0]
        if self.verbose_log:
            print("\nRunning module: asr_detection")
            print(f"Using ASR engine: {asr_engine}")
        search_pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
        delete_pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>"
        punc_pattern = r"[^\w\s]"
        if asr_engine == "sensevoice":
            if self.fast_asr:
                res = self.asr[asr_engine].generate(input=wav_file, cache={}, language=language, use_itn=True, batch_size=128)
            else:
                res = self.asr[asr_engine].generate(input=wav_file, cache={}, language=language, use_itn=True, batch_size_s=0)
            if output_raw_result:
                return res
            for clip in res:
                match = re.match(search_pattern, clip['text'])
                detect_language, detect_emotion, detect_audio_type, detect_itn, detect_text = match.groups()
                is_itn = True if detect_itn == "withitn" else False
                detect_text = re.sub(delete_pattern, "", detect_text)
                if detect_language.lower() == "zh":
                    detect_text = detect_text.replace(" ", "")
                if no_punc:
                    detect_text = re.sub(punc_pattern, "", detect_text)
                    detect_text = detect_text.lower()
                result = {
                    "key": clip['key'],
                    "language": detect_language.lower(),
                    "text": detect_text,
                    "emotion": detect_emotion.lower(),
                }
                result_list.append(result)
        elif asr_engine == "paraformer":
            hotword = ' '.join([word.strip() for word in prompt.split(',')])
            result_list = self.asr[asr_engine].generate(input=wav_file, hotword=hotword)
            if output_raw_result:
                return result_list
            for i, result in enumerate(result_list):
                if "timestamp" in list(result.keys()):
                    timestamp = []
                    texts = result['text'].split(" ")
                    value = result['timestamp']
                    value = [[round(point / 1000, 3) for point in clip] for clip in value]
                    if len(texts) < len(value):
                        texts.extend([""] * (len(value) - len(texts)))
                    for j in range(len(value)):
                        timestamp.append((texts[j], value[j]))
                    result_list[i]['timestamp'] = timestamp
                    # result_list[i]['text'] = self.remove_zh_space(result_list[i]['text'])
                    if not no_punc:
                        result_list[i]['text'] = self.punctuation_restore(result_list[i]['text'])
                if "language" not in list(result.keys()):
                    result_list[i]['language'] = self.detect_language(result['text'])
        elif asr_engine == "whisper_v2" or asr_engine == "whisper_v3":
            if isinstance(wav_file, str):
                audios = [whisper.load_audio(wav_file)]
                keys = [os.path.splitext(os.path.basename(wav_file))[0]]
            elif isinstance(wav_file, list):
                audios, keys = [], []
                for file in wav_file:
                    audios.append(whisper.load_audio(file))
                    keys.append(os.path.splitext(os.path.basename(file))[0])
            elif isinstance(wav_file, np.ndarray):
                audios = [wav_file.copy().flatten().astype(np.float32)]
                keys = ["0"]
            else:
                audios = [np.frombuffer(wav_file, np.int16).flatten().astype(np.float32) / 32768.0]
                keys = ["0"]
            for i, audio in enumerate(audios):
                # audio = whisper.pad_or_trim(audio)
                # n_mels = 128 if asr_engine == "whisper_v3" else 80
                # mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(self.device)
                # transcription_options = whisper.DecodingOptions(
                #     prompt=prompt,
                #     language=None if language == "auto" else language.lower(),
                #     task="transcribe"
                # )
                # res = self.asr[asr_engine].decode(mel, transcription_options)
                res = whisper.transcribe(
                    model=self.asr[asr_engine],
                    audio=audio,
                    initial_prompt=prompt,
                    word_timestamps=True,
                    language=None if language == "auto" else language.lower()
                )
                if output_raw_result:
                    return [res]
                word_list = res['segments'][i].get("words", [])
                timestamp = []
                if word_list:
                    for item in word_list:
                        timestamp.append((str(item['word']), [float(item['start']), float(item['end'])]))
                result = {
                    "key": keys[i],
                    "language": res['language'].lower(),
                    "text": res['segments'][i]['text'],
                    "timestamp": timestamp
                }
                result_list.append(result)
        elif asr_engine == "whisper_finetune":
            if isinstance(wav_file, str):
                audios = [whisper.load_audio(wav_file)]
                keys = [os.path.splitext(os.path.basename(wav_file))[0]]
            elif isinstance(wav_file, list):
                audios, keys = [], []
                for file in wav_file:
                    audios.append(whisper.load_audio(file))
                    keys.append(os.path.splitext(os.path.basename(file))[0])
            elif isinstance(wav_file, np.ndarray):
                audios = [wav_file.copy().flatten().astype(np.float32)]
                keys = ["0"]
            else:
                audios = [np.frombuffer(wav_file, np.int16).flatten().astype(np.float32) / 32768.0]
                keys = ["0"]
            for i, audio in enumerate(audios):
                input_features = self.asr[asr_engine].processor(wav_file, sampling_rate=16000, return_tensors="pt").input_features.half().to(self.device)
                prompt_ids = self.asr[asr_engine].processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)
                predicted_ids = self.asr[asr_engine].model.generate(input_features, task="transcribe", language="chinese", num_beams=1, prompt_ids=prompt_ids)
                res = self.asr[asr_engine].processor.batch_decode(predicted_ids, skip_special_tokens=True)
                result = {
                    "key": keys[i],
                    "language": self.detect_language(res[0]),
                    "text": res[0].strip()
                }
                result_list.append(result)
        if output_text_only:
            texts = ""
            for clip in result_list:
                if not clip['text']:
                    continue
                if clip['text'][-1] in [',', '.', '?', '!']:
                    texts = texts + clip['text'] + " "
                else:
                    texts = texts + clip['text']
            return texts
        else:
            return result_list

    # Predict spoken text from audio using online API
    def asr_detection_api(self, wav_file: Union[str, list, bytes], language: str = "auto", prompt: str = "", asr_engine: str = "tencent", no_punc: bool = False, output_text_only: bool = False, output_raw_result: bool = False):
        result_list = []
        asr_engine = asr_engine.lower()
        if not self.is_asr_api or asr_engine in DISABLED_PACKAGES:
            print("ASR API hasn't been loaded. Return empty result.")
            if output_text_only:
                return ""
            else:
                return result_list
        task_list = []
        punc_pattern = r"[^\w\s]"
        if asr_engine == "tencent":
            if isinstance(wav_file, str):
                wav_file = [wav_file]
            if isinstance(wav_file, list):
                for file in wav_file:
                    with open(file, "rb") as f:
                        audio_bytes = f.read()
                    task = {
                        "key": os.path.basename(file).split('.')[0],
                        "audio_bytes": audio_bytes,
                        "audio_format": os.path.basename(file).split('.')[1].lower()
                    }
                    task_list.append(task)
            else:
                task = {
                    "key": "0",
                    "audio_bytes": wav_file,
                    "audio_format": "wav"
                }
                task_list.append(task)
            if language == "" or language == "auto":
                tencent_language = "16k_zh"
            else:
                tencent_language = f"16k_{language.lower()}"
            if prompt:
                if "|" not in prompt:
                    prompt = ','.join([f"{word.strip()}|11" for word in prompt.split(',')])
            for task in task_list:
                text = ""
                try:
                    req = FlashRecognitionRequest(engine_type=tencent_language)
                    req.set_voice_format(task['audio_format'])
                    req.set_hotword_list(prompt)
                    response = self.tencent_asr.recognize(req, task['audio_bytes'])
                    res = json.loads(response)
                    text = res['flash_result'][0]['text']
                    if output_text_only:
                        return text
                    if no_punc:
                        text = re.sub(punc_pattern, "", text)
                        text = text.lower()
                except:
                    print(f"Failed in func asr_detection_api: {e}")
                if language == "" or language == "auto":
                    language = self.detect_language(text)
                result = {
                    "key": task['key'],
                    "language": language.lower(),
                    "text": text
                }
                result_list.append(result)
        elif asr_engine == "xunfei":
            if isinstance(wav_file, str):
                wav_file = [wav_file]
            if isinstance(wav_file, list):
                for file in wav_file:
                    task = {
                        "key": os.path.basename(file).split('.')[0],
                        "audio_path": os.path.abspath(file)
                    }
                    task_list.append(task)
            else:
                tmp_file = str(uuid.uuid4().hex) + ".wav"
                with open(tmp_file, "wb") as f:
                    f.write(wav_file)
                task = {
                    "key": tmp_file.split('.')[0],
                    "audio_path": os.path.abspath(tmp_file)
                }
                task_list.append(task)
            for task in task_list:
                text = ""
                try:
                    text = self.xunfei_asr.get_result(file_path=task['audio_path'], hotword=prompt)
                    if output_text_only:
                        return text
                    if no_punc:
                        text = re.sub(punc_pattern, "", text)
                        text = text.lower()
                except:
                    print(f"Failed in func asr_detection_api: {e}")
                if language == "" or language == "auto":
                    language = self.detect_language(text)
                result = {
                    "key": task['key'],
                    "language": language.lower(),
                    "text": text
                }
                result_list.append(result)
        elif asr_engine == "gemini":
            if isinstance(wav_file, str):
                wav_file = [wav_file]
            if isinstance(wav_file, list):
                for file in wav_file:
                    task = {
                        "key": os.path.basename(file).split('.')[0],
                        "audio_path": os.path.abspath(file)
                    }
                    task_list.append(task)
            else:
                tmp_file = str(uuid.uuid4().hex) + ".wav"
                with open(tmp_file, "wb") as f:
                    f.write(wav_file)
                task = {
                    "key": tmp_file.split('.')[0],
                    "audio_path": os.path.abspath(tmp_file)
                }
                task_list.append(task)
            if language == "zh":
                asr_language = "zh-CN"
            elif language == "en":
                asr_language = "en-US"
            elif language == "ja":
                asr_language = "ja-JP"
            for task in task_list:
                text = ""
                try:
                    text = self.gemini_asr.get_result(audio=task['audio_path'], audio_language=asr_language, asr_language=asr_language)
                    if output_text_only:
                        return text
                    if no_punc:
                        text = re.sub(punc_pattern, "", text)
                        text = text.lower()
                except:
                    print(f"Failed in func asr_detection_api: {e}")
                if language == "" or language == "auto":
                    language = self.detect_language(text)
                result = {
                    "key": task['key'],
                    "language": language.lower(),
                    "text": text
                }
                result_list.append(result)
        elif asr_engine == "jzx":
            if isinstance(wav_file, str):
                wav_file = [wav_file]
            if isinstance(wav_file, list):
                for file_path in wav_file:
                    key = os.path.splitext(os.path.basename(file_path))[0]
                    audio_data, sampling_rate = self.ap.read_audio(file_path)
                    task = {
                        "key": key,
                        "array": audio_data,
                        "sampling_rate": sampling_rate
                    }
                    task_list.append(task)
            else:
                key = str(uuid.uuid4().hex)
                audio_data, sampling_rate = self.ap.read_audio(wav_file)
                task = {
                    "key": key,
                    "array": audio_data,
                    "sampling_rate": sampling_rate
                }
                task_list.append(task)
            for task in task_list:
                text = ""
                word_list = []
                try:
                    wav_bytes = self.ap.ndarray_to_pcm_bytes(audio_data=task['array'], sampling_rate=task['sampling_rate'])
                    wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")
                    headers = {
                        "Content-Type": "application/json; charset=utf-8"
                    }
                    json_params = {
                        "source_type": 2,
                        "data": wav_base64,
                        "voice_format": "wav",
                        "context": prompt,
                        "enable_word_timestamps": True
                    }
                    res = requests.post(url=self.jzx_asr_endpoint, headers=headers, json=json_params)
                    if res.status_code == 200 and res.json()['code'] == 0:
                        response = res.json()
                        text = response['data']['text']
                        word_list = response['data']['word_list']
                    if output_text_only:
                        return text
                    if no_punc:
                        text = re.sub(punc_pattern, "", text)
                        text = text.lower()
                    timestamp = []
                    if word_list:
                        for item in word_list:
                            timestamp.append((str(item['word']), [float(item['start']), float(item['end'])]))
                except Exception as e:
                    print(f"Failed in func asr_detection_api: {e}")
                if language == "" or language == "auto":
                    language = self.detect_language(text)
                result = {
                    "key": task['key'],
                    "language": language.lower(),
                    "text": text,
                    "timestamp": timestamp
                }
                result_list.append(result)
        if output_text_only:
            text = ' '.join([result['text'] for result in result_list])
            return text
        return result_list

    # Detect voice activity in audio
    def vad_detection(self, wav_file: Union[str, bytes, np.ndarray], min_silence_sec: float = 0.5, min_clip_sec: float = 0.0, max_clip_sec: float = 0.0, format_to_sec: bool = True, output_folder: str = "", output_name: str = "", output_format: str = "wav"):
        if self.verbose_log:
            print("\nRunning module: vad_detection")
        if not self.is_vad:
            print("FunASR VAD model hasn't been loaded. Return empty result.")
            return []
        sampling_rate = 16000
        is_resample = False
        if isinstance(wav_file, str):
            audio_data, sampling_rate = self.ap.read_audio(wav_file)
            if sampling_rate != 16000:
                is_resample = True
                orig_audio_data = audio_data.copy()
                audio_data, _ = self.ap.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
            audio_data = self.ap.audio_to_mono(audio_data)
        elif isinstance(wav_file, bytes):
            audio_data = self.ap.raw_bytes_to_ndarray(audio_bytes=wav_file)
        else:
            audio_data = wav_file
        # res = self.vad.generate(input=audio_data, max_end_silence_time=int(min_silence_sec * 1000))
        max_end_silence_time = int(min_silence_sec * 1000)
        self.vad.kwargs['model_conf']['max_end_silence_time'] = max_end_silence_time
        self.vad.model.vad_opts.max_end_silence_time = max_end_silence_time
        res = self.vad.generate(input=audio_data, cache={})
        value = res[0]['value']
        value_sec = [[round(point / 1000, 3) for point in clip] for clip in value]
        if min_clip_sec > 0:
            merged = []
            current_start, current_end = value_sec[0]
            for i in range(1, len(value_sec)):
                next_start, next_end = value_sec[i]
                if current_end - current_start < min_clip_sec:
                    current_end = next_end
                else:
                    merged.append([current_start, current_end])
                    current_start, current_end = next_start, next_end
            if current_end - current_start >= min_clip_sec:
                merged.append([current_start, current_end])
            else:
                if merged:
                    merged[-1][1] = current_end
                else:
                    merged.append([current_start, current_end])
            value_sec = merged
        if max_clip_sec > 0:
            merged = []
            for current_start, current_end in value_sec:
                duration = current_end - current_start
                if duration > max_clip_sec:
                    num_clips = int(duration // max_clip_sec)
                    clip_duration = duration / (num_clips + 1)
                    for i in range(num_clips):
                        merged.append([current_start + i * clip_duration, current_start + (i + 1) * clip_duration])
                    merged.append([current_start + num_clips * clip_duration, current_end])
                else:
                    merged.append([current_start, current_end])
            value_sec = merged
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            if is_resample:
                audio_data = orig_audio_data
            for i, clip in enumerate(value_sec):
                audio_clip = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=clip[0], end_time=clip[1])
                if output_name:
                    base_name = output_name
                else:
                    if isinstance(wav_file, str):
                        base_name = os.path.basename(wav_file).split(".")[0]
                    else:
                        base_name = os.path.abspath(output_folder).replace('\\', '/').split('/')[-1]
                output_path = f"{output_folder}/{base_name}_{i}.{output_format}"
                self.ap.write_to_file(output_path=output_path, audio_data=audio_clip, sampling_rate=sampling_rate)
        if format_to_sec:
            return value_sec
        else:
            return value

    # Split audio by voice activity
    def asr_vad_split(self, wav_file: Union[str, bytes, None], min_clip_sec: float = 3.0, max_clip_sec: float = 0.0, sample_method: str = "uniform", min_silence_sec: float = 0.3, format_to_sec: bool = True, punc_restore: bool = True, asr_result_list=None):
        if asr_result_list:
            asr_result = asr_result_list[0]
        else:
            asr_result = self.asr['paraformer'].generate(input=wav_file)[0]
        timestamps = asr_result['timestamp']
        words = asr_result['text'].split(" ")
        processed_results = []
        current_text = ""
        current_timestamps = []
        current_start_time = timestamps[0][0]
        ts_index = 0
        while ts_index < len(words):
            if sample_method == "uniform" and max_clip_sec > 0:
                min_clip_ms = random.uniform(min_clip_sec, max_clip_sec) * 1000
                max_clip_ms = max_clip_sec * 1000
            elif sample_method == "normal" and max_clip_sec > 0:
                min_clip_ms = np.random.normal((min_clip_sec + max_clip_sec) / 2, 1) * 1000
                max_clip_ms = max_clip_sec * 1000
            elif max_clip_sec > 0:
                min_clip_ms = min_clip_sec * 1000
                max_clip_ms = max_clip_sec * 1000
            else:
                min_clip_ms = min_clip_sec * 1000
                max_clip_ms = 999999999
            word = words[ts_index]
            if word:
                current_text += word + " "
                current_timestamps.append(timestamps[ts_index])
                ts_index = ts_index + 1
                clip_duration = current_timestamps[-1][1] - current_start_time
                next_start_time = timestamps[ts_index][0] if ts_index < len(timestamps) else None
                if clip_duration >= min_clip_ms and (next_start_time is None or next_start_time - current_timestamps[-1][1] >= min_silence_sec * 1000 or clip_duration >= max_clip_ms):
                    processed_results.append({
                        "text": current_text.strip(),
                        "timestamp": [[t[0] - current_start_time, t[1] - current_start_time] for t in current_timestamps],
                        "timerange": [current_start_time, current_timestamps[-1][1]]
                    })
                    current_text = ""
                    current_timestamps = []
                    if ts_index < len(timestamps):
                        current_start_time = timestamps[ts_index][0]
            else:
                ts_index = ts_index + 1
        if current_text and current_timestamps:
            processed_results.append({
                "text": current_text.strip(),
                "timestamp": [[t[0] - current_start_time, t[1] - current_start_time] for t in current_timestamps],
                "timerange": [current_start_time, current_timestamps[-1][1]]
            })
        if format_to_sec:
            for result in processed_results:
                result["timestamp"] = [[round(t / 1000, 3) for t in timestamp] for timestamp in result["timestamp"]]
                result["timerange"] = [round(t / 1000, 3) for t in result["timerange"]]
        if punc_restore:
            for result in processed_results:
                result["text"] = self.punctuation_restore(result["text"])
        return processed_results

    # Restore punctuation in text
    def punctuation_restore(self, text: Union[str, list]):
        if self.verbose_log:
            print("\nRunning module: punctuation_restore")
        if not self.is_punc:
            print("Modelscope punctuation restorer model hasn't been loaded. Return original result.")
            return text
        if not text:
            return text
        try:
            res = self.punc.inference(input=text)
        except Exception as e:
            print(f"Failed in punctuation_restore: {e}")
            return text
        if isinstance(text, str):
            result = res[0]['text']
        else:
            result = [r['text'] for r in res]
        return result

    # Predict timestamp in audio
    def timestamp_prediction(self, wav_file: Union[str, bytes], text: str = "", format_to_sec: bool = True, output_timestamp_only: bool = False, output_raw_result: bool = False):
        if self.verbose_log:
            print("\nRunning module: timestamp_prediction")
        if isinstance(wav_file, str):
            audio_data, sampling_rate = self.ap.read_audio(wav_file)
            audio_data, _ = self.ap.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
            audio_data = self.ap.audio_to_mono(audio_data)
            audio_data = self.ap.float32_to_int16(audio_data)
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = wav_file
        if not text:
            text = self.asr_detection(wav_file=wav_file, no_punc=True, output_text_only=True)
        else:
            punc_pattern = r"[^\w\s]"
            text = re.sub(punc_pattern, "", text)
            text = text.lower()
        res = self.timestamp(input=(audio_bytes, text), data_type=("sound", "text"))
        if output_raw_result:
            return res
        value = res[0]['timestamp']
        if format_to_sec:
            value = [[round(point / 1000, 3) for point in clip] for clip in value]
        if output_timestamp_only:
            return value
        else:
            result = []
            texts = res[0]['text'].split(" ")
            if len(texts) < len(value):
                texts.extend([""] * (len(value) - len(texts)))
            for i in range(len(value)):
                result.append((texts[i], value[i]))
            return result

    # Detect character emotion in audio
    def emotion_detection(self, wav_file: Union[str, list, bytes], output_emotion_only: bool = False):
        if self.verbose_log:
            print("\nRunning module: emotion_detection")
        result_list = []
        if not self.is_emotion:
            print("Modelscope emotion detection model hasn't been loaded. Return empty result.")
            if output_emotion_only:
                return ""
            else:
                return result_list
        res = self.emotion(wav_file, granularity="utterance", extract_embedding=False)
        for r in res:
            key = r['key']
            labels = r['labels']
            scores = [round(score, 6) for score in r['scores']]
            top_results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
            emotion = top_results[0][0].split('/')[-1].strip()
            score = top_results[0][1]
            if score >= 0.95 and emotion not in ["excited"]:
                if emotion in ["fearful", "disgusted", "sad", "angry"]:
                    emotion_cls = "negative"
                else:
                    emotion_cls = "positive"
            else:
                emotion_cls = "neutral"
            result = {
                "key": key,
                "cls": emotion_cls,
                "emotion": emotion,
                "label_score": top_results
            }
            result_list.append(result)
        if output_emotion_only:
            if len(result_list) == 1:
                return result_list[0]['emotion']
            else:
                return [result['emotion'] for result in result_list]
        else:
            return result_list

    # Diarize speaker in audio
    def speaker_diarization(self, wav_file: Union[str, np.ndarray], sampling_rate: int = 16000, clustering_threshold: float = 0.0):
        if self.verbose_log:
            print("\nRunning module: speaker_diarization")
        if not self.is_diarization:
            print("Pyannote diarization model hasn't been loaded. Return empty result.")
            return {}
        file_name = ""
        if isinstance(wav_file, str):
            file_name = os.path.basename(wav_file).split('.')[0]
        elif isinstance(wav_file, np.ndarray):
            wav_file = {
                "waveform": self.ap.ndarray_to_torchaudio(audio_data=wav_file),
                "sample_rate": sampling_rate
            }
        if clustering_threshold > 0.0:
            self.diarization._pipelines['clustering']._instantiated['threshold'] = float(clustering_threshold)
        res = self.diarization(wav_file)
        result = OrderedDict()
        for segment, track, label in res.itertracks(yield_label=True):
            if file_name:
                label = f"{file_name}_S{label.split('_')[-1]}"
            if label not in result:
                result[label] = []
            result[label].append((round(segment.start, 3), round(segment.end, 3)))
        return result

    # Compute fundamental frequency (F0) in audio
    def f0_compute(self, wav_file: Union[str, np.ndarray], sampling_rate: int = 16000, fmin: float = 50.0, fmax: float = 300.0):
        if isinstance(wav_file, str):
            audio_data, sampling_rate = self.ap.read_audio(file_path=wav_file)
            audio_data = self.ap.audio_to_mono(audio_data=audio_data)
        else:
            audio_data = wav_file
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=fmin, fmax=fmax, sr=sampling_rate)
        return f0

    # Check if character is Chinese
    def is_chinese(self, char: chr):
        if '\u4e00' <= char <= '\u9fff':
            return True
        return False

    # Check if character is English
    def is_english(self, char: chr):
        if 'a' <= char.lower() <= 'z':
            return True
        return False

    # Detect language in text
    def detect_language(self, text: str):
        chinese_count = sum(self.is_chinese(char) for char in text)
        english_count = sum(self.is_english(char) for char in text)
        if english_count > chinese_count:
            return "en"
        else:
            return "zh"

    # Remove space between Chinese and English characters
    def remove_zh_space(self, text: str):
        result = ""
        word_list = text.split(" ")
        for word in word_list:
            if self.is_chinese(word):
                result = result + word
            else:
                if result[-1] == " ":
                    result = result + word + " "
                else:
                    result = result + " " + word + " "
        result = result.strip()
        return result
