# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2025-9-26

import audioread
import librosa
import io
import os
import sys
import urllib.request
import numpy as np
import pickle
import pyloudnorm as pyln
import soundfile as sf
import uuid
from audiostretchy.stretch import AudioStretch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Literal, Union

# Import additional packages for advanced functions
file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
sys.path.append(file_dir)
load_dotenv()
DISABLED_PACKAGES = [item.strip() for item in os.getenv("AUDIOPROCESSOR_DISABLED_PACKAGES", "").split(",") if item.strip()]
if "torch" not in DISABLED_PACKAGES:
    try:
        import torch
    except Exception as e:
        DISABLED_PACKAGES.append("torch")
        print(f"Failed to load torch runtime and skip.\n{e}")
if "mdx" not in DISABLED_PACKAGES:
    try:
        import onnxruntime as ort
    except Exception as e:
        DISABLED_PACKAGES.append("mdx")
        print(f"Failed to load ONNX runtime and skip.\n{e}")
if "noisereduce" not in DISABLED_PACKAGES:
    try:
        import noisereduce as nr
    except Exception as e:
        DISABLED_PACKAGES.append("noisereduce")
        print(f"Failed to load noisereduce and skip.\n{e}")
if "enhancer" not in DISABLED_PACKAGES:
    try:
        from resemble_enhance.enhancer.train import Enhancer, HParams
        from resemble_enhance.inference import inference as enhancer_api
    except Exception as e:
        DISABLED_PACKAGES.append("enhancer")
        print(f"Failed to load resemble package and skip.\n{e}")
if "separater" not in DISABLED_PACKAGES or "restorer" not in DISABLED_PACKAGES:
    try:
        import look2hear.models
        from silero_vad import load_silero_vad, get_speech_timestamps
    except Exception as e:
        DISABLED_PACKAGES.append("separater")
        DISABLED_PACKAGES.append("restorer")
        print(f"Failed to load look2hear package and skip.\n{e}")


# MDX-Net class
class ConvTDFNet:
    def __init__(self, target_name: str, L: int, dim_f: int, dim_t: int, n_fft: int, hop: int = 1024, device: str = "cpu"):
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.device = device if device else "cpu"
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(self.device)
        self.target_name = target_name
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x: torch.Tensor):
        x = x.to(self.device)
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        x = x[:, :, : self.dim_f].contiguous()
        return x

    def istft(self, x: torch.Tensor, freq_pad: torch.Tensor = None):
        x = x.to(self.device)
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        freq_pad = freq_pad.to(self.device)
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        if not torch.is_complex(x):
            x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        x = x.reshape([-1, c, self.chunk_size]).cpu()
        return x


# Main class
class AudioProcessor:
    def __init__(self,
                 is_denoise_vocal: bool = False, mdx_weights_file: str = "mdx/weights/UVR-MDX-NET-Inst_HQ_3.onnx",
                 is_enhance_vocal: bool = False, enhancer_weights_folder: str = "resemble_enhance/model_repo/enhancer_stage2",
                 is_separate_audio: bool = False, separater_weights_folder: str = "look2hear/checkpoints/TFGNet-Noise",
                 is_restore_audio: bool = False, restorer_weights_folder: str = "JusperLee/Apollo",
                 verbose_log: bool = True, cuda_device: int = 0, quality: int = 2):
        # Remove background noise or isolate vocal
        self.is_denoise_vocal = is_denoise_vocal
        # MDX weights file
        self.mdx_weights_file = mdx_weights_file.replace('\\', '/')
        # Enhance vocal from a low quality audio
        self.is_enhance_vocal = is_enhance_vocal
        # Enhancer weights folder
        self.enhancer_weights_folder = enhancer_weights_folder.replace('\\', '/')
        # Separate two vocals from an overlapping vocals audio
        self.is_separate_audio = is_separate_audio
        # Separater weights folder
        self.separater_weights_folder = separater_weights_folder.replace('\\', '/')
        # Restore low quality audio
        self.is_restore_audio = is_restore_audio
        # Restorer weights folder
        self.restorer_weights_folder = restorer_weights_folder.replace('\\', '/')
        # Print verbose log
        self.verbose_log = verbose_log
        # Single CUDA device (-1 for CPU)
        self.cuda_device = cuda_device
        # Quality preset (1=fast 2=balanced 3=best)
        self.quality = quality

        # Correct paths
        self.file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
        def correct_path(path: str):
            path = path.replace("\\", "/").rstrip("/")
            if not os.path.isabs(path):
                if os.path.exists(f"{self.file_dir}/{path}"):
                    path = f"{self.file_dir}/{path}"
            return path
        
        self.mdx_weights_file = correct_path(self.mdx_weights_file)
        self.enhancer_weights_folder = correct_path(self.enhancer_weights_folder)
        self.separater_weights_folder = correct_path(self.separater_weights_folder)
        self.restorer_weights_folder = correct_path(self.restorer_weights_folder)

        # Init models
        if "torch" not in DISABLED_PACKAGES:
            self.get_device()
            if self.is_denoise_vocal and os.path.isfile(self.mdx_weights_file) and "mdx" not in DISABLED_PACKAGES:
                try:
                    self.init_mdx_model()
                except Exception as e:
                    print(f"Failed to init denoiser model: {e}")
                    self.is_denoise_vocal = False
            else:
                self.is_denoise_vocal = False
            if self.is_enhance_vocal and os.path.isdir(self.enhancer_weights_folder) and "enhancer" not in DISABLED_PACKAGES:
                try:
                    self.init_enhancer_model()
                except Exception as e:
                    print(f"Failed to init enhancer model: {e}")
                    self.is_enhance_vocal = False
            else:
                self.is_enhance_vocal = False
            if self.is_separate_audio and os.path.isdir(self.separater_weights_folder) and "separater" not in DISABLED_PACKAGES:
                try:
                    self.init_separater_model()
                except Exception as e:
                    print(f"Failed to init separater model: {e}")
                    self.is_separate_audio = False
            else:
                self.is_separate_audio = False
            if self.is_restore_audio and os.path.isdir(self.restorer_weights_folder) and "restorer" not in DISABLED_PACKAGES:
                try:
                    self.init_restorer_model()
                except Exception as e:
                    print(f"Failed to init restorer model: {e}")
                    self.is_restore_audio = False
            else:
                self.is_restore_audio = False

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

    # Init MDX denoiser model
    def init_mdx_model(self):
        quality_segment_size = {
            1: 256,
            2: 1024,
            3: 2048
        }
        if self.cuda_device == -1 or not torch.cuda.is_available():
            self.mdx_model = ort.InferenceSession(self.mdx_weights_file, providers=["CPUExecutionProvider"])
        else:
            self.mdx_model = ort.InferenceSession(self.mdx_weights_file, providers=["CUDAExecutionProvider"], provider_options=[{'device_id': int(self.cuda_device)}])
        meta = self.mdx_model.get_modelmeta()
        meta_dict = dict(meta.custom_metadata_map.items())
        dim_f = int(meta_dict.get("dim_f", 3072))
        # dim_t = int(meta_dict.get("dim_t", 256))
        dim_t = 8
        n_fft = int(meta_dict.get("n_fft", 6144))
        hop = quality_segment_size.get(self.quality, 1024)
        self.mdx_net = ConvTDFNet(target_name="vocals", L=11, dim_f=dim_f, dim_t=dim_t, n_fft=n_fft, hop=hop, device=self.device)

    # Init Resemble enhancer model
    def init_enhancer_model(self):
        if self.quality == 1:
            nfe = 1
            solver = "euler"
        elif self.quality == 2:
            nfe = 64
            solver = "midpoint"
        elif self.quality == 3:
            nfe = 128
            solver = "midpoint"
        else:
            nfe = 128
            solver = "midpoint"
        run_dir = Path(self.enhancer_weights_folder)
        hp = HParams.load(run_dir)
        path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(path, map_location="cpu")["module"]
        self.enhancer = Enhancer(hp)
        self.enhancer.configurate_(nfe=nfe, solver=solver, lambd=0.5, tau=0.0)
        self.enhancer.load_state_dict(state_dict)
        self.enhancer.eval()
        self.enhancer.to(self.device)

    # Init MossFormer2 separater model via look2hear
    def init_separater_model(self):
        cfg = OmegaConf.load(f"{self.separater_weights_folder}/config.yaml")
        cfg.model.pop("_target_", None)
        self.separater = look2hear.models.ConvTasNet.from_pretrain(f"{self.separater_weights_folder}/best_model.pth", **cfg.model)
        self.separater.eval()
        self.separater.to(self.device)
        self.silero_vad = load_silero_vad()
    
    # Init Apollo restorer model via look2hear
    def init_restorer_model(self):
        weights_file = f"{self.restorer_weights_folder}/pytorch_model.bin"
        self.restorer = look2hear.models.BaseModel.from_pretrain(weights_file, sr=44100, win=20, feature_dim=256, layer=6)
        self.restorer.eval()
        self.restorer.to(self.device)

    # Module chain function
    def run_modules(self, audio_data: np.ndarray, module_chain: list = []):
        for module in module_chain:
            for method_name, params in module.items():
                method = getattr(self, method_name, None)
                if method:
                    audio_data = method(audio_data, **params)
                else:
                    print(f"Method {method_name} not exists.")
        return audio_data

    # Download an audio file from URL
    def download_audio(self, url: str, output_path: str = ""):
        if self.verbose_log:
            print("\nRunning module: download_audio")
            print(f"Download audio from: {url}")
        if not output_path:
            filename = url.split("?")[0].split("/")[-1]
            if "." not in filename:
                filename = f"{filename}.wav"
            output_path = f"/tmp/{filename}"
        urllib.request.urlretrieve(url, output_path)
        return output_path

    # Read audio from a local file, URL or binary, and convert it into numpy ndarray (np.float32)
    def read_audio(self, file_path: Union[str, bytes, io.BytesIO, np.ndarray]):
        if self.verbose_log:
            print("\nRunning module: read_audio")
            print(f"Read file from: {file_path}")
        is_temp = False
        if isinstance(file_path, np.ndarray):
            if self.verbose_log:
                print("Input audio is a np.ndarray object, assume sampling_rate=16000")
            audio_data = self.int16_to_float32(file_path)
            return audio_data, 16000
        if isinstance(file_path, io.BytesIO):
            file_path = file_path.getvalue()
        if isinstance(file_path, bytes):
            is_temp = True
            output_path = f"/tmp/{str(uuid.uuid4().hex)}.tmp"
            with open(output_path, "wb") as f:
                f.write(file_path)
            file_path = output_path
            if self.verbose_log:
                print(f"Temp audio file saved: {output_path}")
        if isinstance(file_path, str):
            if file_path.startswith("http"):
                is_temp = True
                file_path = self.download_audio(url=file_path)
        # audio_data, sampling_rate = sf.read(file_path, dtype="float32")
        with audioread.audio_open(file_path) as f:
            sampling_rate = f.samplerate
            channel_num = f.channels
            data = []
            for buffer in f:
                data.append(np.frombuffer(buffer, dtype=np.int16))
            data = np.concatenate(data)
        if channel_num > 1:
            data = data.reshape((-1, channel_num))
        else:
            data = data.reshape((-1,))
        audio_data = self.int16_to_float32(data)
        if is_temp and os.path.isfile(file_path):
            os.remove(file_path)
        return audio_data, sampling_rate

    # Convert audio to mono
    def audio_to_mono(self, audio_data: np.ndarray):
        if len(audio_data.shape) == 1:
            if self.verbose_log:
                print("Original channel num is 1. Skip.")
            return audio_data
        input_channels = audio_data.shape[1]
        if self.verbose_log:
            print("\nRunning module: audio_to_mono")
            print(f"Original channel num: {input_channels}")
            print(f"Target channel num: 1")
        audio_data = audio_data.reshape((-1, input_channels))
        if input_channels == 6:
            weights = np.array([1, 1, 1, 0.5, 0.7, 0.7])
        elif input_channels == 8:
            weights = np.array([1, 1, 1, 0.5, 0.7, 0.7, 0.5, 0.5])
        else:
            weights = np.ones(input_channels)
        output_audio = np.average(audio_data, axis=1, weights=weights).astype(np.float32)
        return output_audio

    # Convert audio to stereo
    def mono_to_stereo(self, audio_data: np.ndarray):
        if self.verbose_log:
            print("\nRunning module: mono_to_stereo")
            print(f"Original channel num: 1")
            print(f"Target channel num: 2")
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape((-1, 1))
        elif audio_data.shape[1] > 1:
            if self.verbose_log:
                print(f"Original channel num is {audio_data.shape[1]}. Skip.")
            return audio_data
        audio_data = np.repeat(audio_data, 2, 1)
        return audio_data

    # Apply gain to audio
    def audio_gain(self, audio_data: np.ndarray, gain_db: float, clip_limit: bool = False):
        if self.verbose_log:
            print("\nRunning module: audio_gain")
            print(f"Gain dB: {gain_db}")
            print(f"Apply clip limit: {clip_limit}")
        if gain_db == 0.0:
            if self.verbose_log:
                print("gain_db is zero. Skip.")
            return audio_data
        gain_factor = 10 ** (gain_db / 20.0)
        output_audio = audio_data * gain_factor
        if clip_limit:
            output_audio = np.clip(output_audio, -1, 1)
        return output_audio

    # Normalize audio to target peak dB
    def audio_normalize(self, audio_data: np.ndarray, target_peak_db: float = -0.1):
        if self.verbose_log:
            print("\nRunning module: audio_normalize")
            print(f"Target peak dB: {target_peak_db}")
        current_peak = np.max(np.abs(audio_data))
        if current_peak == 0.0:
            return audio_data
        current_peak_db = 20 * np.log10(current_peak)
        gain_db = target_peak_db - current_peak_db
        gain_factor = 10 ** (gain_db / 20)
        output_audio = audio_data * gain_factor
        output_audio = np.clip(output_audio, -1.0, 1.0)
        return output_audio

    # Normalize audio to target loudness
    def audio_loudness_control(self, audio_data: np.ndarray, sampling_rate: int, target_loudness: float = -23.0):
        if self.verbose_log:
            print("\nRunning module: audio_loudness_control")
        if audio_data.shape[0] / sampling_rate < 0.4:
            if self.verbose_log:
                print("Audio clip duration is less than 0.4 sec. Skip.")
            return audio_data
        loudness = self.meter_loudness(audio_data=audio_data, sampling_rate=sampling_rate)
        if self.verbose_log:
            print(f"Original loudness: {round(loudness, 1)} LUFS")
            print(f"Target loudness: {target_loudness} LUFS")
        output_audio = pyln.normalize.loudness(audio_data, loudness, target_loudness)
        return output_audio

    # Apply compressor to audio
    def audio_compressor(self, audio_data: np.ndarray, threshold_db: float = -6.0, ratio: float = 5.0):
        if self.verbose_log:
            print("\nRunning module: audio_compressor")
            print(f"Threshold dB: {threshold_db}")
            print(f"Compression ratio: {ratio}")
        if ratio <= 0.0:
            if self.verbose_log:
                print("Compression ratio must be a positive value. Skip.")
            return audio_data
        def gain_reduction(audio_db, threshold_db, ratio):
            if audio_db > threshold_db:
                return threshold_db + (audio_db - threshold_db) / ratio
            else:
                return audio_db
        audio_db = 20 * np.log10(np.abs(audio_data) + 1e-6)
        compressed_db = np.vectorize(gain_reduction)(audio_db, threshold_db, ratio)
        output_audio = np.sign(audio_data) * (10 ** (compressed_db / 20.0))
        return output_audio

    # Apply pitch shift to audio
    def audio_pitch(self, audio_data: np.ndarray, sampling_rate: int, pitch_semitone: float):
        if self.verbose_log:
            print("\nRunning module: audio_pitch")
            print(f"Pitch semitone: {pitch_semitone}")
        if pitch_semitone == 0.0:
            if self.verbose_log:
                print("Pitch semitone is zero. Skip.")
            return audio_data
        if len(audio_data.shape) > 1:
            output_data_t = [librosa.effects.pitch_shift(audio_data.T[i], sr=sampling_rate, n_steps=pitch_semitone) for i in range(audio_data.T.shape[0])]
            output_audio = np.array(output_data_t).T.astype(np.float32)
        else:
            output_audio = librosa.effects.pitch_shift(audio_data, sr=sampling_rate, n_steps=pitch_semitone)
        return output_audio

    # Apply time stretch to audio
    def audio_stretch(self, audio_data: np.ndarray, sampling_rate: int, speed_factor: float):
        if self.verbose_log:
            print("\nRunning module: audio_stretch")
            print(f"Speed factor: {speed_factor}")
        if speed_factor == 0.0:
            if self.verbose_log:
                print("Speed factor is zero. Skip.")
            return audio_data
        audio_int16 = self.float32_to_int16(audio_data)
        audio_stretch = AudioStretch()
        if len(audio_data.shape) > 1:
            audio_stretch.nchannels = audio_data.shape[1]
            audio_int16 = audio_int16.reshape(-1)
        else:
            audio_stretch.nchannels = 1
        audio_stretch.sampwidth = 2
        audio_stretch.framerate = sampling_rate
        audio_stretch.nframes = audio_data.shape[0]
        audio_stretch.in_samples = audio_int16
        audio_stretch.samples = audio_stretch.in_samples
        try:
            audio_stretch.stretch(ratio=1 / speed_factor)
            output_audio = self.int16_to_float32(audio_stretch.samples)
        except:
            if self.verbose_log:
                print("Failed in using audiostretchy. Try to use librosa.")
            if len(audio_data.shape) > 1:
                output_data_t = [librosa.effects.time_stretch(audio_data.T[i], rate=speed_factor) for i in range(audio_data.T.shape[0])]
                output_audio = np.array(output_data_t).T.astype(np.float32)
            else:
                output_audio = librosa.effects.time_stretch(audio_data, rate=speed_factor)
        return output_audio

    # Equalize source audio to target audio
    def eq_match(self, source_audio: np.ndarray, target_audio: Union[np.ndarray, str, dict], source_sampling_rate: int = 16000, target_sampling_rate: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        target_dict = None
        target_stft = None
        if isinstance(target_audio, str):
            if target_audio.endswith(".pkl"):
                with open(target_audio, "rb") as f:
                    target_dict = pickle.load(f)
            else:
                target_audio, target_sampling_rate = self.read_audio(file_path=target_audio)
        elif isinstance(target_audio, dict):
            target_dict = target_audio.copy()
        if target_dict is not None:
            target_audio = target_dict.get("array", None)
            target_stft = target_dict.get("stft", None)
            target_sampling_rate = target_dict.get("sampling_rate", 16000)
            n_fft = target_dict.get("n_fft", 2048)
            hop_length = target_dict.get("hop_length", 512)
        if hop_length > n_fft:
            hop_length = int(n_fft / 4)
        if self.verbose_log:
            print("\nRunning module: eq_match")
            print(f"FFT window size: {n_fft}")
            print(f"Hop length: {hop_length}")
        if source_sampling_rate < target_sampling_rate:
            source_audio, _ = self.audio_resample(audio_data=source_audio, orig_sr=source_sampling_rate, target_sr=target_sampling_rate)
        elif source_sampling_rate > target_sampling_rate:
            if target_audio is None and target_stft is not None:
                target_audio = librosa.istft(target_stft, hop_length=hop_length)
                target_stft = None
            target_audio, _ = self.audio_resample(audio_data=target_audio, orig_sr=target_sampling_rate, target_sr=source_sampling_rate)
        if target_stft is None:
            target_stft = librosa.stft(target_audio, n_fft=n_fft, hop_length=hop_length)
        source_stft = librosa.stft(source_audio, n_fft=n_fft, hop_length=hop_length)
        source_mag, source_phase = librosa.magphase(source_stft)
        target_mag, _ = librosa.magphase(target_stft)
        avg_source_mag = np.mean(source_mag, axis=1)
        avg_target_mag = np.mean(target_mag, axis=1)
        eq_filter = avg_target_mag / avg_source_mag
        eq_filter = np.clip(eq_filter, 0.1, 10)
        matched_mag = source_mag * eq_filter[:, np.newaxis]
        matched_stft = matched_mag * source_phase
        output_audio = librosa.istft(matched_stft, hop_length=hop_length)
        if source_sampling_rate < target_sampling_rate:
            output_audio, _ = self.audio_resample(audio_data=output_audio, orig_sr=target_sampling_rate, target_sr=source_sampling_rate)
        return output_audio

    # Resample audio to target sampling rate
    def audio_resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int, output_audio_only: bool = False):
        if self.verbose_log:
            print("\nRunning module: audio_resample")
            print(f"Original sampling rate: {orig_sr}")
            print(f"Target sampling rate: {target_sr}")
        if target_sr == orig_sr:
            if self.verbose_log:
                print("target_sr equals orig_sr. Skip.")
            if output_audio_only:
                return audio_data
            else:
                return audio_data, target_sr
        if len(audio_data.shape) > 1:
            output_data_t = [librosa.resample(audio_data.T[i], orig_sr=orig_sr, target_sr=target_sr) for i in range(audio_data.T.shape[0])]
            output_audio = np.array(output_data_t).T.astype(np.float32)
        else:
            output_audio = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        if output_audio_only:
            return output_audio
        else:
            return output_audio, target_sr

    # Add silence clip to audio
    def add_silence(self, audio_data: np.ndarray, sampling_rate: int, duration_sec: float = 1.0, add_to: str = "end"):
        if self.verbose_log:
            print("\nRunning module: add_silence")
            print(f"Silence duration: {duration_sec}")
            print(f"Add to: {add_to}")
        if duration_sec <= 0.0:
            if self.verbose_log:
                print("duration_sec must be a positive value. Skip.")
            return audio_data
        duration_nframes = int(sampling_rate * duration_sec)
        if len(audio_data.shape) > 1:
            silence_data = np.zeros((duration_nframes, audio_data.shape[1]), dtype=np.float32)
        else:
            silence_data = np.zeros(duration_nframes, dtype=np.float32)
        if add_to == "end":
            output_audio = np.concatenate([audio_data, silence_data])
        elif add_to == "begin":
            output_audio = np.concatenate([silence_data, audio_data])
        else:
            output_audio = audio_data
        return output_audio

    # Remove silence clip from audio
    def remove_silence(self, audio_data: np.ndarray, sampling_rate: int, silence_thresh_db: int = -30, min_silence_sec: float = 0.5, min_chunk_sec: float = 5.0):
        audio_data_list = self.split_audio_by_silence(audio_data=audio_data, sampling_rate=sampling_rate, silence_thresh_db=silence_thresh_db, min_silence_sec=min_silence_sec, min_chunk_sec=min_chunk_sec)
        output_audio = self.combine_audio_chunks(audio_data_list)
        return output_audio

    # Remove non-vocal noise from audio using MDX-Net or noisereduce
    def denoise_vocal(self, audio_data: np.ndarray, sampling_rate: int = 16000, fast_mode: bool = False):
        chunk_sec = 15.0
        margin_sec = 1.0
        use_io_binding = False
        def process_audio_chunk(chunk_data: np.ndarray):
            mix_data = chunk_data.T
            n_sample = mix_data.shape[1]
            mdx_net = self.mdx_net
            trim = mdx_net.n_fft // 2
            gen_size = mdx_net.chunk_size - 2 * trim
            pad = (gen_size - (n_sample % gen_size)) % gen_size
            mix_pad_trim = np.concatenate(
                (np.zeros((2, trim)), mix_data, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            current_frame = 0
            while current_frame < n_sample + pad:
                waves = np.array(mix_pad_trim[:, current_frame : current_frame + mdx_net.chunk_size])
                mix_waves.append(waves)
                current_frame = current_frame + gen_size
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)
            with torch.no_grad():
                mdx_model = self.mdx_model
                mix_spec = mdx_net.stft(mix_waves)
                if use_io_binding:
                    torch.cuda.synchronize()
                    io_binding, pred_spec_tensor = self.create_io_binding(mix_spec)
                    mdx_model.run_with_iobinding(io_binding)
                else:
                    pred_spec = mdx_model.run(None, {"input": mix_spec.cpu().numpy()})[0]
                    pred_spec_tensor = torch.tensor(pred_spec)
            pred_wav = mdx_net.istft(pred_spec_tensor)
            output = (
                pred_wav[:, :, trim:-trim]
                .transpose(0, 1)
                .reshape(2, -1)
                .numpy()[:, :-pad]
            ).T
            if "inst" in os.path.basename(self.mdx_weights_file).lower():
                vocals = np.clip(chunk_data - output, -1.0, 1.0)
            else:
                vocals = np.clip(output, -1.0, 1.0)
            return vocals
        
        if not self.is_denoise_vocal:
            fast_mode = True
        if self.verbose_log:
            print("\nRunning module: denoise_vocal")
            if fast_mode:
                print("Using method: noisereduce")
            else:
                print("Using method: MDX-Net")
                print(f"Using model: {self.mdx_weights_file}")
        if fast_mode:
            output_audio = nr.reduce_noise(y=audio_data, sr=sampling_rate)
            return output_audio
        if sampling_rate != 44100:
            audio_data, new_sampling_rate = self.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=44100)
        else:
            new_sampling_rate = sampling_rate
        if len(audio_data.shape) == 1:
            is_mono = True
            audio_data = self.mono_to_stereo(audio_data)
        else:
            is_mono = False
        total_samples = audio_data.shape[0]
        chunk_size = int(chunk_sec * new_sampling_rate)
        margin_size = int(margin_sec * new_sampling_rate)
        if total_samples > chunk_size:
            if margin_size > chunk_size:
                margin_size = chunk_size
            segmented_audio = {}
            current_segment = 0
            cursor = 0
            while cursor < total_samples:
                start_margin = 0 if current_segment == 0 else margin_size
                start = cursor - start_margin
                start = max(0, start)
                chunk_end = cursor + chunk_size
                is_last_segment = chunk_end >= total_samples
                if is_last_segment:
                    end = total_samples
                else:
                    end = min(chunk_end + margin_size, total_samples)
                segmented_audio[cursor] = audio_data[start:end].copy()
                current_segment = current_segment + 1
                cursor = cursor + chunk_size
                if is_last_segment:
                    break
            processed_segments = []
            segment_keys = list(segmented_audio.keys())
            for segment_idx, (segment_key, segment_data) in enumerate(segmented_audio.items()):
                processed_segment = process_audio_chunk(segment_data)
                is_first = segment_idx == 0
                is_last = segment_idx == len(segment_keys) - 1
                if is_first:
                    start_trim = 0
                else:
                    start_trim = min(margin_size, len(processed_segment) // 2)
                if is_last:
                    end_trim = None
                else:
                    end_trim = -min(margin_size, len(processed_segment) // 2)
                trimmed_segment = processed_segment[start_trim:end_trim]
                processed_segments.append(trimmed_segment)
            output_audio = np.concatenate(processed_segments, axis=0)
        else:
            output_audio = process_audio_chunk(audio_data)
        if is_mono:
            output_audio = self.audio_to_mono(audio_data=output_audio)
        if new_sampling_rate != sampling_rate:
            output_audio, sampling_rate = self.audio_resample(audio_data=output_audio, orig_sr=new_sampling_rate, target_sr=sampling_rate)
        return output_audio

    # Enhance vocal using Resemble enhancer
    def enhance_vocal(self, audio_data: np.ndarray, sampling_rate: int, method: Literal["enhance", "denoise", "both"] = "enhance", keep_sampling_rate: bool = False, output_audio_only: bool = False):
        if not self.is_enhance_vocal:
            print("\nSkip module: enhance_vocal")
            return audio_data
        if self.verbose_log:
            print("\nRunning module: enhance_vocal")
        input_tensor = self.ndarray_to_torchaudio(audio_data=audio_data)
        input_tensor = input_tensor.mean(dim=0)
        if method == "enhance":
            output_tensor, new_sr = enhancer_api(model=self.enhancer, dwav=input_tensor, sr=sampling_rate, device=self.device)
        elif method == "denoise":
            output_tensor, new_sr = enhancer_api(model=self.enhancer.denoiser, dwav=input_tensor, sr=sampling_rate, device=self.device)
        else:
            output_tensor, new_sr = enhancer_api(model=self.enhancer.denoiser, dwav=input_tensor, sr=sampling_rate, device=self.device)
            output_tensor, new_sr = enhancer_api(model=self.enhancer, dwav=output_tensor, sr=sampling_rate, device=self.device)
        output_audio = self.torchaudio_to_ndarray(output_tensor)
        if keep_sampling_rate:
            output_audio, new_sr = self.audio_resample(audio_data=output_audio, orig_sr=new_sr, target_sr=sampling_rate)
        if output_audio_only:
            return output_audio
        else:
            return output_audio, new_sr

    # Split audio by timerange
    def split_audio_by_time(self, audio_data: np.ndarray, sampling_rate: int, start_time: float, end_time: float):
        if self.verbose_log:
            print("\nRunning module: split_audio_by_time")
            print(f"Split duration: {round(end_time - start_time, 3)}")
        start_frame = max(0, int(start_time * sampling_rate))
        end_frame = min(int(end_time * sampling_rate), audio_data.shape[0])
        output_audio = audio_data.copy()[start_frame:end_frame]
        return output_audio

    # Split audio by silence
    def split_audio_by_silence(self, audio_data: np.ndarray, sampling_rate: int, silence_thresh_db: int = -30, min_silence_sec: float = 0.5, min_chunk_sec: float = 0.0):
        if self.verbose_log:
            print("\nRunning module: split_audio_by_silence")
        if len(audio_data.shape) > 1:
            num_channels = audio_data.shape[1]
        else:
            num_channels = 1
            audio_data = audio_data.reshape(-1, 1)
        audio_segments = []
        for channel in range(num_channels):
            channel_data = (audio_data[:, channel] * 32767).astype(np.int16)
            audio_segment = AudioSegment(channel_data.tobytes(), frame_rate=sampling_rate, sample_width=2, channels=1)
            audio_segments.append(audio_segment)
        split_channels = [split_on_silence(segment, min_silence_len=int(min_silence_sec * 1000), silence_thresh=silence_thresh_db, keep_silence=True) for segment in audio_segments]
        np_chunks = []
        for chunk_group in zip(*split_channels):
            np_chunks.append([np.frombuffer(chunk.raw_data, dtype=np.int16).astype(np.float32) / 32767 for chunk in chunk_group])
        if min_chunk_sec <= 0:
            return np_chunks
        combined_chunks = []
        temp_chunks = []
        temp_chunks_frame = 0
        for chunk_group in np_chunks:
            temp_chunks.append(chunk_group)
            temp_chunks_frame += len(chunk_group[0])
            if temp_chunks_frame >= int(min_chunk_sec * sampling_rate):
                combined_chunk = np.stack([np.concatenate([chunk[i] for chunk in temp_chunks]) for i in range(num_channels)], axis=1)
                combined_chunks.append(combined_chunk)
                temp_chunks = []
                temp_chunks_frame = 0
        if temp_chunks:
            combined_chunk = np.stack([np.concatenate([chunk[i] for chunk in temp_chunks]) for i in range(num_channels)], axis=1)
            combined_chunks.append(combined_chunk)
        return combined_chunks

    # Generate singal (pink, brown, silence)
    def generate_noise(self, sampling_rate: int = 16000, duration_sec: float = 1.0, gain_db: float = 0.0, noise_type: str = "brown"):
        num_samples = int(duration_sec * sampling_rate)
        noise = np.random.normal(0, 1, num_samples)
        if noise_type == "pink":
            freqs = np.fft.rfftfreq(num_samples, d=1.0 / sampling_rate)
            white_noise_fft = np.fft.rfft(noise)
            pink_filter = 1 / np.sqrt(freqs[1:])
            pink_filter = np.concatenate(([1], pink_filter))
            pink_noise_fft = white_noise_fft * pink_filter
            pink_noise = np.fft.irfft(pink_noise_fft, n=num_samples)
            noise = pink_noise / np.max(np.abs(pink_noise))
        elif noise_type == "brown":
            brown_noise = np.cumsum(noise)
            noise = brown_noise / np.max(np.abs(brown_noise))
        elif noise_type == "silence":
            duration_nframes = int(duration_sec * sampling_rate)
            noise = np.zeros(duration_nframes)
        noise = noise.astype(np.float32)
        if gain_db != 0.0:
            noise = self.audio_gain(audio_data=noise, gain_db=gain_db)
        return noise

    # Combine separated audio channels
    def mix_audio(self, audio_data_list: list, combine_channels: bool = True, normalize: bool = True):
        if self.verbose_log:
            print("\nRunning module: mix_audio")
        if isinstance(audio_data_list, np.ndarray):
            print("Input should be a np.ndarray audio list.")
            return audio_data_list
        if len(audio_data_list) == 1:
            print("Length of audio_data_list should be greater than 2.")
            return audio_data_list[0]
        for i in range(len(audio_data_list)):
            if len(audio_data_list[i].shape) > 1:
                audio_data_list[i] = self.int16_to_float32(audio_data=audio_data_list[i])
                audio_data_list[i] = self.audio_to_mono(audio_data=audio_data_list[i])
        max_length = 0
        for audio_data in audio_data_list:
            if audio_data.shape[0] > max_length:
                max_length = audio_data.shape[0]
        audio_list = []
        for audio_data in audio_data_list:
            if audio_data.shape[0] < max_length:
                silence_audio = np.zeros((max_length - audio_data.shape[0]), dtype=np.float32)
                audio_data = np.concatenate((audio_data, silence_audio))
            audio_list.append(audio_data)
        if combine_channels:
            output_audio = np.zeros((max_length), dtype=np.float32)
            for audio_data in audio_list:
                output_audio = output_audio + audio_data
            # output_audio = output_audio / len(audio_list)
        else:
            output_audio = np.stack(audio_list, axis=1)
        output_audio = output_audio.astype(np.float32)
        if normalize:
            output_audio = self.audio_normalize(audio_data=output_audio)
        return output_audio

    # Mix audio by frequency
    def mix_audio_by_freq(self, audio_main: np.ndarray, audio_aux: np.ndarray, sampling_rate: int = 16000, main_freq_range: Union[list, None] = [0, 4000], aux_freq_range: Union[list, None] = [0, 8000], force_align: bool = False):
        if self.verbose_log:
            print("\nRunning module: mix_audio_by_freq")
        if audio_main.shape[0] != audio_aux.shape[0]:
            if force_align:
                if audio_main.shape[0] < audio_aux.shape[0]:
                    audio_aux = audio_aux[:audio_main.shape[0]]
                else:
                    silence_audio = np.zeros((audio_main.shape[0] - audio_aux.shape[0]), dtype=np.float32)
                    audio_main = np.concatenate((audio_main, silence_audio))
            else:
                print("audio_main and audio_aux should have same lengths with same sampling rates.")
                return audio_main
        if not main_freq_range:
            main_freq_range = [0, int(sampling_rate / 4)]
        if not aux_freq_range:
            aux_freq_range = [0, int(sampling_rate / 2)]
        main_freq_range = [max(0, main_freq_range[0]), min(main_freq_range[1], int(sampling_rate / 2))]
        aux_freq_range = [max(0, aux_freq_range[0]), min(aux_freq_range[1], int(sampling_rate / 2))]
        if self.verbose_log:
            print(f"Frequency selected from audio_main: {main_freq_range}")
            print(f"Frequency selected from audio_aux: {aux_freq_range}")
        fft_main = np.fft.rfft(audio_main)
        fft_aux = np.fft.rfft(audio_aux)
        freqs = np.fft.rfftfreq(len(audio_main), 1 / sampling_rate)
        fft_mix = np.zeros_like(fft_main)
        main_freq_indices = (freqs >= main_freq_range[0]) & (freqs < main_freq_range[1])
        fft_mix[main_freq_indices] = fft_main[main_freq_indices]
        aux_freq_indices = (freqs >= aux_freq_range[0]) & (freqs <= aux_freq_range[1])
        fft_mix[aux_freq_indices] = fft_aux[aux_freq_indices]
        overlap_indices = (freqs >= max(main_freq_range[0], aux_freq_range[0])) & (freqs <= min(main_freq_range[1], aux_freq_range[1]))
        if np.any(overlap_indices):
            overlap_freqs = freqs[overlap_indices]
            overlap_weights_main = np.linspace(1, 0, len(overlap_freqs))
            overlap_weights_aux = 1 - overlap_weights_main
            fft_mix[overlap_indices] = fft_main[overlap_indices] * overlap_weights_main + fft_aux[overlap_indices] * overlap_weights_aux
        output_audio = np.fft.irfft(fft_mix)
        return output_audio

    # Separate 2 speakers from overlapping audio
    def separate_speaker(self, audio_data: np.ndarray, sampling_rate: int = 16000, low_gpu_ram: bool = False):
        if not self.is_separate_audio:
            print("\nSkip module: separate_speaker")
            return audio_data, audio_data
        orig_sr = sampling_rate
        if sampling_rate != 16000:
            audio_data, sampling_rate = self.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
        if low_gpu_ram:
            window_size = 16000
            is_vad = True
        else:
            window_size = 160000
            is_vad = False
        if self.verbose_log:
            print("\nRunning module: separate_speaker")
            print(f"Window size: {window_size}")
            print(f"Use VAD: {is_vad}")
        if is_vad:
            audio_tensor = self.ndarray_to_torchaudio(audio_data=audio_data)
            res = get_speech_timestamps(audio=audio_tensor, model=self.silero_vad, threshold=0.5, min_silence_duration_ms=100)
            vad_frames = [[clip['start'], clip['end']] for clip in res]
        else:
            vad_frames = [[0, audio_data.shape[0]]]
        audio_data_spk1 = np.array([], dtype=np.float32)
        audio_data_spk2 = np.array([], dtype=np.float32)
        for i, vad_frame in enumerate(vad_frames):
            if vad_frame[0] > audio_data_spk1.shape[0]:
                if i == 0:
                    middle_clip = np.zeros(vad_frame[0], dtype=np.float32)
                    # middle_clip = audio_data[0:vad_frame[0]].copy()
                else:
                    middle_clip = np.zeros(vad_frame[0] - vad_frames[i - 1][1], dtype=np.float32)
                    # middle_clip = audio_data[vad_frames[i - 1][1]:vad_frame[0]].copy()
                audio_data_spk1 = np.concatenate([audio_data_spk1, middle_clip])
                audio_data_spk2 = np.concatenate([audio_data_spk2, middle_clip])
            start_frames = []
            end_frames = []
            round_num = (vad_frame[1] - vad_frame[0]) // window_size
            if round_num == 0:
                start_frames.append(vad_frame[0])
                end_frames.append(vad_frame[1])
            else:
                for j in range(round_num):
                    start_frames.append(vad_frame[0] + j * window_size)
                    end_frames.append(vad_frame[0] + (j + 1) * window_size)
                if (vad_frame[1] - vad_frame[0]) % window_size > 0:
                    if (vad_frame[1] - vad_frame[0]) % window_size > window_size / 2:
                        start_frames.append(end_frames[-1])
                        end_frames.append(vad_frame[1])
                    else:
                        end_frames[-1] = vad_frame[1]
            for i in range(len(start_frames)):
                start_frame = start_frames[i]
                end_frame = end_frames[i]
                audio_data_split = audio_data[start_frame: end_frame].copy()
                audio_data_tensor = self.ndarray_to_torchaudio(audio_data_split)
                audio_data_tensor = audio_data_tensor.to(self.device)
                with torch.no_grad():
                    output = self.separater(audio_data_tensor)
                if output.dim() == 3:
                    output = output.squeeze(0)
                audio_data_outputs = output.cpu().detach().numpy()
                audio_data_spk1 = np.concatenate([audio_data_spk1, audio_data_outputs[0]])
                audio_data_spk2 = np.concatenate([audio_data_spk2, audio_data_outputs[1]])
        loudness_spk1 = self.meter_loudness(audio_data=audio_data_spk1, sampling_rate=sampling_rate)
        loudness_spk2 = self.meter_loudness(audio_data=audio_data_spk2, sampling_rate=sampling_rate)
        if loudness_spk1 < loudness_spk2:
            audio_data_spk1, audio_data_spk2 = audio_data_spk2, audio_data_spk1
        if orig_sr != sampling_rate:
            audio_data_spk1, _ = self.audio_resample(audio_data=audio_data_spk1, orig_sr=sampling_rate, target_sr=orig_sr)
            audio_data_spk2, _ = self.audio_resample(audio_data=audio_data_spk2, orig_sr=sampling_rate, target_sr=orig_sr)
        return audio_data_spk1, audio_data_spk2

    # Restore audio using Apollo
    def restore_audio(self, audio_data: np.ndarray, sampling_rate: int, keep_sampling_rate: bool = False, output_audio_only: bool = False):
        if not self.is_restore_audio:
            print("\nSkip module: restore_audio")
            return audio_data
        if self.verbose_log:
            print("\nRunning module: restore_audio")
        orig_sr = sampling_rate
        audio_data, sampling_rate = self.audio_resample(audio_data=audio_data, orig_sr=orig_sr, target_sr=44100)
        audio_data_tensor = self.ndarray_to_torchaudio(audio_data=audio_data, device=self.device)
        audio_data_tensor = audio_data_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.restorer(audio_data_tensor)
        if output.dim() == 3:
            output = output.squeeze(0).squeeze(0)
        elif output.dim() == 2:
            output = output.squeeze(0)
        output_audio = output.cpu().detach().numpy()
        if keep_sampling_rate:
            output_audio, sampling_rate = self.audio_resample(audio_data=output_audio, orig_sr=44100, target_sr=orig_sr)
        if output_audio_only:
            return output_audio
        return output_audio, sampling_rate

    # Write audio data from ndarray to file
    def write_to_file(self, output_path: str, audio_data: np.ndarray, sampling_rate: int, audio_format: str = "", bit_depth: int = 16):
        if not audio_format:
            audio_format = output_path.split(".")[-1]
        if len(audio_data.shape) == 1:
            channel_num = 1
        else:
            channel_num = audio_data.shape[1]
        wav_bit_dict = {
            8: "PCM_S8",
            16: "PCM_16",
            24: "PCM_24",
            32: "PCM_32"
        }
        if audio_format == "wav":
            subtype = wav_bit_dict[bit_depth]
        elif audio_format == "mp3":
            subtype = "MPEG_LAYER_III"
        elif audio_format == "ogg":
            subtype = "VORBIS"
        elif audio_format == "opus":
            subtype = "OPUS"
        else:
            subtype = "PCM_16"
        if self.verbose_log:
            print("\nRunning module: write_to_file")
            print(f"Sample rate: {sampling_rate}")
            print(f"Channel num: {channel_num}")
            print(f"Output format: {audio_format} | {subtype}")
        ogg_chunk_size = 102400
        if audio_format == "ogg" and len(audio_data) >= ogg_chunk_size:
            with sf.SoundFile(file=output_path, mode="w", channels=channel_num, samplerate=sampling_rate, subtype=subtype, format=audio_format.upper(), closefd=True) as f:
                num_chunks = (len(audio_data) + ogg_chunk_size - 1) // ogg_chunk_size
                for chunk in np.array_split(audio_data, num_chunks, axis=0):
                    f.write(chunk)
        else:
            sf.write(file=output_path, data=audio_data, samplerate=sampling_rate, subtype=subtype, format=audio_format.upper())
        if self.verbose_log:
            print(f"File saved: {output_path}")

    # Write multi ndarray to folder
    def write_to_folder(self, output_folder: str, audio_data_list: list, sampling_rate: int, audio_format: str = "wav", bit_depth: int = 16, output_name: str = ""):
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        for i, audio_data in enumerate(audio_data_list):
            if not output_name:
                output_name = output_folder.replace("\\", "/").split("/")[-1]
            output_path = f"{output_folder}/{output_name}_{i}.{audio_format}"
            self.write_to_file(output_path=output_path, audio_data=audio_data, sampling_rate=sampling_rate, audio_format=audio_format, bit_depth=bit_depth)
            if self.verbose_log:
                print(f"File saved: {output_path}")

    # Convert ndarray from float32 to int16
    def float32_to_int16(self, audio_data: np.ndarray, force_convert: bool = False):
        if np.issubdtype(audio_data.dtype, np.integer) and not force_convert:
            output_audio = audio_data.astype(np.int16)
        else:
            output_audio = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
        return output_audio

    # Convert ndarray from int16 to float32
    def int16_to_float32(self, audio_data: np.ndarray, force_convert: bool = False):
        if np.issubdtype(audio_data.dtype, np.floating) and not force_convert:
            output_audio = audio_data.astype(np.float32)
        else:
            output_audio = np.clip(audio_data / 32768.0, -1.0, 1.0).astype(np.float32)
        return output_audio

    # Convert ndarray to torchaudio tensor
    def ndarray_to_torchaudio(self, audio_data: np.ndarray, device=None):
        if len(audio_data.shape) > 1:
            output_tensor = torch.tensor(audio_data.T, device=device)
        else:
            output_tensor = torch.tensor(audio_data.reshape(1, -1), device=device)
        return output_tensor

    # Convert torchaudio tensor to ndarray
    def torchaudio_to_ndarray(self, audio_tensor):
        audio_data = audio_tensor.cpu().numpy()
        if len(audio_tensor.shape) > 1:
            output_audio = audio_data[1].reshape(-1, audio_tensor.shape[0])
        else:
            output_audio = audio_data
        return output_audio

    # Convert ndarray to bytes
    def ndarray_to_raw_bytes(self, audio_data: np.ndarray):
        output_audio = audio_data.tobytes()
        return output_audio

    # Convert bytes to ndarray
    def raw_bytes_to_ndarray(self, audio_bytes: bytes, dtype=np.float32):
        output_audio = np.frombuffer(buffer=audio_bytes, dtype=dtype)
        return output_audio

    # Convert ndarray to wav PCM bytes
    def ndarray_to_pcm_bytes(self, audio_data: np.ndarray, sampling_rate: int):
        with io.BytesIO() as audio_buffer:
            sf.write(file=audio_buffer, data=audio_data, samplerate=sampling_rate, subtype="PCM_32", format="WAV")
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()
        return audio_bytes

    # Convert ndarray to file bytes
    def ndarray_to_file_bytes(self, audio_data: np.ndarray, sampling_rate: int, audio_format: str = "", bit_depth: int = 16):
        wav_bit_dict = {
            8: "PCM_S8",
            16: "PCM_16",
            24: "PCM_24",
            32: "PCM_32"
        }
        if audio_format == "wav":
            subtype = wav_bit_dict[bit_depth]
        elif audio_format == "mp3":
            subtype = "MPEG_LAYER_III"
        elif audio_format == "ogg":
            subtype = "VORBIS"
        elif audio_format == "opus":
            subtype = "OPUS"
        else:
            subtype = "PCM_16"
        with io.BytesIO() as audio_buffer:
            sf.write(audio_buffer, audio_data, sampling_rate, format=audio_format, subtype=subtype)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()
        return audio_bytes

    # Seperate channels from audio
    def seperate_channels(self, audio_data: np.ndarray, channel_num: int):
        reshaped_audio = audio_data.reshape(-1, channel_num)
        output_audio = np.array([reshaped_audio[:, i] for i in range(channel_num)]).astype(np.float32)
        return output_audio

    # Combine audio chunks
    def combine_audio_chunks(self, audio_data_list: list):
        if len(audio_data_list) == 1:
            return audio_data_list[0]
        output_audio = np.concatenate(audio_data_list)
        return output_audio
    
    # Meter loudness in LUFS
    def meter_loudness(self, audio_data: np.ndarray, sampling_rate: int):
        meter = pyln.Meter(sampling_rate)
        loudness = meter.integrated_loudness(audio_data)
        loudness = round(loudness, 1)
        return loudness
    
    # eq_match utility function
    def create_eq_match_pickle(self, ir_audio_path: str, output_path: str = ""):
        n_fft = 2048
        hop_length = 512
        if not output_path:
            output_path = f"{os.path.splitext(ir_audio_path)[0]}.pkl"
        audio_data, sampling_rate = self.read_audio(file_path=ir_audio_path)
        target_stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        result = {
            "array": audio_data,
            "stft": target_stft,
            "sampling_rate": sampling_rate,
            "n_fft": n_fft,
            "hop_length": hop_length
        }
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        return result
    
    # ONNX I/O binding utility function
    def create_io_binding(self, input_tensor: torch.Tensor):
        device_type = "cuda" if "cuda" in self.device else "cpu"
        device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
        io_binding = self.mdx_model.io_binding()
        io_binding.bind_input(
            name=self.mdx_model.get_inputs()[0].name,
            device_type=device_type,
            device_id=device_id,
            element_type=np.float32,
            shape=input_tensor.shape,
            buffer_ptr=input_tensor.data_ptr()
        )
        output_shape = input_tensor.shape
        output_tensor = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        io_binding.bind_output(
            name=self.mdx_model.get_outputs()[0].name,
            device_type=device_type,
            device_id=device_id,
            element_type=np.float32,
            shape=output_tensor.shape,
            buffer_ptr=output_tensor.data_ptr()
        )
        return io_binding, output_tensor
