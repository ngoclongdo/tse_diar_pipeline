# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2025-9-24

import hdbscan
import io
import numpy as np
import os
import shutil
import sys
import torch
# import umap
from silero_vad import load_silero_vad, get_speech_timestamps
from modelscope import pipeline
from typing import Union, Literal

file_dir = str(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
sys.path.append(file_dir)
ap_dir = os.path.abspath(f"{file_dir}/../AudioProcessor")
if os.path.isdir(ap_dir):
    sys.path.append(ap_dir)
from AudioProcessor import AudioProcessor
from ASRProcessor import ASRProcessor


# Main class
class TargetASR:
    def __init__(self, cuda_device: int = 0, embedding_model_dir: Union[str, list] = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common", vad_model_dir: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", diarization_model_dir: Union[str, None] = None, asr_model_dir: Union[str, list, None] = "checkpoints/whisper-large-v2-finetune-v8", mdx_weights_file: Union[str, None] = None, separater_weights_folder: Union[str, None] = None, restorer_weights_folder: Union[str, None] = None, verbose_log: bool = False):
        self.cuda_device = cuda_device
        self.asr_model_dir = asr_model_dir
        self.embedding_model_dir = embedding_model_dir
        self.vad_model_dir = vad_model_dir
        self.diarization_model_dir = diarization_model_dir
        self.mdx_weights_file = mdx_weights_file
        self.separater_weights_folder = separater_weights_folder
        self.restorer_weights_folder = restorer_weights_folder
        self.verbose_log = verbose_log
        self.get_device()
        self.load_model()

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
        if self.mdx_weights_file:
            is_denoise_vocal = True
        else:
            is_denoise_vocal = False
            self.mdx_weights_file = ""
            print("mdx_weights_file is not given. Skip denoise_vocal.")
        if self.separater_weights_folder:
            is_separate_audio = True
        else:
            is_separate_audio = False
            self.separater_weights_folder = ""
            print("separater_weights_folder is not given. Skip separate_audio.")
        if self.restorer_weights_folder:
            is_restore_audio = True
        else:
            is_restore_audio = False
            self.restorer_weights_folder = ""
            print("restorer_weights_folder is not given. Skip restore_audio.")
        self.ap = AudioProcessor(is_denoise_vocal=is_denoise_vocal, mdx_weights_file=self.mdx_weights_file, is_separate_audio=is_separate_audio, separater_weights_folder=self.separater_weights_folder, is_restore_audio=is_restore_audio, restorer_weights_folder=self.restorer_weights_folder, verbose_log=False, cuda_device=self.cuda_device, quality=3)
        if not self.asr_model_dir:
            print("asr_model_dir is not given. Skip ASR.")
            is_asr = False
            is_punc = False
        elif "paraformer" in self.asr_model_dir:
            print("Activate punctuation restore model to fit Paraformer ASR engine.")
            is_asr = True
            is_punc = True
        else:
            is_asr = True
            is_punc = False
        if not self.diarization_model_dir:
            print("diarization_model_dir is not given. Skip speaker_diarization.")
            is_diarization = False
        else:
            is_diarization = True
        self.asrp = ASRProcessor(is_asr=is_asr, asr_model_dir=self.asr_model_dir, is_asr_api=True, is_punc=is_punc, is_vad=True, vad_model_dir=self.vad_model_dir, is_diarization=is_diarization, diarization_model_dir=self.diarization_model_dir, cuda_device=self.cuda_device, ap=self.ap, verbose_log=False)
        if isinstance(self.embedding_model_dir, str):
            self.embedding_model_dir = [self.embedding_model_dir]
        self.embedding = {}
        for embedding_model in self.embedding_model_dir:
            if "eres2netv2w24s4ep4_sv" in embedding_model:
                self.embedding['eres2netv2_large'] = pipeline(task="speaker-verification", model=embedding_model, device=self.device)
            if "eres2netv2_sv" in embedding_model:
                self.embedding['eres2netv2'] = pipeline(task="speaker-verification", model=embedding_model, device=self.device)
            if "eres2net_sv" in embedding_model:
                self.embedding['eres2net'] = pipeline(task="speaker-verification", model=embedding_model, device=self.device)
            if "campp" in embedding_model:
                self.embedding['campp'] = pipeline(task="speaker-verification", model=embedding_model, device=self.device)
        self.silero_vad = load_silero_vad()

    # Pyannote VAD + ASR
    def pyannote_asr(self, asr_audio: str, target_audio: str = ""):
        asr_audio_data, asr_audio_sr = self.ap.read_audio(file_path=asr_audio)
        asr_audio_data, asr_audio_sr = self.ap.audio_resample(audio_data=asr_audio_data, orig_sr=asr_audio_sr, target_sr=16000)
        if target_audio:
            target_audio_data, target_audio_sr = self.ap.read_audio(file_path=target_audio)
            target_audio_data, target_audio_sr = self.ap.audio_resample(audio_data=target_audio_data, orig_sr=target_audio_sr, target_sr=16000)
            combine_audio_data = self.ap.combine_audio_chunks(audio_data_list=[target_audio_data, asr_audio_data])
        else:
            combine_audio_data = asr_audio_data
        diarization_result = self.asrp.speaker_diarization(wav_file=combine_audio_data, sampling_rate=16000, clustering_threshold=1.0)
        target_diarization = diarization_result[next(iter(diarization_result))]
        if target_audio:
            target_diarization.pop(0)
            for i in range(len(target_diarization)):
                new_start_time = max(0.0, round(target_diarization[i][0] - target_audio_data.shape[0] / 16000, 3))
                new_end_time = max(0.0, round(target_diarization[i][1] - target_audio_data.shape[0] / 16000, 3))
                target_diarization[i] = [new_start_time, new_end_time]
        result = []
        for timerange in target_diarization:
            clip_audio_data = self.ap.split_audio_by_time(audio_data=asr_audio_data, sampling_rate=16000, start_time=timerange[0], end_time=timerange[1])
            if not self.asr_model_dir:
                asr_text = ""
            else:
                asr_text = self.asrp.asr_detection(wav_file=clip_audio_data, asr_engine="paraformer", output_text_only=True, no_punc=True)
            result.append({
                "timerange": timerange,
                "text": asr_text
            })
        return result

    # Compute cosine similarity between 2 speaker embeddings
    def cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray):
        if np.all(embedding_a == 0.0) or np.all(embedding_b == 0.0):
            return 1.0
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        similarity = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
        similarity = max(0.0, min(similarity, 1.0))
        similarity = float(similarity)
        return similarity
    
    # Audio to speaker embedding
    def get_speaker_embedding(self, wav_file: Union[str, list, np.ndarray], embedding_model: Literal["eres2netv2_large", "eres2netv2", "eres2net", "campp"] = "eres2netv2_large"):
        if isinstance(wav_file, np.ndarray):
            wav_file = wav_file.reshape(1, -1)
        elif isinstance(wav_file, str):
            wav_file = [wav_file]
        with torch.no_grad():
            result = self.embedding[embedding_model](wav_file, output_emb=True)
        embedding = result['embs'].reshape(-1)
        return embedding

    # Audio to speaker embedding with some processing
    def get_target_embedding(self, target_audio: Union[str, list, np.ndarray], is_preprocess: bool = True, is_cluster: bool = True, embedding_model: Literal["eres2netv2_large", "eres2netv2", "eres2net", "campp"] = "eres2netv2_large", audio_input_type: Literal["auto", "separate", "merge", "longest"] = "separate", output_embedding_list: bool = True):
        audio_data_list = []
        sampling_rate = 16000
        if isinstance(target_audio, str):
            target_audio = [target_audio]
        if isinstance(target_audio, list):
            for audio_path in target_audio:
                audio_data, orig_sampling_rate = self.ap.read_audio(file_path=audio_path)
                audio_data, _ = self.ap.audio_resample(audio_data=audio_data, orig_sr=orig_sampling_rate, target_sr=sampling_rate)
                audio_data_list.append(audio_data)
        else:
            audio_data_list = [target_audio.copy()]
        if self.verbose_log:
            print(f"Length of audio_data_list before preprocess: {len(audio_data_list)}")
        if is_preprocess:
            preprocess_result = []
            for audio_data in audio_data_list:
                # audio_data = self.ap.denoise_vocal(audio_data=audio_data)
                vad_result = self.asrp.vad_detection(wav_file=audio_data)
                # audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=audio_data)
                # res = get_speech_timestamps(audio=audio_tensor, model=self.silero_vad, min_silence_duration_ms=0, return_seconds=True)
                # vad_result = [[clip['start'], clip['end']] for clip in res]
                if not vad_result:
                    print("Failed in func get_target_embedding: No VAD result.")
                    continue
                audio_data_vad = []
                for vad in vad_result:
                    audio_data_clip = self.ap.split_audio_by_time(audio_data=audio_data, sampling_rate=sampling_rate, start_time=vad[0], end_time=vad[1])
                    audio_data_vad.append(audio_data_clip)
                if audio_data_vad:
                    audio_data = self.ap.combine_audio_chunks(audio_data_list=audio_data_vad)
                audio_data = self.ap.audio_loudness_control(audio_data=audio_data, sampling_rate=sampling_rate)
                preprocess_result.append(audio_data)
            if self.verbose_log:
                print(f"Length of preprocess_result: {len(preprocess_result)}")
            audio_data_list = preprocess_result.copy()
        if not audio_data_list:
            print("Length of embedding_list shouldn't be zero. Return a 192-dim full-zero embedding.")
            target_embedding = np.zeros([192], dtype=np.float32)
            return target_embedding
        longest_audio = max(audio_data_list, key=lambda x: x.shape[0])
        normal_len_audios = [audio_data for audio_data in audio_data_list if audio_data.shape[0] >= int(sampling_rate * 0.4)]
        merged_audio = self.ap.combine_audio_chunks(audio_data_list=audio_data_list)
        if audio_input_type == "auto":
            if longest_audio.shape[0] >= 3.0 * sampling_rate:
                audio_input_type = "longest"
            elif len(normal_len_audios) <= 2:
                audio_input_type = "merge"
            else:
                audio_input_type = "separate"
        if audio_input_type == "merge":
            audio_data_list = [merged_audio]
        elif audio_input_type == "longest":
            audio_data_list = [longest_audio]
        else:
            audio_data_list = normal_len_audios
        for i in range(len(audio_data_list)):
            if audio_data_list[i].shape[0] > 30 * sampling_rate:
                audio_data_list[i] = audio_data_list[i][:30 * sampling_rate]
        if self.verbose_log:
            print(f"Length of audio_data_list after selected: {len(audio_data_list)}")
        embedding_list = []
        for audio_data in audio_data_list:
            if audio_data.shape[0] < 400:
                continue
            embedding = self.get_speaker_embedding(wav_file=audio_data, embedding_model=embedding_model)
            if np.isnan(embedding).any():
                print("NaN value in embedding. Skip.")
                continue
            embedding_list.append(embedding)
        if self.verbose_log:
            print(f"Length of embedding_list before clustering: {len(embedding_list)}")
        if is_cluster and len(embedding_list) > 2:
            embeddings = np.stack(embedding_list)
            # reducer = umap.UMAP(n_components=50, n_neighbors=15, metric="cosine", random_state=616)
            # embeddings_50dim = reducer.fit_transform(embeddings)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
            labels = clusterer.fit_predict(embeddings)
            valid_indexs = np.where(labels != -1)[0]
            if len(valid_indexs) > 0:
                embedding_list = [embedding_list[i] for i in valid_indexs]
            if self.verbose_log:
                print(f"Length of embedding_list after clustering: {len(embedding_list)}")
        if output_embedding_list:
            return embedding_list
        if len(embedding_list) == 0:
            print("Length of embedding_list shouldn't be zero. Return a 192-dim full-zero embedding.")
            target_embedding = np.zeros([192], dtype=np.float32)
        elif len(embedding_list) == 1:
            target_embedding = embedding_list[0]
        else:
            target_embedding = np.mean(embedding_list, axis=0)
        return target_embedding
    
    # Audio preprocess: Read, to mono, resample
    def input_audio_preprocess(self, audio: Union[str, io.BytesIO, np.ndarray]):
        if isinstance(audio, np.ndarray):
            # print("asr_audio is a np.ndarray object, assume sampling_rate=16000")
            audio_data = audio
            sampling_rate = 16000
        else:
            audio_data, sampling_rate = self.ap.read_audio(file_path=audio)
        if len(audio_data.shape) > 1:
            audio_data = self.ap.audio_to_mono(audio_data=audio_data)
        audio_data, sampling_rate = self.ap.audio_resample(audio_data=audio_data, orig_sr=sampling_rate, target_sr=16000)
        return audio_data, sampling_rate

    # Main method: ASR target speaker with time domain separation
    def target_speaker_asr(self, asr_audio: Union[str, io.BytesIO, np.ndarray], target_audio: Union[str, list, np.ndarray, None] = None, target_embedding: Union[np.ndarray, list, None] = None, threshold: float = 0.4, audio_input_type: Literal["separate", "merge"] = "merge", is_output_audio: bool = False, more_args: dict = {}):
        more_args_dict = {
            "vad_silence_threshold": 0.0,
            "vad_model": "funasr",
            "embedding_model": "eres2netv2_large",
            "asr_engine": "whisper_finetune",
            "preprocess": [],
            "prompt": ""
        }
        if more_args:
            more_args_dict.update(more_args)
        output_audio = None
        asr_audio_data, sampling_rate = self.input_audio_preprocess(audio=asr_audio)
        if "vocal_denoise" in more_args_dict['preprocess'] and self.mdx_weights_file:
            asr_audio_data = self.ap.denoise_vocal(audio_data=asr_audio_data)
        if "loudness_control" in more_args_dict['preprocess']:
            asr_audio_data = self.ap.audio_loudness_control(audio_data=asr_audio_data, sampling_rate=sampling_rate)
        if more_args_dict['vad_model'] == "silero_vad":
            asr_audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=asr_audio_data)
            res = get_speech_timestamps(audio=asr_audio_tensor, model=self.silero_vad, min_silence_duration_ms=int(more_args_dict['vad_silence_threshold'] * 1000), return_seconds=True)
            vad_result = [[clip['start'], clip['end']] for clip in res]
        else:
            vad_result = self.asrp.vad_detection(wav_file=asr_audio_data, min_silence_sec=more_args_dict['vad_silence_threshold'])
        if not vad_result:
            return []
        if target_embedding is None:
            if not target_audio:
                target_audio_data = self.ap.split_audio_by_time(audio_data=asr_audio_data, sampling_rate=sampling_rate, start_time=vad_result[0][0], end_time=vad_result[0][1])
            else:
                target_audio_data, _ = self.input_audio_preprocess(audio=target_audio)
            target_embedding = self.get_target_embedding(target_audio=target_audio_data, is_preprocess=True if "vocal_denoise" in more_args_dict['preprocess'] else False, embedding_model=more_args_dict['embedding_model'])
        if not self.asr_model_dir and audio_input_type == "merge":
            audio_input_type = "separate"
        result = []
        merge_audio_data = None
        merge_timeranges = []
        for i, timerange in enumerate(vad_result):
            clip_audio_data = self.ap.split_audio_by_time(audio_data=asr_audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
            if clip_audio_data.shape[0] < int(0.1 * sampling_rate):
                continue
            # if "vocal_denoise" in more_args_dict['preprocess']:
            #     clip_audio_data = self.ap.denoise_vocal(audio_data=clip_audio_data)
            if "loudness_control" in more_args_dict['preprocess']:
                clip_audio_data = self.ap.audio_loudness_control(audio_data=clip_audio_data, sampling_rate=sampling_rate)
            clip_embedding = self.get_speaker_embedding(wav_file=clip_audio_data, embedding_model=more_args_dict['embedding_model'])
            if np.isnan(clip_embedding).any():
                print("clip_embedding has NaN value. Skip.")
                continue
            if isinstance(target_embedding, list):
                similarity_list = []
                for embedding in target_embedding:
                    similarity = self.cosine_similarity(embedding_a=clip_embedding, embedding_b=embedding)
                    similarity_list.append(similarity)
                if similarity_list:
                    clip_target_similarity = max(similarity_list)
                else:
                    clip_target_similarity = 1.0
            else:
                clip_target_similarity = self.cosine_similarity(embedding_a=clip_embedding, embedding_b=target_embedding)
            if audio_input_type == "separate":
                if clip_target_similarity >= threshold:
                    if not self.asr_model_dir:
                        asr_text = ""
                    else:
                        asr_text = self.asrp.asr_detection(wav_file=clip_audio_data, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=True)
                    if is_output_audio:
                        output_audio = clip_audio_data
                    else:
                        output_audio = np.array([], dtype=np.float32)
                    result.append({
                        "timerange": timerange,
                        "text": asr_text,
                        "score": round(clip_target_similarity, 2),
                        "sampling_rate": sampling_rate,
                        "audio": output_audio
                    })
            else:
                if clip_target_similarity >= threshold:
                    if merge_audio_data is None:
                        merge_audio_data = clip_audio_data.copy()
                    else:
                        merge_audio_data = self.ap.combine_audio_chunks([merge_audio_data, clip_audio_data])
                    merge_timeranges.append(timerange)
        if merge_audio_data is not None and merge_timeranges:
            silence_clip = self.ap.generate_noise(sampling_rate=16000, duration_sec=0.5, noise_type="silence")
            merge_audio_data = self.ap.combine_audio_chunks([merge_audio_data, silence_clip])
            if not self.asr_model_dir:
                asr_text = ""
            else:
                asr_text = self.asrp.asr_detection(wav_file=merge_audio_data, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=True)
            merge_embedding = self.get_speaker_embedding(wav_file=merge_audio_data, embedding_model=more_args_dict['embedding_model'])
            if isinstance(target_embedding, list):
                similarity_list = []
                for embedding in target_embedding:
                    similarity = self.cosine_similarity(embedding_a=merge_embedding, embedding_b=embedding)
                    similarity_list.append(similarity)
                if similarity_list:
                    merge_target_similarity = max(similarity_list)
                else:
                    merge_target_similarity = 1.0
            else:
                merge_target_similarity = self.cosine_similarity(embedding_a=merge_embedding, embedding_b=target_embedding)
            if is_output_audio:
                output_audio = merge_audio_data
            result.append({
                "timerange": [merge_timeranges[0][0], merge_timeranges[-1][1]],
                "text": asr_text,
                "score": round(merge_target_similarity, 2),
                "sampling_rate": sampling_rate,
                "audio": output_audio
            })
        return result

    # Target ASR batch processing
    def batch_target_speaker_asr(self, asr_audio_list: list, target_audio_list: Union[list, str] = [], prompt_list: list = [], threshold: float = 0.4, more_args: dict = {}):
        if isinstance(target_audio_list, str):
            target_audio_list = [target_audio_list]
        results = []
        target_audio_temp = []
        for target_audio in target_audio_list:
            if target_audio.startswith("http"):
                temp_audio = self.ap.download_audio(url=target_audio)
            else:
                temp_audio = shutil.copy(target_audio, f"/tmp/{os.path.basename(target_audio)}")
            target_audio_temp.append(temp_audio)
        target_embedding = self.get_target_embedding(target_audio=target_audio_temp)
        for temp_audio in target_audio_temp:
            if os.path.isfile(temp_audio):
                os.remove(temp_audio)
        for i, asr_audio in enumerate(asr_audio_list):
            if self.verbose_log:
                print(f"Processing audio: {asr_audio}")
            if asr_audio.startswith("http"):
                asr_audio_local = self.ap.download_audio(url=asr_audio)
            else:
                asr_audio_local = asr_audio
            if len(prompt_list) == len(asr_audio_list):
                more_args_temp = more_args.update({
                    "prompt": prompt_list[i]
                })
            else:
                more_args_temp = more_args.copy()
            result = self.target_speaker_asr(asr_audio=asr_audio_local, target_embedding=target_embedding, threshold=threshold, more_args=more_args_temp)
            if asr_audio.startswith("http"):
                if os.path.isfile(asr_audio_local):
                    os.remove(asr_audio_local)
            results.append(result)
        text_results = []
        for result in results:
            result_text = ""
            for r in result:
                if r and "text" in r.keys():
                    result_text = f"{result_text} {r['text']}"
            result_text = result_text.strip()
            text_results.append(result_text)
        return text_results

    # Target speaker diarization (simple version)
    def target_speaker_duration(self, input_audio: Union[str, io.BytesIO], target_embedding: Union[np.ndarray, list, None] = None, threshold: float = 0.4, more_args: dict = {}):
        more_args_dict = {
            "vad_silence_threshold": 0.0,
            "vad_model": "funasr",
            "embedding_model": "eres2netv2_large",
            "preprocess": [],
        }
        if more_args:
            more_args_dict.update(more_args)
        result = {
            "target_duration": [],
            "others_duration": []
        }
        input_audio_data, sampling_rate = self.ap.read_audio(file_path=input_audio)
        if len(input_audio_data.shape) > 1:
            input_audio_data = self.ap.audio_to_mono(audio_data=input_audio_data)
        input_audio_data, sampling_rate = self.ap.audio_resample(audio_data=input_audio_data, orig_sr=sampling_rate, target_sr=16000)
        if "vocal_denoise" in more_args_dict['preprocess']:
            input_audio_data = self.ap.denoise_vocal(audio_data=input_audio_data)
        if "loudness_control" in more_args_dict['preprocess']:
            input_audio_data = self.ap.audio_loudness_control(audio_data=input_audio_data, sampling_rate=sampling_rate)
        if more_args_dict['vad_model'] == "silero_vad":
            input_audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=input_audio_data)
            res = get_speech_timestamps(audio=input_audio_tensor, model=self.silero_vad, min_silence_duration_ms=int(more_args_dict['vad_silence_threshold'] * 1000), return_seconds=True)
            vad_result = [[clip['start'], clip['end']] for clip in res]
        else:
            vad_result = self.asrp.vad_detection(wav_file=input_audio_data, min_silence_sec=more_args_dict['vad_silence_threshold'])
        if not vad_result:
            return result
        for i, timerange in enumerate(vad_result):
            clip_audio_data = self.ap.split_audio_by_time(audio_data=input_audio_data, sampling_rate=sampling_rate, start_time=timerange[0], end_time=timerange[1])
            if clip_audio_data.shape[0] < int(0.1 * sampling_rate):
                continue
            # if "vocal_denoise" in more_args_dict['preprocess']:
            #     clip_audio_data = self.ap.denoise_vocal(audio_data=clip_audio_data)
            if "loudness_control" in more_args_dict['preprocess']:
                clip_audio_data = self.ap.audio_loudness_control(audio_data=clip_audio_data, sampling_rate=sampling_rate)
            clip_embedding = self.get_speaker_embedding(wav_file=clip_audio_data, embedding_model=more_args_dict['embedding_model'])
            if np.isnan(clip_embedding).any():
                print("clip_embedding has NaN value. Skip.")
                continue
            if isinstance(target_embedding, list):
                similarity_list = []
                for embedding in target_embedding:
                    similarity = self.cosine_similarity(embedding_a=clip_embedding, embedding_b=embedding)
                    similarity_list.append(similarity)
                if similarity_list:
                    clip_target_similarity = max(similarity_list)
                else:
                    clip_target_similarity = 1.0
            else:
                clip_target_similarity = self.cosine_similarity(embedding_a=clip_embedding, embedding_b=target_embedding)
            if threshold <= clip_target_similarity < 1.0:
                result['target_duration'].append(timerange)
            elif clip_target_similarity < threshold:
                result['others_duration'].append(timerange)
        return result

    # Check if the target speaker is the same person
    def is_same_person(self, existed_embeddings: Union[list, np.ndarray], target_embedding: np.ndarray, threshold: float = 0.4, verbose_result: bool = False):
        if isinstance(existed_embeddings, np.ndarray):
            existed_embeddings = [existed_embeddings]
        existed_embeddings_mean = np.mean(existed_embeddings, axis=0)
        similarity = self.cosine_similarity(embedding_a=existed_embeddings_mean, embedding_b=target_embedding)
        if similarity >= threshold:
            if verbose_result:
                return {"is_same": True, "score": round(similarity, 3)}
            else:
                return True
        else:
            if verbose_result:
                return {"is_same": False, "score": round(similarity, 3)}
            else:
                return False
    
    # Main method: ASR target speaker with frequency domain separation
    def target_speaker_separate_asr(self, asr_audio: Union[str, io.BytesIO, np.ndarray], target_audio: Union[str, io.BytesIO, np.ndarray, None] = None, target_embedding: Union[np.ndarray, None] = None, threshold: float = 0.4, is_output_asr: bool = True, is_output_audio: bool = True, more_args: dict = {}):
        result = []
        more_args_dict = {
            "vad_silence_threshold": 0.0,
            "vad_model": "funasr",
            "embedding_model": "eres2netv2_large",
            "asr_engine": "whisper_finetune",
            "preprocess": [],
            "prompt": ""
        }
        if more_args:
            more_args_dict.update(more_args)
        asr_audio_data, sampling_rate = self.input_audio_preprocess(audio=asr_audio)
        if "vocal_denoise" in more_args_dict['preprocess'] and self.mdx_weights_file:
            asr_audio_data = self.ap.denoise_vocal(audio_data=asr_audio_data)
        if "loudness_control" in more_args_dict['preprocess']:
            asr_audio_data = self.ap.audio_loudness_control(audio_data=asr_audio_data, sampling_rate=sampling_rate)
            if more_args_dict['vad_model'] == "silero_vad":
                asr_audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=asr_audio_data)
                res = get_speech_timestamps(audio=asr_audio_tensor, model=self.silero_vad, min_silence_duration_ms=int(more_args_dict['vad_silence_threshold'] * 1000), return_seconds=True)
                vad_result = [[clip['start'], clip['end']] for clip in res]
            elif more_args_dict['vad_model'] == "funasr":
                vad_result = self.asrp.vad_detection(wav_file=asr_audio_data, min_silence_sec=more_args_dict['vad_silence_threshold'])
            else:
                vad_result = [[0.0, round(asr_audio_data.shape[0] / sampling_rate, 3)]]
        if not vad_result:
            return result
        if target_embedding is None:
            if not target_audio:
                target_audio_data = self.ap.split_audio_by_time(audio_data=asr_audio_data, sampling_rate=sampling_rate, start_time=vad_result[0][0], end_time=vad_result[0][1])
            else:
                target_audio_data, _ = self.input_audio_preprocess(audio=target_audio)
            target_embedding = self.get_speaker_embedding(wav_file=target_audio_data, embedding_model=more_args_dict['embedding_model'])
        spk1_audio, spk2_audio = self.ap.separate_speaker(audio_data=asr_audio_data)
        spk1_embedding = self.get_speaker_embedding(wav_file=spk1_audio, embedding_model=more_args_dict['embedding_model'])
        spk2_embedding = self.get_speaker_embedding(wav_file=spk2_audio, embedding_model=more_args_dict['embedding_model'])
        spk1_score = self.cosine_similarity(embedding_a=spk1_embedding, embedding_b=target_embedding)
        spk2_score = self.cosine_similarity(embedding_a=spk2_embedding, embedding_b=target_embedding)
        if spk1_score < threshold and spk2_score < threshold:
            return result
        if spk1_score > spk2_score:
            score = spk1_score
            output_audio = spk1_audio
        else:
            score = spk2_score
            output_audio = spk2_audio
        if self.restorer_weights_folder:
            output_audio = self.ap.restore_audio(audio_data=output_audio, sampling_rate=sampling_rate, keep_sampling_rate=True, output_audio_only=True)
        asr_text = ""
        if is_output_asr:
            asr_text = self.asrp.asr_detection(wav_file=output_audio, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=True)
        if not is_output_audio:
            output_audio = np.array([], dtype=np.float32)
        result.append({
            "timerange": [vad_result[0][0], vad_result[-1][1]],
            "text": asr_text,
            "score": round(score, 2),
            "sampling_rate": sampling_rate,
            "audio": output_audio
        })
        return result
    
    # ASR target and non-target speakers with frequency domain separation
    def multi_speakers_separate_asr(self, asr_audio: Union[str, io.BytesIO, np.ndarray], target_audio: Union[str, io.BytesIO, np.ndarray, None] = None, target_embedding: Union[np.ndarray, None] = None, threshold: float = 0.4, is_output_asr: bool = True, is_output_audio: bool = True, more_args: dict = {}):
        def audio_vad(audio_data: np.ndarray):
            if more_args_dict['vad_model'] == "silero_vad":
                asr_audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=audio_data)
                res = get_speech_timestamps(audio=asr_audio_tensor, model=self.silero_vad, min_silence_duration_ms=int(more_args_dict['vad_silence_threshold'] * 1000), return_seconds=True)
                vad_result = [[clip['start'], clip['end']] for clip in res]
            elif more_args_dict['vad_model'] == "funasr":
                vad_result = self.asrp.vad_detection(wav_file=audio_data, min_silence_sec=more_args_dict['vad_silence_threshold'])
            else:
                vad_result = [[0.0, round(audio_data.shape[0] / sampling_rate, 3)]]
            return vad_result
        
        result = []
        more_args_dict = {
            "vad_silence_threshold": 0.0,
            "vad_model": "funasr",
            "embedding_model": "eres2netv2_large",
            "asr_engine": "whisper_finetune",
            "preprocess": [],
            "prompt": "",
            "no_punc": True
        }
        if more_args:
            more_args_dict.update(more_args)
        asr_audio_data, sampling_rate = self.input_audio_preprocess(audio=asr_audio)
        if "vocal_denoise" in more_args_dict['preprocess'] and self.mdx_weights_file:
            asr_audio_data = self.ap.denoise_vocal(audio_data=asr_audio_data)
        if "loudness_control" in more_args_dict['preprocess']:
            asr_audio_data = self.ap.audio_loudness_control(audio_data=asr_audio_data, sampling_rate=sampling_rate)
        vad_result = audio_vad(audio_data=asr_audio_data)
        if not vad_result:
            return result
        if target_embedding is None:
            if not target_audio:
                target_audio_data = self.ap.split_audio_by_time(audio_data=asr_audio_data, sampling_rate=sampling_rate, start_time=vad_result[0][0], end_time=vad_result[0][1])
            else:
                target_audio_data, _ = self.input_audio_preprocess(audio=target_audio)
            target_embedding = self.get_speaker_embedding(wav_file=target_audio_data, embedding_model=more_args_dict['embedding_model'])
        spk1_audio, spk2_audio = self.ap.separate_speaker(audio_data=asr_audio_data)
        spk1_embedding = self.get_speaker_embedding(wav_file=spk1_audio, embedding_model=more_args_dict['embedding_model'])
        spk2_embedding = self.get_speaker_embedding(wav_file=spk2_audio, embedding_model=more_args_dict['embedding_model'])
        spk1_score = self.cosine_similarity(embedding_a=spk1_embedding, embedding_b=target_embedding)
        spk2_score = self.cosine_similarity(embedding_a=spk2_embedding, embedding_b=target_embedding)
        if spk1_score < threshold and spk2_score < threshold:
            return result
        if spk1_score > spk2_score:
            target_score = spk1_score
            target_output_audio = spk1_audio
            noise_score = spk2_score
            noise_output_audio = spk2_audio
        else:
            target_score = spk2_score
            target_output_audio = spk2_audio
            noise_score = spk1_score
            noise_output_audio = spk1_audio
        if self.restorer_weights_folder:
            target_output_audio = self.ap.restore_audio(audio_data=target_output_audio, sampling_rate=sampling_rate, keep_sampling_rate=True, output_audio_only=True)
            noise_output_audio = self.ap.restore_audio(audio_data=noise_output_audio, sampling_rate=sampling_rate, keep_sampling_rate=True, output_audio_only=True)
        target_asr_text = ""
        noise_asr_text = ""
        if is_output_asr:
            target_asr_text = self.asrp.asr_detection(wav_file=target_output_audio, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=more_args_dict['no_punc'])
            noise_asr_text = self.asrp.asr_detection(wav_file=noise_output_audio, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=more_args_dict['no_punc'])
        target_vad_result = audio_vad(audio_data=target_output_audio)
        noise_vad_result = audio_vad(audio_data=noise_output_audio)
        if not is_output_audio:
            target_output_audio = np.array([], dtype=np.float32)
            noise_output_audio = np.array([], dtype=np.float32)
        if target_vad_result:
            result.append({
                "timerange": [target_vad_result[0][0], target_vad_result[-1][1]],
                "text": target_asr_text,
                "score": round(target_score, 2),
                "sampling_rate": sampling_rate,
                "audio": target_output_audio
            })
        if noise_vad_result:
            result.append({
                "timerange": [noise_vad_result[0][0], noise_vad_result[-1][1]],
                "text": noise_asr_text,
                "score": round(noise_score, 2),
                "sampling_rate": sampling_rate,
                "audio": noise_output_audio
            })
        return result
    
    # Simple ASR with formatted result
    def single_speaker_asr(self, asr_audio: Union[str, io.BytesIO, np.ndarray], is_output_audio: bool = False, more_args: dict = {}):
        result = []
        more_args_dict = {
            "asr_engine": "whisper_finetune",
            "prompt": "",
            "no_punc": True,
            "preprocess": []
        }
        if more_args:
            more_args_dict.update(more_args)
        output_audio = None
        asr_audio_data, sampling_rate = self.input_audio_preprocess(audio=asr_audio)
        if "vocal_denoise" in more_args_dict['preprocess'] and self.mdx_weights_file:
            asr_audio_data = self.ap.denoise_vocal(audio_data=asr_audio_data)
        if "loudness_control" in more_args_dict['preprocess']:
            asr_audio_data = self.ap.audio_loudness_control(audio_data=asr_audio_data, sampling_rate=sampling_rate)
        asr_text = self.asrp.asr_detection(wav_file=asr_audio_data, asr_engine=more_args_dict['asr_engine'], prompt=more_args_dict['prompt'], output_text_only=True, no_punc=more_args_dict['no_punc'])
        if is_output_audio:
            output_audio = asr_audio_data
        else:
            output_audio = np.array([], dtype=np.float32)
        result.append({
            "timerange": [0.0, round(asr_audio_data.shape[0] / sampling_rate, 2)],
            "text": asr_text,
            "score": 1.0,
            "sampling_rate": sampling_rate,
            "audio": output_audio
        })
        return result
    
    def mix_audio_processor(self, audio: Union[str, io.BytesIO, np.ndarray], target_embedding: Union[np.ndarray, None] = None, similarity_threshold: float = 0.4, loudness_threshold: float = -40.0):
        audio_data, sampling_rate = self.input_audio_preprocess(audio=audio)
        result = {
            "audio": audio_data,
            "sampling_rate": sampling_rate,
            "type": "noise",
            "score": 0.0
        }
        audio_duration = round(audio_data.shape[0] / sampling_rate, 3)
        if audio_duration >= 0.4:
            loudness = self.ap.meter_loudness(audio_data=audio_data, sampling_rate=sampling_rate)
            if loudness <= loudness_threshold:
                return result
            audio_data = self.ap.denoise_vocal(audio_data=audio_data, sampling_rate=sampling_rate)
            audio_data = self.ap.audio_loudness_control(audio_data=audio_data, sampling_rate=sampling_rate)
        pyannote_result = []
        if self.asrp.is_diarization:
            pyannote_result = self.asrp.speaker_diarization(wav_file=audio_data, sampling_rate=sampling_rate)
        speaker_type = "single"
        if not pyannote_result:
            # audio_tensor = self.ap.ndarray_to_torchaudio(audio_data=audio_data)
            # res = get_speech_timestamps(audio=audio_tensor, model=self.silero_vad, return_seconds=True)
            # vad_result = [[clip['start'], clip['end']] for clip in res]
            vad_result = self.asrp.vad_detection(wav_file=audio_data)
            if not vad_result:
                speaker_type = "noise"
        elif len(pyannote_result) == 1:
            speaker_type = "single"
        elif len(pyannote_result) > 1:
            speaker_type = "multi"
        result['type'] = speaker_type
        if speaker_type == "noise":
            result['audio'] = np.full(audio.shape[0], fill_value=0.00001, dtype=np.float32)
            return result
        elif speaker_type == "single":
            result['audio'] = audio_data
            result['score'] = 1.0
            return result
        if target_embedding is None:
            result['audio'] = audio_data
            result['score'] = 0.0
            return result
        spk1_audio, spk2_audio = self.ap.separate_speaker(audio_data=audio_data)
        spk1_embedding = self.get_speaker_embedding(wav_file=spk1_audio, embedding_model="eres2netv2_large")
        spk2_embedding = self.get_speaker_embedding(wav_file=spk2_audio, embedding_model="eres2netv2_large")
        spk1_score = self.cosine_similarity(embedding_a=spk1_embedding, embedding_b=target_embedding)
        spk2_score = self.cosine_similarity(embedding_a=spk2_embedding, embedding_b=target_embedding)
        result['score'] = round(max(spk1_score, spk2_score), 3)
        if spk1_score < similarity_threshold and spk2_score < similarity_threshold:
            result['audio'] = audio_data
        elif spk1_score >= spk2_score:
            result['audio'] = spk1_audio
        elif spk2_score > spk1_score:
            result['audio'] = spk2_audio
        else:
            result['audio'] = audio_data
        return result
