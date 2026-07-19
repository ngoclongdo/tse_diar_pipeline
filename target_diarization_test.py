import time
from TargetDiarization import TargetDiarization
from TargetDiarizationStream import TargetDiarizationStream

# Conversation audio file
wav_file = "assets/chat_mix.wav"
# Target speaker audio sample
target_file = "assets/female_a.wav"


def simulate_audio_stream(audio_file, chunk_duration=1.0):
    audio_data, sr = td.ap.read_audio(audio_file)
    audio_data = td.ap.audio_to_mono(audio_data)
    audio_data, sr = td.ap.audio_resample(audio_data, sr, 16000)
    
    chunk_samples = int(chunk_duration * sr)
    total_samples = len(audio_data)
    
    for i in range(0, total_samples, chunk_samples):
        chunk = audio_data[i:i+chunk_samples]
        yield chunk
        time.sleep(chunk_duration * 0.5)


if __name__ == "__main__":
    print("===== Testing non-streaming inference =====")
    td = TargetDiarization()
    t1 = time.time()
    target_spk, result, target_audio = td.infer(
        wav_file=wav_file, 
        target_file=target_file,
        sampling_rate=16000,
        is_single=False,
        output_target_audio=True
    )
    t2 = time.time()
    print("Target speaker:", target_spk)
    print("Diarization result:", result)
    print("Target audio (np.array):", target_audio)
    print(f"Used time: {round(t2 - t1, 3)}s")
    
    print("\n===== Testing streaming inference =====")
    td = TargetDiarizationStream()
    stream_gen = simulate_audio_stream(wav_file)
    for result in td.infer_stream(
        stream_gen,
        target_file=target_file,
        sampling_rate=16000,
        is_single=False,
        output_target_audio=False
    ):
        print("Chunk diarization result:", result)
