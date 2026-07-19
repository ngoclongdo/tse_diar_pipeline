import os
import sys
import argparse
from datasets import load_dataset

# Thêm thư mục gốc vào đường dẫn hệ thống để import thư viện TargetDiarization
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from TargetDiarization import TargetDiarization
except ImportError as e:
    print(f"Lỗi Import: {e}. Vui lòng đảm bảo script này nằm trong thư mục 'experiment'.")
    sys.exit(1)

def load_test_data(dataset_name):
    print(f"[INFO] Đang tải bộ dữ liệu {dataset_name} từ HuggingFace...")
    if dataset_name == "viyt":
        dataset = load_dataset("tuanduy1612/ViYT-Diar", split="test")
    elif dataset_name == "voxconverse":
        dataset = load_dataset("openslr/voxconverse", split="test") 
    elif dataset_name == "minilibrimix":
        dataset = load_dataset("JorisCos/MiniLibriMix", "mix_both", split="test")
    else:
        raise ValueError("Tập dữ liệu không được hỗ trợ.")
    print(f"[INFO] Tải thành công {len(dataset)} mẫu.")
    return dataset

def print_results(results):
    print(">> Kết quả hệ thống trả về:")
    for res in results:
        start, end = res['timerange'][0], res['timerange'][1]
        print(f"  [{start:05.2f}s - {end:05.2f}s] Speaker {res['speaker']}: {res['text']}")

def run_tier_evaluation(dataset, tier, num_samples):
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU ĐÁNH GIÁ - TIER {tier} (Lan truyền lỗi)")
    print(f"{'='*60}")
    
    # Setup môi trường tuỳ theo Cấp độ (Tier)
    if tier < 4:
        print("[Config] Đang ép TẮT module Denoising (UVR) và Phục hồi (Apollo) để cô lập lỗi...")
        os.environ["RESTORER_WEIGHTS_FOLDER"] = ""
        os.environ["MDX_WEIGHTS_FILE"] = "" 
    else:
        print("[Config] Chế độ Full End-to-End: Bật toàn bộ các module có sẵn.")

    print("\nKhởi tạo Pipeline TargetDiarization...")
    cuda_device = int(os.environ.get("CUDA_DEVICE", 0))
    td_pipeline = TargetDiarization(cuda_device=cuda_device)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        audio_info = sample.get("audio", {})
        if "array" in audio_info:
            audio_data = audio_info["array"]
            sr = audio_info.get("sampling_rate", 16000)
            file_name = audio_info.get("path", f"sample_{i}.wav")
        else:
            audio_data = sample.get("audio_path") or sample.get("file")
            sr = 16000
            file_name = os.path.basename(audio_data) if audio_data else "unknown"

        if audio_data is None: 
            continue
            
        print(f"\n--- Đang xử lý mẫu {i+1}/{num_samples}: {file_name} ---")
        
        try:
            if tier == 0:
                print(">> [Tier 0 - Absolute Oracle]: Đang trích xuất Ground Truth Audio -> Chạy ASR trực tiếp.")
                print(">> (Đang phát triển module parser cắt âm thanh chuẩn theo Ground Truth...)")
                
            elif tier == 1:
                print(">> [Tier 1 - Separation Eval]: Bỏ qua Diarization -> Ép Ground Truth Overlap -> MossFormer2 -> ASR.")
                print(">> (Đang phát triển module nạp Ground Truth vào MossFormer2...)")
                
            elif tier == 2:
                print(">> [Tier 2 - Diarization Eval]: Bỏ qua VAD -> Ép Ground Truth VAD -> CAM++/Pyannote -> MossFormer2 -> ASR.")
                print(">> (Đang phát triển module nạp Ground Truth vào Pyannote...)")
                
            elif tier == 3:
                print(">> [Tier 3 - VAD Eval]: FSMN VAD -> CAM++/Pyannote -> MossFormer2 -> ASR (TẮT Denoise).")
                target_spk, results, target_audio = td_pipeline.infer(wav_file=audio_data, sampling_rate=sr, target_file=None)
                print_results(results)
                
            elif tier == 4:
                print(">> [Tier 4 - Full E2E]: UVR -> VAD -> Diarization -> Separation -> Apollo -> ASR.")
                target_spk, results, target_audio = td_pipeline.infer(wav_file=audio_data, sampling_rate=sr, target_file=None)
                print_results(results)
                
        except Exception as e:
            print(f"[LỖI] Xử lý thất bại: {e}")

def main():
    parser = argparse.ArgumentParser(description="TargetDiarization Cascading Evaluator")
    parser.add_argument("--dataset", type=str, choices=["viyt", "voxconverse", "minilibrimix"], required=True)
    parser.add_argument("--tier", type=int, choices=[0, 1, 2, 3, 4], default=4, 
                        help="Cấp độ bóc tách: 0 (Oracle ASR), 1 (Sep), 2 (Diar), 3 (VAD), 4 (Full E2E)")
    parser.add_argument("--samples", type=int, default=3, help="Số lượng file audio chạy test")
    
    args = parser.parse_args()
    dataset = load_test_data(args.dataset)
    run_tier_evaluation(dataset, args.tier, args.samples)

if __name__ == "__main__":
    main()
