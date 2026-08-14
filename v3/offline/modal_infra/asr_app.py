"""Modal app cho ASR — transcript tiếng nói (bản tin, phóng viên, MC) khai thác cho:
  A. Đưa transcript làm ngữ cảnh bổ sung cho VQA (submission_pipeline.py::_vqa_answer) —
     nhiều câu hỏi Q&A hỏi về thứ chỉ tồn tại trong lời nói (số liệu đọc lên, tên được nhắc),
     không suy ra được từ ảnh.
  B. Tên riêng/thuật ngữ được NÓI ra — bổ sung entity detection, object detector 514 nhãn
     OpenImages không phân biệt được tên riêng (người/địa danh/sự kiện cụ thể).
  C. Tín hiệu củng cố chéo với OCR (cùng lúc nói + hiện chữ giống nhau).
(xem hội thoại 2026-08-10 — quyết định KHÔNG làm audio-query-by-example, BTC không nhận audio
làm input query).

Model: PhoWhisper (VinAI) — fine-tune riêng cho tiếng Việt từ Whisper, chính xác hơn nhiều so
với Whisper gốc cho domain bản tin tiếng Việt (tên riêng/địa danh Việt) — xem quyết định lúc
bàn kiến trúc ASR. Dùng bản "small" (2026-08-11, quay lại từ "medium" sau khi đo A10G+medium
mất ~6.6 giờ ước tính cho full corpus — "small" ưu tiên tốc độ, batch nhiều audio lợi ích quá
nhỏ ~5% nên không đáng làm thêm, xem hội thoại).

Input: audio bytes (WAV mono 16kHz) — driver (build_asr_index.py) trích SẴN bằng ffmpeg local
trước khi gửi lên, không gửi nguyên video (nặng hơn nhiều).

Chạy thử (dev, hot-reload): modal serve asr_app.py
Deploy thật:                modal deploy asr_app.py
"""
import modal

MODEL_NAME = "vinai/PhoWhisper-small"
MAX_CONTAINERS = 8  # gioi han tai khoan Modal da xac nhan (xem grounding_dino_app.py)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # transformers pipeline ASR can ffmpeg de decode audio input
    .pip_install("torch", "transformers", "accelerate", "numpy<2", "librosa", "soundfile")
)

WINDOW_S = 28.0  # do dai moi doan TU CHIA (giay) - xem giai thich o transcribe()

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung

app = modal.App("aic2026-asr")


@app.cls(
    image=image,
    gpu="A10G",  # 2026-08-10: doi tu T4 sang A10G de do tocdo that (nghi T4 la nut that cho
    # PhoWhisper-medium, xem "LAN 1 COLD-START 94.7s / LAN 2 WARM 38.3s cho 60s audio" da do).
    max_containers=MAX_CONTAINERS,
    # 2026-08-11: tang tu 5 phut -> 20 phut - nghi ngo dung (nguoi dung quan sat qua Modal
    # dashboard: nhieu container cold-start lai lien tuc) - PHA 1 (trich audio CPU cho batch
    # tiep theo, xem build_asr_index.py) co the mat >5 phut, container GPU bi tat trong luc
    # cho roi phai cold-start lai moi batch, ton oan ~45s/container/batch.
    scaledown_window=20 * 60,
    timeout=3600,  # video dai nhat corpus 45.6 phut (2734s) * ~0.64 he so compute -> can
    # timeout du dai, KHONG con dung mac dinh 300s (da crash that o test truoc).
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class Transcriber:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import pipeline

        # BUG THAT (2026-08-10): truyen cache_dir= THANG vao pipeline() bi forward nham vao
        # model.generate() luc inference ("model_kwargs are not used by the model: ['cache_dir']")
        # - dung HF_HOME (env var) giong het pattern resources.py/vocab_discovery.py da dung,
        # KHONG truyen cache_dir vao constructor.
        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # BUG THAT NGHIEM TRONG (2026-08-11): return_timestamps=True + chunk_length_s=30 dua
        # vao token-level timestamp ma MODEL tu sinh ra luc generate() - da thu bo batch_size=8
        # (nghi ngo ban dau) nhung test lai tren container HOAN TOAN MOI van bi (None, None).
        # Sau do test rieng 1 audio 20s NGAN (khong can chia doan long-form) van ra
        # timestamp=(0.02, None) - CHUNG TO day KHONG PHAI loi batch_size hay loi merge nhieu
        # doan, ma la ban fine-tune "vinai/PhoWhisper-small" KHONG sinh timestamp token dong
        # doan (end) dang tin cay - co le fine-tune khong giu lai kha nang nay tu Whisper goc.
        # -> BO HOAN TOAN co che timestamp cua model, TU CHIA audio thanh cac doan WINDOW_S
        # giay co dinh (transcribe() ben duoi), gan start/end THEO CHINH BIEN CHIA CUA MINH
        # thay vi tin model doan. Da chay NHAM toan bo 873 video voi loi nay - phai chay LAI het.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=self.device,
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> list[dict]:
        """Tra ve list[{"text","start","end"}] (giay, tinh tu dau audio) - moi phan tu la 1
        doan WINDOW_S giay TU CHIA (KHONG dua vao timestamp token cua model - xem ghi chu o
        load(), model nay khong sinh timestamp dong doan dang tin cay)."""
        import io

        import librosa
        import numpy as np

        # decode bang librosa/soundfile (qua ffmpeg noi bo) thanh mang float32 16kHz mono -
        # tu quan ly viec chia doan thay vi de pipeline tu chia (chunk_length_s) va tin
        # timestamp token cua no.
        audio_arr, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        total_s = len(audio_arr) / sr
        window_samples = int(WINDOW_S * sr)

        out = []
        n_windows = max(1, int(np.ceil(len(audio_arr) / window_samples)))
        for i in range(n_windows):
            lo = i * window_samples
            hi = min(lo + window_samples, len(audio_arr))
            if hi <= lo:
                continue
            segment = audio_arr[lo:hi]
            result = self.pipe(segment)
            text = result["text"].strip()
            if not text:
                continue
            out.append({
                "text": text,
                "start": lo / sr,
                "end": min(hi / sr, total_s),
            })
        return out

@app.local_entrypoint()
def test():
    """modal run asr_app.py -- --audio-path <file.wav> — test nhanh voi 1 file audio that."""
    import sys

    if len(sys.argv) < 2:
        print("Dung: modal run asr_app.py -- --audio-path <file.wav>")
        return
    path = sys.argv[sys.argv.index("--audio-path") + 1] if "--audio-path" in sys.argv else sys.argv[1]
    with open(path, "rb") as f:
        audio_bytes = f.read()

    t = Transcriber()
    results = t.transcribe.remote(audio_bytes)
    for r in results:
        print(f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['text']}")
