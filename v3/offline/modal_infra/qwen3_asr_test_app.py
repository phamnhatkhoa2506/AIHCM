"""TAM THOI (2026-08-10) - app Modal RIENG chi de so sanh truc tiep Qwen3-ASR-0.6B voi
PhoWhisper (asr_app.py) tren cung 1 doan audio that, quyet dinh dung ban nao truoc khi build
build_asr_index.py chinh thuc theo huong nay. XOA sau khi co ket luan.

Chay thu (dev): modal run qwen3_asr_test_app.py -- --audio-path <file.wav>
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch", "transformers", "accelerate", "qwen-asr", "soundfile")
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)

app = modal.App("aic2026-qwen3-asr-test")


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class Qwen3ASRTest:
    @modal.enter()
    def load(self):
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B", cache_dir="/root/.cache/huggingface"
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import io

        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes))
        result = self.model.transcribe((data, sr), language="Vietnamese")
        return str(result)


@app.local_entrypoint()
def test():
    import sys

    path = sys.argv[sys.argv.index("--audio-path") + 1] if "--audio-path" in sys.argv else sys.argv[1]
    with open(path, "rb") as f:
        audio_bytes = f.read()
    m = Qwen3ASRTest()
    print(m.transcribe.remote(audio_bytes))
