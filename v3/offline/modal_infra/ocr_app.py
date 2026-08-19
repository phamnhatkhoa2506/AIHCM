"""Modal app cho OCR (chữ trên màn hình — lower-third bản tin, watermark kênh, biển hiệu/màn
hình trong khung hình) — chốt hướng 2026-08-10: toạ độ trực tiếp (PaddleOCR tự detect vùng chữ
DBNet + nhận diện CRNN/SVTR nội bộ), KHÔNG qua object detection trước (khác Grounding DINO —
xem lý do trong hội thoại: chữ không phải "object", DBNet quét toàn khung hình trực tiếp).

Dùng PaddleOCR (không phải VLM) vì OCR cần độ chính xác ký tự tuyệt đối (tên riêng/số liệu),
không cần suy luận ngữ nghĩa — VLM (Qwen2.5-VL) rẻ hơn về hạ tầng có sẵn nhưng rủi ro
hallucinate chữ/số, không chấp nhận được cho việc này.

Pattern giống hệt grounding_dino_app.py (@app.cls + @modal.enter + @modal.method, không cần
HTTP server) — nhưng PaddleOCR (API .ocr() cổ điển, bản 2.7.x) KHÔNG hỗ trợ batch nhiều ảnh
thật sự trong 1 lần gọi (khác GroundingDinoProcessor) — ocr_batch() bên dưới LẶP từng ảnh
trong 1 lượt remote call, chỉ để amortize chi phí cold-start/load model qua nhiều ảnh, KHÔNG
phải batch song song thật trên GPU (ghi rõ ở đây tránh hiểu nhầm khi đọc lại sau).

Chạy thử (dev, hot-reload): modal serve ocr_app.py
Deploy thật:                modal deploy ocr_app.py
"""
import modal

MAX_CONTAINERS = 8  # gioi han tai khoan Modal da xac nhan (2026-08-06, xem vocab_discovery_app.py)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # dependency cua opencv (PaddleOCR dung ben trong)
    # BUG THAT (2026-08-10, lan 3): paddlepaddle-gpu==2.6.1 tu PyPI thuong can cuDNN runtime
    # KHONG co san trong image debian_slim - "Cannot load cudnn shared library". Thu doi sang
    # paddlepaddle CPU (lan 4) van loi y het -> nghi wheel PyPI thieu bundle cuDNN dung cach.
    # LAN 5 (theo huong dan chinh thuc paddlepaddle.org.cn): cai paddlepaddle-gpu==3.3.1 tu
    # INDEX RIENG cua PaddlePaddle (khac PyPI mac dinh) khop dung CUDA 12.9 - wheel chinh thuc
    # nay bundle/khop cuDNN dung cach hon ban PyPI generic.
    .pip_install(
        "paddlepaddle-gpu==3.3.1",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu129/",
        force_build=True,
    )
    .pip_install("paddleocr==2.7.3", "pillow", "numpy<2", force_build=True)
    .run_commands("pip install 'numpy<2' --force-reinstall --no-deps")
)

model_cache_vol = modal.Volume.from_name("aic2026-ocr-cache", create_if_missing=True)

app = modal.App("aic2026-ocr")


@app.cls(
    image=image,
    gpu="T4",  # paddlepaddle-gpu can GPU that - CUDA 12.9 wheel, driver Modal ho tro nguoc.
    max_containers=MAX_CONTAINERS,
    scaledown_window=5 * 60,
    volumes={"/root/.paddleocr": model_cache_vol},
)
class OCRDetector:
    @modal.enter()
    def load(self):
        from paddleocr import PaddleOCR

        # lang='vi': PaddleOCR co san model nhan dien tieng Viet (PP-OCR multilingual series).
        # use_angle_cls=True: xu ly chu bi xoay/nghieng (bien hieu, chu tren vat the ngoai doi
        # - khac lower-third luon thang truc).
        self.ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)

    @modal.method()
    def ocr_batch(self, images_bytes: list[bytes]) -> list[list[dict]]:
        """LAP tung anh (xem docstring dau file - PaddleOCR khong batch that duoc). Tra ve list
        (dung thu tu images_bytes), moi phan tu la list[{"text","score","ymin","xmin","ymax","xmax"}]
        (bbox da normalize [0,1] theo kich thuoc anh THAT, khong phai px tho - khac
        grounding_dino_app.py tra px roi de driver tu chia, o day chia san vi khong can gi them)."""
        import io
        import sys

        import numpy as np
        from PIL import Image

        out = []
        for b in images_bytes:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = img.size
            arr = np.array(img)

            try:
                result = self.ocr.ocr(arr, cls=True)
            except Exception as e:
                # BUG THAT (2026-08-10): truoc day nuot IM LANG loi that (vd cuDNN thieu) ->
                # tuong nhu "chay duoc, khong co chu" thay vi crash ro rang. In ra stderr de
                # loi that hien trong `modal app logs`, KHONG con nuot cau cam.
                print(f"[OCR LOI] {type(e).__name__}: {e}", file=sys.stderr)
                out.append([])
                continue

            items = []
            lines = result[0] if result and result[0] else []
            for box_points, (text, score) in lines:
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                items.append({
                    "text": text,
                    "score": float(score),
                    "ymin": max(0.0, min(ys) / h),
                    "xmin": max(0.0, min(xs) / w),
                    "ymax": min(1.0, max(ys) / h),
                    "xmax": min(1.0, max(xs) / w),
                })
            out.append(items)
        return out

    @modal.method()
    def debug_paddle_info(self) -> str:
        """TAM THOI - kiem tra dung ban paddle nao dang chay THAT trong container (2026-08-10,
        van gap loi cudnn dua ca sau khi doi sang paddlepaddle CPU - can xac nhan co phai
        dang chay nham ban -gpu khong)."""
        import paddle

        return (
            f"paddle.__version__={paddle.__version__}, "
            f"is_compiled_with_cuda={paddle.is_compiled_with_cuda()}, "
            f"paddle.__file__={paddle.__file__}"
        )

    @modal.method()
    def debug_raw(self, image_bytes: bytes) -> str:
        """TAM THOI - debug cau truc result() that cua PaddleOCR 2.7.3 (2026-08-10, ocr_batch
        tra 0 dong chu tren frame chac chan CO chu ro net - can soi raw output truoc khi doan)."""
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result = self.ocr.ocr(arr, cls=True)
        return repr(result)[:3000]


@app.local_entrypoint()
def test():
    """modal run ocr_app.py — test nhanh voi 1 anh co chu that."""
    import urllib.request

    # anh mau co chu tieng Anh ro rang (bien bao duong) - chi de kiem tra pipeline chay duoc,
    # khong danh gia chat luong tieng Viet (can test rieng tren keyframe that - xem
    # offline/build_ocr_index.py --limit).
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()

    detector = OCRDetector()
    results = detector.ocr_batch.remote([img_bytes])
    print(results)
