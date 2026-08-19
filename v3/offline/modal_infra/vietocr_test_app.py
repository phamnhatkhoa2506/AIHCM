"""Modal app THU NGHIEM (2026-08-18, theo yeu cau nguoi dung): test pipeline CRAFT (detect) +
PARSeq fine-tune tieng Viet (recognize) tu repo
https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition — so sanh voi
PaddleOCR tren dung 1 anh kho da biet (bien 3 dong "GIÁO HỘI PHẬT GIÁO HÒA HẢO/BAN TRỊ SỰ TRUNG
ƯƠNG/BAN QUẢN TỰ AN HÒA TỰ" - PaddleOCR bo sot 2 dong dau, doc sai dong con lai, xem hoi thoai).

KHONG chay local (theo yeu cau nguoi dung "Chạy trên modal á bạn, đừng chạy ở local") - CRAFT +
PARSeq deu la model PyTorch, co the chay CPU nhung cham; chay tren Modal cho nhanh + khong dung
tai nguyen may nguoi dung.

Deploy:  modal run offline/modal_infra/vietocr_test_app.py --image-path <duong dan anh>
"""
import modal

app = modal.App("aic2026-vietocr-test")

REPO_URL = "https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition"
# id file gdown lay tu download_models.sh cua repo - 1 file zip DUY NHAT chua ca 2 checkpoint
# (weights/detect/craft_mlt_25k.pth + weights/rec/best-parseq.ckpt).
WEIGHTS_GDRIVE_ID = "1Az9psFV6C1qiqFCL2s6DIBYRKr3Bkjd7"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "opencv-python-headless", "scikit-image",
        "shapely", "gdown", "einops", "timm==0.6.12",
        # PARSeq (strhub) dung EPOCH_OUTPUT/STEP_OUTPUT tu pytorch_lightning.utilities.types,
        # bi xoa o pytorch-lightning>=2.0 -> ghim ban 1.9.x cuoi cung con giu 2 alias nay.
        "pytorch-lightning==1.9.5", "hydra-core", "omegaconf", "nltk", "Pillow", "lmdb",
        "setuptools<81",  # setuptools>=81 bo pkg_resources - lightning_fabric can module nay
        # KHONG cai streamlit that (repo chi dung no cho decorator @st.cache - streamlit
        # 1.16.0 doi altair<5 nhung pip resolve altair moi hon, xung dot ImportError. Gia lap
        # 1 module "streamlit" GIA voi ham cache() la no-op decorator - xem run_commands duoi).
    )
    .run_commands(
        "python -c \""
        "import os; "
        "os.makedirs('/usr/local/lib/python3.10/site-packages/streamlit', exist_ok=True); "
        "open('/usr/local/lib/python3.10/site-packages/streamlit/__init__.py','w').write("
        "'def cache(*a, **k):\\n'"
        "'    def deco(f):\\n'"
        "'        return f\\n'"
        "'    return deco if not (a and callable(a[0])) else a[0]\\n'"
        ")\""
    )
    .run_commands(
        f"git clone --depth 1 {REPO_URL} /root/repo",
        # strhub (goi PARSeq that su) can cai o dang editable de import duoc tu
        # src.parseq.strhub - --no-deps vi cac dep that (torch/torchvision...) da cai o tren.
        "cd /root/repo/src/parseq && pip install -e . --no-deps",
        f"cd /root/repo && gdown {WEIGHTS_GDRIVE_ID} -O weights.zip "
        "&& unzip -q weights.zip -d . && rm weights.zip",
    )
)


@app.function(image=image, timeout=900)
def run_craft_parseq(image_bytes: bytes) -> list[dict]:
    """Chay CRAFT (detect vung chu, khong can fine-tune rieng - da train tren MLT, da ngon
    ngu) + PARSeq (nhan dang, DA fine-tune tieng Viet theo mo ta repo) tren 1 anh - tra ve
    list[{"box": [[x,y]x4], "text": str, "prob": float}]."""
    import sys
    import os

    sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")

    with open("upload_image.jpg", "wb") as f:
        f.write(image_bytes)

    import cv2
    import numpy as np
    from PIL import Image as PILImage

    # torchvision moi (>=0.15) da xoa han torchvision.models.vgg.model_urls (repo cu
    # src/craft/basenet/vgg16_bn.py con dung API nay de tai pretrained VGG16-BN) -> vá tay
    # bang cach gan lai dict model_urls voi URL chinh thuc cua pytorch truoc khi import repo.
    import torchvision.models.vgg as _tv_vgg
    if not hasattr(_tv_vgg, "model_urls"):
        _tv_vgg.model_urls = {
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        }

    # torch>=2.6 doi default torch.load(weights_only=True), nhung checkpoint PARSeq cu
    # (pytorch_lightning .ckpt) chua pickle cua class/ham thuong (vd getattr) nen bi chan.
    # Tin tuong nguon checkpoint nay (repo cong khai nguoi dung cung cap) -> vá lai default.
    import functools
    import torch as _torch
    _torch.load = functools.partial(_torch.load, weights_only=False)

    from src.craft.load_model import load_model_craft
    from src.craft.craft_predict import predict_craft
    from src.parseq.load_model import load_model_parseq
    from src.parseq.parseq_predict import predict_parseq
    from src.utils.four_points_transform import four_points_transform

    net, refine_net = load_model_craft()
    parseq, img_transform = load_model_parseq(device="cpu")

    boxes = predict_craft(net, refine_net, image_path="upload_image.jpg", text_threshold=0.65, cuda_state=False)
    main_image = cv2.imread("upload_image.jpg")

    results = []
    for box in boxes:
        sub_img = four_points_transform(main_image, np.array(box, dtype="float32"))
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        sub_img_pil = PILImage.fromarray(sub_img)
        pred, prob = predict_parseq(parseq=parseq, img_transform=img_transform, image=sub_img_pil, device="cpu")
        results.append({"box": box, "text": pred[0] if isinstance(pred, list) else pred, "prob": float(prob)})
    return results


@app.local_entrypoint()
def main(image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    results = run_craft_parseq.remote(image_bytes)
    print(f"\n{len(results)} vung chu phat hien duoc:")
    for r in results:
        print(f"  '{r['text']}'  (prob={r['prob']:.3f})  box={r['box']}")
