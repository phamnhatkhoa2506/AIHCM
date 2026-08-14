"""So sánh nhanh Grounding DINO (open-vocabulary) vs detector cũ (closed-set 514 nhãn) trên
đúng 3 frame đã xác nhận bị gán nhãn SAI ở audit_object_labels.py (2026-08-06):
  - L24_V026/176: detector cũ gán "Dog" (0.82) cho đầu lân múa lân.
  - L24_V003/97:  detector cũ gán "Camel" cho người múa rồng.
  - L21_V013/14:  detector cũ gán "Watercraft" cho background trường quay tin tức.

CHỈ chạy CPU local cho vài ảnh (demo/so sánh, không phải full corpus) — máy này không có CUDA.
Nếu kết quả tốt mới cân nhắc hạ tầng GPU (Modal) để chạy full corpus sau.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys

import torch
from PIL import Image
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from keyframe_images import read_keyframe_bytes
import io

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

# Grounding DINO: prompt la cac cum tu cach nhau boi dau cham, lowercase.
TEXT_PROMPT = (
    "person. dog. camel. boat. ship. "
    "lion dance costume. dragon dance costume. mascot costume. "
    "news studio backdrop. television screen. building."
)

TEST_CASES = [
    ("L24_V026", 176, "detector cu gan 'Dog' -> that ra la dau lan mua lan"),
    ("L24_V003", 97, "detector cu gan 'Camel' -> that ra la nguoi mua rong"),
    ("L21_V013", 14, "detector cu gan 'Watercraft' -> that ra la truong quay tin tuc"),
]


def main() -> None:
    print(f"Dang tai model {MODEL_NAME} (lan dau se tai ve, sau do cache)...", file=sys.stderr)
    processor = GroundingDinoProcessor.from_pretrained(MODEL_NAME)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    for video_id, local_idx, note in TEST_CASES:
        print("\n" + "=" * 70)
        print(f"{video_id}/{local_idx} — {note}")
        img_bytes = read_keyframe_bytes(video_id, local_idx)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.25,
            text_threshold=0.20,
            target_sizes=[image.size[::-1]],
        )[0]

        if len(results["labels"]) == 0:
            print("  (khong detect duoc gi voi threshold hien tai)")
        for label, score, box in zip(results["labels"], results["scores"], results["boxes"]):
            box = [round(float(x), 1) for x in box.tolist()]
            print(f"  {label:<30} score={float(score):.3f}  box(px)={box}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
