"""Dịch 514 nhãn OpenImages V4 (tiếng Anh, từ objects_index.parquet) sang tiếng Việt,
dùng để người dùng gõ tiếng Việt trong app.py thay vì phải biết đúng tên nhãn tiếng Anh.

Dịch máy từng-từ (deep_translator/Google) hay sai với từ ngắn đa nghĩa (vd "Van" ->
"Vân" thay vì "xe tải nhỏ", vì model coi là tên riêng). OVERRIDES bên dưới patch tay
các trường hợp đã phát hiện — soát lại LABEL_VI_PATH sau khi build, thêm override nếu
thấy sai thêm.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import sys
import time

from deep_translator import GoogleTranslator

from config import INDEX_DIR, OBJECTS_INDEX_PATH

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"

# Patch tay: nhãn ngắn/đa nghĩa mà dịch máy hay hiểu nhầm thành tên riêng hoặc nghĩa khác.
OVERRIDES: dict[str, str] = {
    "Van": "xe tải nhỏ",
    "Bat (Animal)": "con dơi",
    "Turkey": "con gà tây",
    "Crown": "vương miện",
    "Mouse": "con chuột",
    "Tap": "vòi nước",
    "Mixer": "máy trộn",
    "Drum": "cái trống",
    "Cricket ball": "bóng cricket",
    "Winter melon": "bí đao",
    # phát hiện thêm khi soát toàn bộ 514 nhãn (2026-08-05):
    "Boot": "giày ống",
    "Bust": "tượng bán thân",
    "Building": "toà nhà",
    "Dumbbell": "quả tạ",
    "Drink": "đồ uống",
    "Organ": "đàn organ",
    "Oven": "lò nướng",
    "Scale": "cái cân",
    "Jet ski": "mô tô nước",
    "Land vehicle": "phương tiện đường bộ",
    "Stool": "ghế đẩu",  # dịch máy ra "Phân" (nhầm nghĩa khác của "stool") - lỗi rõ nhất
    "Tart": "bánh tart",
    "Woman": "phụ nữ",
    "Doll": "búp bê",
}


def main() -> None:
    df_labels = _load_labels()
    translator = GoogleTranslator(source="en", target="vi")

    result: dict[str, str] = {}
    for i, label in enumerate(df_labels, 1):
        if label in OVERRIDES:
            result[label] = OVERRIDES[label]
        else:
            try:
                result[label] = translator.translate(label)
            except Exception as e:
                print(f"loi dich '{label}': {e}", file=sys.stderr)
                result[label] = label  # fallback: giữ nguyên tiếng Anh
            time.sleep(0.05)  # tránh spam API dịch miễn phí

        if i % 100 == 0 or i == len(df_labels):
            print(f"[{i}/{len(df_labels)}]", file=sys.stderr)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_VI_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Xong: {len(result)} nhan -> {LABEL_VI_PATH}")


def _load_labels() -> list[str]:
    import pandas as pd

    df = pd.read_parquet(OBJECTS_INDEX_PATH, columns=["label"])
    return sorted(df["label"].unique().tolist())


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
