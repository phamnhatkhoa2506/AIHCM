"""Ghep candidate_manifest.json (100 (video_id,local_idx) da lay mau) + query da viet tay
(sau khi XEM THAT tung frame qua candidate_sheets/*.png, khong doan mu) -> queries_100.jsonl,
CUNG schema voi queries.jsonl goc (6 mau cu, van giu nguyen, khong dung file nay de khong xoa
mat ground-truth da verify truoc).

Chay: python offline/benchmark/build_queries_100.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "share"))

import json

import pandas as pd

from config import INDEX_DIR

MANIFEST_PATH = _Path(__file__).resolve().parent / "candidate_manifest.json"
OUT_PATH = _Path(__file__).resolve().parent / "queries_100.jsonl"
WINDOW = 150  # +-150 frame_idx quanh mốc, giong dung ty le voi queries.jsonl goc (~300 rong)

# Query viet tay theo DUNG THU TU candidate_manifest.json (index 0..99) - da xem tung anh
# qua candidate_sheets/*.png truoc khi viet, KHONG doan tu ten file/metadata.
QUERIES: list[str] = [
    "một bát salad dưa leo trộn rau",  # 0
    "một người đàn ông trả lời phỏng vấn cạnh hồ bơi có trẻ em đang bơi",  # 1
    "tay đang chiên trứng trong chảo nhỏ",  # 2
    "một đàn hồng hạc đứng trong nước, có người đang cho ăn",  # 3
    "màn hình tối chuyển cảnh có chữ mờ",  # 4
    "cà chua bổ múi cau xếp thành hình ngôi sao nhìn từ trên xuống",  # 5
    "hai ly nước có đá viên và thạch rau câu màu xanh",  # 6
    "toà nhà trường học nhìn từ dưới tán cây, có biển hiệu trường quốc tế",  # 7
    "tay bỏ hành lá cắt nhỏ vào tô thuỷ tinh",  # 8
    "hai người đứng trong bếp, một người mặc áo trắng một người áo xanh",  # 9
    "hai viên thịt viên nhỏ đặt trên đĩa trắng",  # 10
    "rưới sốt xanh lên đĩa gỏi có thịt cua",  # 11
    "chảo đỏ đun nước sôi trên bếp gas lửa xanh",  # 12
    "bàn bày nguyên liệu cuốn gồm bánh tráng, rau và cà rốt cạnh chảo",  # 13
    "một cái bánh xèo vàng nằm trong chảo chống dính màu đen",  # 14
    "hai đầu bếp áo trắng đứng nói chuyện trong bếp",  # 15
    "tay đang cắt khoanh dưa leo",  # 16
    "người phụ nữ áo trắng cầm đĩa rau cải bó xôi",  # 17
    "đầu bếp nam đứng cạnh bảng hiệu chương trình nấu ăn",  # 18
    "lời giải bài toán vật lý về mạch điện hiển thị trên màn hình",  # 19
    "hai người đứng bếp, nam mặc áo trắng nữ mặc áo hồng",  # 20
    "một người đàn ông đội đèn pin trên đầu đứng trong bóng tối",  # 21
    "tay cầm thìa nhỏ múc nước sốt trắng",  # 22
    "tay cầm một hộp muối nhỏ màu trắng có logo đỏ",  # 23
    "hai người đứng nấu ăn cạnh bảng hiệu chương trình nấu ăn",  # 24
    "một chiếc thuyền chở người mặc áo phao cam trên sông có nhà sàn",  # 25
    "đậu que, ớt chuông vàng và củ hành đặt trên hai đĩa trắng",  # 26
    "đầu bếp nam đứng cạnh logo thương hiệu gia vị đang nói chuyện",  # 27
    "củ sen cắt lát xếp hình hoa trên đĩa màu xanh",  # 28
    "tô thuỷ tinh đựng nấm mèo và tỏi băm nhỏ",  # 29
    "một bé gái mặc đồng phục học sinh xanh trắng ngồi trong studio",  # 30
    "một người đàn ông đeo kính đứng trước kệ sách",  # 31
    "hai người đứng bếp cạnh bồn rửa, nam áo trắng nữ áo hồng",  # 32
    "tay dùng dao cắt miếng thịt gà trên thớt gỗ",  # 33
    "hai đầu bếp đứng nấu ăn cạnh bếp gas đôi",  # 34
    "bài tập hoá học so sánh nhiệt độ sôi các chất lỏng trên màn hình",  # 35
    "một nhóm trẻ em mặc áo bơi ngồi cạnh hồ bơi",  # 36
    "một người đội lưới trên đầu đứng cạnh nhiều thùng chai dầu ăn màu vàng đỏ",  # 37
    "sân khấu ngoài trời có hồ bơi nhỏ và cầu trượt trang trí lễ hội",  # 38
    "hai đầu bếp đứng cạnh bồn rửa trong bếp, một áo trắng một áo hồng",  # 39
    "các viên thịt cuộn được nướng trên vỉ nướng màu đen",  # 40
    "hai người đàn ông đội mũ xanh đứng giữa rừng cây",  # 41
    "logo kênh tin tức chữ màu xanh trên nền xám",  # 42
    "một cây cầu gỗ bắc giữa hàng cây xanh có cờ đỏ",  # 43
    "hai nữ sinh mặc áo đỏ trắng ngồi nói chuyện trên bãi cỏ",  # 44
    "một đống củi gỗ xếp chồng lên nhau",  # 45
    "hai vận động viên đua xe đạp mặc áo xanh trên đường phố",  # 46
    "một đĩa gỏi trắng trang trí đẹp mắt có đũa gắp",  # 47
    "một người phụ nữ áo trắng đỏ đứng cạnh bảng hiệu chương trình nấu ăn",  # 48
    "một mặt bàn gỗ trống không có người",  # 49
    "hiệu ứng chuyển cảnh số bốn màu tím hồng",  # 50
    "một con bò nằm trong chuồng, bản tin nói về bò sữa bị chết hàng loạt",  # 51
    "một người phụ nữ áo trắng đứng nấu ăn một mình trong bếp trống",  # 52
    "đoàn xe mô tô hộ tống nhìn từ trên cao trên đường đua",  # 53
    "một người phụ nữ mặc áo hoa đứng cạnh bàn bếp",  # 54
    "hai người xem điện thoại, trong đó có một cụ già tóc bạc",  # 55
    "tay dùng nĩa trộn bột trong tô trắng",  # 56
    "đoàn đông vận động viên đua xe đạp mặc áo nhiều màu",  # 57
    "ớt chuông đỏ vàng và ớt xanh xếp thành hình trái tim",  # 58
    "một nhóm người lội biển lúc hoàng hôn gần thuyền đậu bờ",  # 59
    "hai người đứng cạnh bếp, nam áo trắng nữ áo xám",  # 60
    "một đầu lân màu vàng đang biểu diễn múa lân trên sân khấu",  # 61
    "tay dùng dao cắt xà lách búp thành từng miếng",  # 62
    "hai người đứng bếp, nam áo trắng nữ áo xanh lá",  # 63
    "một người đàn ông áo trắng đứng nói chuyện trong bếp trống",  # 64
    "hai người đứng nấu ăn cạnh bếp gas đôi trong bếp",  # 65
    "đầu lân vàng và hoa mai trang trí trên sân khấu ngoài trời",  # 66
    "tay cầm một cây kèn saxophone màu bạc",  # 67
    "một buổi lễ ký kết học bổng có bốn người đứng phát biểu trên sân khấu",  # 68
    "một cổng chùa cổ mái ngói đỏ có chữ Hán giữa cây xanh",  # 69
    "một màn hình trắng trống chuyển cảnh",  # 70
    "một màn hình chứng khoán màu hồng trong phòng giao dịch",  # 71
    "tay cầm một quả dừa trắng đã gọt vỏ",  # 72
    "một đội trống hội mặc áo vàng hồng biểu diễn vào ban đêm",  # 73
    "bài giải hình học không gian về mặt phẳng trên màn hình",  # 74
    "một chiếc ghe máy chạy trên sông nhìn từ trên cao",  # 75
    "tay cầm một cái bánh tráng tròn màu trắng",  # 76
    "một đĩa trắng có nước sốt màu cam đổ thành hình tròn ở giữa",  # 77
    "cận cảnh hạt gạo trắng đang đổ xuống",  # 78
    "một nồi thuỷ tinh đun nước có cà chua, tay cầm thìa khuấy",  # 79
    "một người đàn ông đội nón lá đứng giữa ruộng ngô xanh",  # 80
    "đũa gắp một cuốn chả giò chấm vào chén nước mắm",  # 81
    "tay vẽ hình một cái cây có chấm bi nhiều màu trên giấy",  # 82
    "hai vận động viên nữ ôm nhau ăn mừng sau trận đấu karate",  # 83
    "tay hái quả mọng đỏ trên cây, bản tin nói về Thanh Hoá",  # 84
    "công thức hoá học về amino axit viết trên bảng",  # 85
    "một tô canh trong niêu đất đặt trên khăn trải bàn trắng",  # 86
    "một người đàn ông mặc áo sọc đứng tung đồ trong bếp",  # 87
    "một người đàn ông áo be ngồi cạnh bình hoa tím trong bếp",  # 88
    "hai người đứng bếp, nam áo trắng nữ áo tím nhạt",  # 89
    "một chảo xào nấm đang bốc khói",  # 90
    "hai người đứng trong bếp, một người đang khoanh tay",  # 91
    "một người phụ nữ tóc đen dài đứng nói chuyện, hơi nheo mắt",  # 92
    "hai người phụ nữ lớn tuổi đứng cạnh bồn rửa trong bếp",  # 93
    "hành tím cắt múi và dưa leo xanh đặt trên thớt",  # 94
    "sóng biển khổng lồ, bản tin nói về việc chinh phục những con sóng lớn",  # 95
    "tay đeo găng tay đang cắt hành lá trên thớt gỗ",  # 96
    "một người đàn ông áo trắng đứng đọc sách trong thư viện",  # 97
    "những tấm pin năng lượng mặt trời đặt giữa cây xanh",  # 98
    "một đầu lân màu đỏ trắng đang múa trên sân khấu có treo cờ",  # 99
]

# (index candidate, question, answer) - chi mot phan (khong phai het 100) co cau hoi QA ro
# rang, chi tiet co the tra loi CHAC CHAN tu hinh/OCR nhin thay (khong doan).
QA_ITEMS: list[tuple[int, str, str]] = [
    (1, "người đàn ông trong khung hình tên gì?", "Phạm Thanh Thới"),
    (51, "bản tin nói có bao nhiêu con bò sữa bị chết?", "hơn 200 con"),
    (67, "người trong khung hình đang cầm nhạc cụ gì?", "kèn saxophone"),
    (83, "hai vận động viên vừa thi đấu môn thể thao gì?", "karate"),
    (95, "địa danh nào được nhắc tới trong bản tin?", "Itacoatiara"),
    (43, "chữ gì xuất hiện ở góc màn hình?", "VIỆT NAM"),
    (16, "tay đang cắt loại rau củ gì?", "dưa leo"),
    (58, "ớt được xếp thành hình gì?", "hình trái tim"),
    (73, "đội trống hội mặc áo màu gì?", "vàng và hồng"),
    (61, "đầu lân đang biểu diễn có màu gì?", "màu vàng"),
]


def main() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == len(QUERIES), f"manifest={len(manifest)} queries={len(QUERIES)} lech nhau"

    meta = pd.read_parquet(INDEX_DIR / "meta.parquet")

    def frame_range_for(video_id: str, local_idx: int) -> list[int]:
        row = meta[(meta["video_id"] == video_id) & (meta["local_idx"] == local_idx)]
        frame_idx = int(row.iloc[0]["frame_idx"])
        return [max(0, frame_idx - WINDOW), frame_idx + WINDOW]

    lines = []
    for i, (cand, query_text) in enumerate(zip(manifest, QUERIES)):
        gt_range = frame_range_for(cand["video_id"], cand["local_idx"])
        lines.append({
            "id": f"kis100_{i:03d}",
            "type": "KIS",
            "query": query_text,
            "gt_video_id": cand["video_id"],
            "gt_frame_range": gt_range,
            "tags": ["sample100"],
            "source": f"sample_candidates.py idx={i}, xem that qua candidate_sheets/",
        })

    for i, question, answer in QA_ITEMS:
        cand = manifest[i]
        gt_range = frame_range_for(cand["video_id"], cand["local_idx"])
        lines.append({
            "id": f"qa100_{i:03d}",
            "type": "QA",
            "event_query": QUERIES[i],
            "question": question,
            "gt_video_id": cand["video_id"],
            "gt_frame_range": gt_range,
            "gt_answer": answer,
            "tags": ["sample100", "qa"],
            "source": f"sample_candidates.py idx={i}, xem that qua candidate_sheets/",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"da viet {len(lines)} dong ({len(QUERIES)} KIS + {len(QA_ITEMS)} QA) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
