import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pyvi import ViTokenizer
import underthesea

queries = [
    "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa",
    "một vận động viên đang băng bó cổ tay bị thương",
    "cận cảnh đầu gối của cầu thủ sau cú va chạm",
    "người phụ nữ đeo đồng hồ ở cổ tay đang cầm ly cà phê",
]

for q in queries:
    print("Q:", q)
    print("  pyvi       :", ViTokenizer.tokenize(q))
    print("  underthesea:", underthesea.word_tokenize(q, format="text"))
    print()
