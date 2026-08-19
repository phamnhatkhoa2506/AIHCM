import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))
import pandas as pd
from config import INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
frcnn = pd.read_parquet(DENSE_DIR / "objects_baseline_frcnn.parquet")
owlv2 = pd.read_parquet(DENSE_DIR / "objects_owlv2_samesample.parquet")

N_IMAGES = 8000

for name, df in [("Faster R-CNN Inception-ResNet-v2 (baseline BTC)", frcnn), ("OWLv2 (open-vocab, 514 nhan)", owlv2)]:
    n_det = len(df)
    n_imgs_with_det = df[["video_id", "frame_idx"]].drop_duplicates().shape[0]
    n_labels = df["label"].nunique()
    print(f"\n=== {name} ===")
    print(f"Tong detection (score>=0.15): {n_det}")
    print(f"Trung binh detection/anh (tren {N_IMAGES} anh): {n_det/N_IMAGES:.2f}")
    print(f"So anh CO it nhat 1 detection: {n_imgs_with_det}/{N_IMAGES} ({n_imgs_with_det/N_IMAGES*100:.1f}%)")
    print(f"So nhan KHAC NHAU xuat hien: {n_labels}")
    print(f"Diem trung binh: {df['score'].mean():.3f}, median: {df['score'].median():.3f}")
    top10 = df["label"].value_counts().head(10)
    print("Top 10 nhan pho bien nhat:")
    for lbl, cnt in top10.items():
        print(f"  {lbl}: {cnt}")

# So sanh tren TUNG anh cu the (video_id, frame_idx) - so detection moi model
frcnn_per_img = frcnn.groupby(["video_id", "frame_idx"]).size().rename("n_frcnn")
owlv2_per_img = owlv2.groupby(["video_id", "frame_idx"]).size().rename("n_owlv2")
merged = pd.concat([frcnn_per_img, owlv2_per_img], axis=1).fillna(0)
print(f"\n=== So sanh tren {len(merged)} anh co it nhat 1 detection (1 trong 2 model) ===")
print(f"Ca 2 model DEU co detection: {((merged['n_frcnn']>0)&(merged['n_owlv2']>0)).sum()}")
print(f"CHI FRCNN co detection: {((merged['n_frcnn']>0)&(merged['n_owlv2']==0)).sum()}")
print(f"CHI OWLv2 co detection: {((merged['n_frcnn']==0)&(merged['n_owlv2']>0)).sum()}")

frcnn_labels = set(frcnn["label"].unique())
owlv2_labels = set(owlv2["label"].unique())
print(f"\nNhan chung (co trong CA 2): {len(frcnn_labels & owlv2_labels)}")
print(f"Nhan CHI co o FRCNN: {len(frcnn_labels - owlv2_labels)}")
print(f"Nhan CHI co o OWLv2: {len(owlv2_labels - frcnn_labels)}")
