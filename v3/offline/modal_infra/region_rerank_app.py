"""Modal app NHE, CPU-only, LUON GIU AM (min_containers=1) - Region-CLIP rerank server (SigLIP2
CHI 1 model, theo yeu cau nguoi dung 2026-08-15) cho bo dense.

Y TUONG (2026-08-15, theo de xuat nguoi dung "region CLIP cung lam 1 server rieng nhu embedding
model"): GIONG HET ly do query_encoders_app.py ra doi - truoc day apply_region_clip_rerank()
trong dense_search.py phai nap 2 file NANG (region_embeddings_siglip.npy 5.3GB +
objects_index.parquet 14.5 trieu dong) TRUC TIEP tren may local MOI PHIEN Streamlit - lan dau
mat ~1-2 phut, an RAM may (~5.3GB, co luc con hon RAM trong con lai). Chuyen het sang server
Modal: du lieu CHI nap 1 LAN duy nhat luc container khoi dong (khong phai moi phien local), may
local CHI gui danh sach (video_id, frame_id) can rerank + ten thuoc tinh (rat nhe qua mang),
nhan ve diem so - khong con phai giu 5.3GB trong RAM may local nua.

Du lieu: Volume "aic2026-region-index" (xem offline/upload_region_index_to_volume.py) - PHAI
chay lai script do MOI LAN co embedding moi (server KHONG tu dong sync).

Tu ma hoa TEXT thuoc tinh NGAY TRONG APP NAY (co san SigLIP2, khong goi vong qua
aic2026-query-encoders nua) - tranh 1 vong mang thua.

Deploy: modal deploy region_rerank_app.py
Goi tu: share/tiers/dense_search.py::apply_region_clip_rerank (thay logic local cu)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "pillow", "accelerate", "numpy", "pandas", "pyarrow")
)

region_index_vol = modal.Volume.from_name("aic2026-region-index", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung
app = modal.App("aic2026-region-rerank")

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
VOLUME_MOUNT = "/data"
FRAME_IOU_UNUSED = None  # khong dung IoU o day - match theo (video_id, frame_id) + nhan, giong dense_search.py cu


@app.cls(
    image=image,
    min_containers=1,  # LUON giu 1 container am - muc tieu chinh cua app nay.
    scaledown_window=30 * 60,
    timeout=120,
    volumes={VOLUME_MOUNT: region_index_vol, "/root/.cache/huggingface": hf_cache_vol},
)
class RegionRerankServer:
    @modal.enter()
    def load(self):
        import os

        import numpy as np
        import pandas as pd
        from transformers import AutoModel, AutoProcessor

        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

        # SigLIP2 - CHI can nhanh TEXT (dung y het query_encoders_app.py::encode_siglip_text).
        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

        # Du lieu region-embedding + scope detections - nap 1 LAN duy nhat luc container
        # khoi dong (KHONG phai moi query, KHAC han ban local truoc day).
        self._scope_df = pd.read_parquet(f"{VOLUME_MOUNT}/scope_detections_cache.parquet")
        self._vecs = np.load(f"{VOLUME_MOUNT}/region_embeddings_siglip.npy")
        ids = np.load(f"{VOLUME_MOUNT}/region_embeddings_siglip_detection_ids.npy")
        self._id_to_row = {int(did): i for i, did in enumerate(ids)}

    def _encode_text(self, text: str):
        import torch

        inputs = self._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            feats = self._siglip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy()

    @modal.method()
    def rerank(
        self, frame_keys: list[list], entity_labels: list[str], attribute_text: str
    ) -> list[float]:
        """frame_keys: [[video_id, frame_id], ...] (candidate pool cua 1 lan query, ~40-200
        frame). Tra ve list[float] DUNG THU TU frame_keys - diem CAO NHAT trong cac box khop
        entity_labels cua frame do (0.0 neu khong co box nao khop hoac chua co embedding).
        Vectorized .isin() (giong fix o dense_search.py) - CHI loc dong lien quan, khong duyet
        toan bo scope_df moi lan goi."""
        import numpy as np
        import pandas as pd

        text_vec = self._encode_text(attribute_text)

        frame_key_set = {(vid, int(fid)) for vid, fid in frame_keys}
        mi = pd.MultiIndex.from_arrays([self._scope_df["video_id"], self._scope_df["frame_idx"]])
        scoped = self._scope_df[mi.isin(frame_key_set) & self._scope_df["label"].isin(entity_labels)]

        frame_best: dict[tuple[str, int], float] = {}
        for detection_id, video_id, frame_idx in zip(scoped["detection_id"], scoped["video_id"], scoped["frame_idx"]):
            row_pos = self._id_to_row.get(int(detection_id))
            if row_pos is None:
                continue
            sim = float(np.dot(self._vecs[row_pos], text_vec))
            key = (video_id, int(frame_idx))
            if sim > frame_best.get(key, -1.0):
                frame_best[key] = sim

        return [frame_best.get((vid, int(fid)), 0.0) for vid, fid in frame_keys]


@app.local_entrypoint()
def test():
    server = RegionRerankServer()
    scores = server.rerank.remote(
        frame_keys=[["L22_V008", 16680], ["L21_V021", 28730]],
        entity_labels=["Dog"],
        attribute_text="a yellow dog",
    )
    print("scores:", scores)
