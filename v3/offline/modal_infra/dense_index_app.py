"""Modal app NHE, CPU-only, LUON GIU AM (min_containers=1) - giu 3 ma tran embedding + FAISS
index (SigLIP2/PE-Core/BEiT-3, ~7.4GB tong cong) cho Tang 2 (vector search) cua bo dense.

Y TUONG (2026-08-16, theo yeu cau nguoi dung "may minh moi lan chay len no chiem gan het RAM ...
bo len modal roi goi", cung 1 ly do voi region_rerank_app.py/query_encoders_app.py da lam):
truoc day _load_dense_index() (dense_search.py) nap CA matrix.npy LAN faiss.index TRUC TIEP
tren may local, cache vinh vien qua lru_cache - mode "All"/rrf goi ca 3 model nen thuong xuyen
chiem het ~7.4GB RAM may local. Chuyen toan bo sang server Modal: may local CHI gui (model,
query_vector da encode san, top_k, candidate_keys) - rat nhe qua mang - nhan ve KET QUA DA XEP
HANG (list[dict]), khong con giu matrix/faiss trong RAM may local nua.

QUAN TRONG: server nay CHI lam dung 1 viec la XEP HANG theo cosine similarity (giong het logic
_rank_single() cu) - hard-filter theo object/OCR (candidates set) VAN tinh o may local (dung
objects_index.parquet, ~134MB, nhe hon nhieu nen KHONG chuyen - theo quyet dinh nguoi dung
2026-08-16), server chi nhan candidate_keys (video_id, frame_id) DA loc san tu truoc.

Du lieu: Volume "aic2026-dense-index" (xem offline/upload_dense_index_to_volume.py) - PHAI chay
lai script do MOI LAN co embedding moi (server KHONG tu dong sync).

Deploy: modal deploy dense_index_app.py
Goi tu: share/tiers/dense_search.py::_rank_single (thay logic local cu khi AIC_LOCAL_MODELS!=1)
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "pandas", "pyarrow", "faiss-cpu"
)

dense_index_vol = modal.Volume.from_name("aic2026-dense-index", create_if_missing=True)
app = modal.App("aic2026-dense-index")

VOLUME_MOUNT = "/data"
MODELS = ("siglip", "pe_core", "beit3")


@app.cls(
    image=image,
    min_containers=1,  # LUON giu 1 container am - muc tieu chinh cua app nay.
    scaledown_window=30 * 60,
    timeout=120,
    volumes={VOLUME_MOUNT: dense_index_vol},
)
# 2026-08-20 (cung ly do voi query_encoders_app.py::QueryEncoder - xem ghi chu o do): _rank_rrf
# goi CA 2 model (siglip/pe_core) "song song" qua thread, nhung neu container nay khong cho
# concurrent thi request thu 2 van phai XEP HANG cho request dau xong tren server that. rank()
# CHI DOC self._matrix/self._index/self._meta (khong ghi/doi) nen an toan chay dong thoi.
@modal.concurrent(max_inputs=8)
class DenseIndexServer:
    @modal.enter()
    def load(self):
        import numpy as np
        import pandas as pd
        import faiss

        self._matrix: dict[str, np.ndarray] = {}
        self._index: dict[str, "faiss.Index"] = {}
        for model in MODELS:
            self._matrix[model] = np.load(f"{VOLUME_MOUNT}/{model}_matrix.npy")
            self._index[model] = faiss.read_index(f"{VOLUME_MOUNT}/{model}_faiss.index")

        self._meta = pd.read_parquet(f"{VOLUME_MOUNT}/dense_meta.parquet")
        self._row_pos = {
            (vid, int(fid)): i
            for i, (vid, fid) in enumerate(zip(self._meta["video_id"], self._meta["frame_idx"]))
        }

    @modal.method()
    def rank(
        self, model: str, query_vec: list[float], top_k: int, candidate_keys: list[list] | None,
        light: bool = False,
    ) -> list:
        """Y HET logic _rank_single() cu (dense_search.py) - candidate_keys=None -> tim tren
        TOAN BO index qua FAISS; [] -> tra rong (khong phai "khong loc"); co gia tri -> tinh
        cosine TRUC TIEP tren dung tap do (KHONG dung FAISS-pool-roi-loc).

        light (2026-08-20, theo yeu cau nguoi dung: "vẫn chạy bằng Modal... TRAKE mở rộng pool
        lên 20000 chạy cực kỳ lâu, 160s"): do that goc re THAT SU la TRUYEN TAI qua mang (FAISS
        server-side gan nhu KHONG doi theo top_k, xem dense_search.py - nhung payload tra ve
        cang lon top_k cang nang, dac biet cot "path" la CHUOI DUONG DAN TUYET DOI dai). light=
        True -> CHI tra [video_id, frame_id, score] (list 3 phan tu, KHONG co shot_idx/path,
        KHONG doc self._meta.iloc[pos] - re hon CA server-side) - client tu ghep lai shot_idx/
        path tu dense_meta.parquet DA CO SAN LOCAL (khong qua Modal, xem
        _rank_single_remote()). light=False (mac dinh) giu HANH VI CU (dict day du) - tuong
        thich nguoc cho caller nao con goi truc tiep khong qua _rank_single_remote."""
        import numpy as np

        matrix = self._matrix[model]
        index = self._index[model]
        qvec = np.asarray([query_vec], dtype=np.float32)

        if candidate_keys is None:
            scores, idx = index.search(qvec, top_k)
            rows_iter = list(zip(idx[0].tolist(), scores[0].tolist()))
        elif not candidate_keys:
            return []
        else:
            candidate_set = {(vid, int(fid)) for vid, fid in candidate_keys}
            positions = np.array(
                [self._row_pos[k] for k in candidate_set if k in self._row_pos], dtype=np.int64
            )
            if len(positions) == 0:
                return []
            sub = matrix[positions]
            sc = sub @ qvec[0]
            order = np.argsort(-sc)[:top_k]
            rows_iter = list(zip(positions[order].tolist(), sc[order].tolist()))

        if light:
            vid_arr = self._meta["video_id"].values
            fidx_arr = self._meta["frame_idx"].values
            return [
                [vid_arr[pos], int(fidx_arr[pos]), float(score)]
                for pos, score in rows_iter
            ]

        out = []
        for pos, score in rows_iter:
            r = self._meta.iloc[pos]
            out.append({
                "video_id": r["video_id"], "frame_id": int(r["frame_idx"]),
                "shot_idx": int(r["shot_idx"]), "path": r["path"], "score": float(score),
            })
        return out


@app.local_entrypoint()
def test():
    import numpy as np

    server = DenseIndexServer()
    qvec = np.random.randn(768).astype("float32")
    qvec /= np.linalg.norm(qvec)
    results = server.rank.remote("siglip", qvec.tolist(), 5, None)
    print("results:", results)
