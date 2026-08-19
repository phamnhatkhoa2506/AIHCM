"""Pipeline THỰC THI ONLINE cho query graph — 3 tầng lọc + graph filter (Neo4j).

Kiến trúc (xem thảo luận thiết kế + notes/ban-do-du-lieu-pipeline.md):

    Query graph
        │
        ▼  TẦNG 1 · VECTOR  (Faiss/CLIP ANN, toàn corpus)      → "trông giống"
        ▼  TẦNG 2 · INVERTED INDEX  (label/count/triple, giao) → "có chứa"
        │        └─ 2 tầng này = LỌC THÔ (coarse). shortlist = vài trăm frame.
        ▼  (tuỳ) TẦNG 2.5 · TEMPORAL-ORDERING  (join cấp video)→ "A → B → C đúng thứ tự"
        │        └─ CHỈ khi query có chuỗi anchor. RẺ + chọn lọc mạnh → chạy TRƯỚC graph.
        ▼  TẦNG 3 · GRAPH  (Neo4j, NEO vào shortlist)          → "đúng cấu trúc/quan hệ"
        ▼  (tuỳ) TẦNG 4 · LAZY VLM EDGE  (chặn budget, shortlist tí hon)
        │
        ▼  kết quả xếp hạng + MatchTrace (giải thích được)

TEMPORAL — 2 loại KHÁC HẲN, đặt ở 2 chỗ ngược nhau:
  (a) Hard filter thời gian ("tuần trước", publish_date) = metadata frame-level →
      tầng 1, SỚM NHẤT. KHÔNG liên quan graph.
  (b) Thứ tự sự kiện ("A rồi B rồi C") = thuật toán ANCHOR chuỗi (Allen before) →
      tầng 2.5, SAU coarse-per-anchor, TRƯỚC graph. Rẻ (so timestamp + gom video) và
      chọn lọc mạnh (đa số video không có đủ chuỗi đúng thứ tự) → cắt sớm để khỏi phí
      graph/VLM trên frame sẽ bị temporal loại.

NGUYÊN TẮC LATENCY (đã phân tích):
  - Phần graph-DB (tra cạnh đã có sẵn) NHANH — ms — nếu LUÔN neo vào shortlist.
  - Bẫy chí mạng: để Neo4j làm coarse (global scan) → sập. Coarse là việc của
    inverted index, KHÔNG phải Neo4j. Neo4j chỉ verify cấu trúc trên tập đã nhỏ.
  - Cạnh ngữ nghĩa (holding/riding...) sinh bằng VLM là thứ DUY NHẤT không đảm bảo
    latency → để cuối, chặn budget, degrade xuống human-verify (gap honesty, Tier B).

TRẠNG THÁI (2026-07-24): bản KHUNG để tinh chỉnh. Chưa có data AIC 2026, chưa dựng
Neo4j thật. Các chỗ phụ thuộc ngoài (Faiss index, Neo4j driver, VLM) được đánh dấu
STUB rõ ràng — logic điều phối là thật, chạy được khi cắm backend vào.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol


# ============================================================ Kiểu dữ liệu I/O
# (bản rút gọn của aic/core/graph.py — xem thiết kế QueryGraph)

@dataclass
class GNode:
    id: str
    type: str                      # "entity" | "scene" | "action"
    label: str
    count: int | None = None
    attributes: list[str] = field(default_factory=list)
    evaluator: str = ""            # tool/route đã gán ở bước decompose


@dataclass
class GEdge:
    id: str
    src: str
    dst: str
    relation: str
    quantifier: str = "this"       # "this" | "existential" | "universal"
    evaluator: str = "lvlm_verify"
    is_spatial: bool = False        # True = đã precompute offline (hình học box)
    is_temporal: bool = False       # True = cạnh thứ tự sự kiện (A "sau đó" B), Allen before


@dataclass
class QueryGraph:
    nodes: list[GNode]
    edges: list[GEdge]
    raw_text: str = ""
    combine: str = "and"           # "and" (min) | "or" (max)

    def scene_nodes(self) -> list[GNode]:
        return [n for n in self.nodes if n.type == "scene"]

    def count_constraints(self) -> list[GNode]:
        return [n for n in self.nodes if n.count is not None]

    def spatial_edges(self) -> list[GEdge]:
        return [e for e in self.edges if e.is_spatial]

    def semantic_edges(self) -> list[GEdge]:
        # cạnh KHÔNG gian, KHÔNG temporal → cần VLM (holding/riding/wearing...) → đắt, cuối
        return [e for e in self.edges if not e.is_spatial and not e.is_temporal]

    def temporal_edges(self) -> list[GEdge]:
        return [e for e in self.edges if e.is_temporal]

    def temporal_chain(self) -> list[GNode]:
        """Trả chuỗi anchor theo thứ tự A -> B -> C từ các cạnh temporal (Allen before).

        Rỗng nếu query không có yếu tố thứ tự. Xử lý chuỗi tuyến tính (trường hợp phổ
        biến); nếu graph temporal phân nhánh/vòng thì fallback về thứ tự khai báo node.
        """
        tedges = self.temporal_edges()
        if not tedges:
            return []
        nxt = {e.src: e.dst for e in tedges}
        dsts = set(nxt.values())
        starts = [e.src for e in tedges if e.src not in dsts]  # node không là đích của ai
        by_id = {n.id: n for n in self.nodes}
        if len(starts) != 1:  # phân nhánh/không xác định -> fallback
            ordered_ids = [e.src for e in tedges] + [tedges[-1].dst]
            return [by_id[i] for i in dict.fromkeys(ordered_ids) if i in by_id]
        chain, cur, seen = [], starts[0], set()
        while cur in by_id and cur not in seen:
            chain.append(by_id[cur]); seen.add(cur)
            cur = nxt.get(cur)
            if cur and cur not in nxt and cur in by_id:  # node cuối chuỗi
                chain.append(by_id[cur]); break
        return chain


@dataclass
class MatchTrace:
    """Điểm từng thành phần → phục vụ giải thích subgraph (tô xanh/đỏ trên UI)."""
    node: dict[str, float] = field(default_factory=dict)
    edge: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)   # gap honesty: cạnh chưa kiểm


@dataclass
class Candidate:
    frame_id: str
    score: float
    trace: MatchTrace = field(default_factory=MatchTrace)


@dataclass
class QueryResult:
    candidates: list[Candidate]
    stages_ms: dict[str, float] = field(default_factory=dict)   # đo latency từng tầng
    degraded: bool = False                                       # có phải xuống human-verify không
    unverified_edges: list[str] = field(default_factory=list)   # cạnh chưa kiểm (báo người)


# ============================================================ Cổng backend (STUB)
# Cắm implementation thật vào 3 Protocol này. Pipeline không quan tâm bên dưới là gì.

class VectorIndex(Protocol):
    def ann_search(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """CLIP ANN toàn corpus → [(frame_id, score)]. Faiss IndexFlatIP/IVFPQ."""
        ...


class InvertedIndex(Protocol):
    def frames_with(self, label: str, min_count: int = 1) -> set[str]:
        """Tập frame có ≥min_count object 'label'. Đây là TẦNG 2 — KHÔNG dùng Neo4j."""
        ...

    def frames_with_spatial(self, rel: str, subj_label: str, obj_label: str) -> set[str]:
        """Tập frame có triple không gian (subj, rel, obj) precomputed."""
        ...


class GraphStore(Protocol):
    def verify_structure(self, shortlist: list[str], graph: QueryGraph) -> list[Candidate]:
        """TẦNG 3 — traversal NEO vào shortlist. Xem CYPHER_ANCHORED bên dưới."""
        ...


class VlmRelation(Protocol):
    def check_edge(self, frame_id: str, edge: GEdge) -> float:
        """TẦNG 4 — sinh cạnh ngữ nghĩa lazy. ĐẮT. Chỉ gọi trên shortlist tí hon."""
        ...


class FrameMeta(Protocol):
    def resolve(self, frame_id: str) -> tuple[str, float]:
        """frame_id -> (video_id, timestamp giây). Metadata frame-level, tra ms.
        Cần cho TẦNG 2.5 temporal-ordering (gom theo video + so timestamp)."""
        ...


# ============================================================ Cypher mẫu (TẦNG 3)
# LUÔN mở đầu bằng WHERE f.id IN $shortlist — đây là toàn bộ bí quyết giữ Neo4j nhanh.
# Không có dòng neo này → Neo4j quét toàn corpus → sập. Xem PROFILE để chắc không có
# CartesianProduct operator, và có index trên :Frame(id), :Object(label).

CYPHER_ANCHORED = """
// TẦNG 3: verify cấu trúc trên shortlist đã nhỏ (KHÔNG phải coarse)
MATCH (f:Frame) WHERE f.id IN $shortlist              // ← CÁI NEO, bắt buộc
// đếm ràng buộc count (vd "3 người")
MATCH (f)-[:CONTAINS]->(p:Object {label:$subj_label})
WITH f, count(p) AS n_subj                            // WITH chốt cardinality từng chặng
WHERE n_subj >= $min_count
// cạnh không gian đã precompute (vd người ON thuyền)
MATCH (f)-[:CONTAINS]->(a:Object {label:$subj_label})-[:SPATIAL {rel:$rel}]->(b:Object {label:$obj_label})
RETURN f.id AS frame_id, n_subj
"""


# ============================================================ TẦNG 2.5 · TEMPORAL

@dataclass
class TemporalHit:
    """1 video thoả chuỗi anchor, kèm frame được chọn cho từng anchor."""
    video_id: str
    chain_frames: list[str]        # frame_id chọn cho anchor 0,1,2... (đúng thứ tự thời gian)
    score: float


def temporal_join(
    anchor_candidates: list[dict[str, float]],
    fmeta: FrameMeta,
    aggregate: str = "min",
) -> list[TemporalHit]:
    """Thuật toán ANCHOR chuỗi: A -> B -> C -> ...

    Vào: anchor_candidates[k] = {frame_id: score} — frame ứng viên cho anchor thứ k
         (theo đúng thứ tự chuỗi), lấy từ coarse recall RIÊNG từng anchor.
    Ra:  các video mà TỒN TẠI 1 frame cho mỗi anchor với timestamp TĂNG DẦN (t0<t1<t2...).

    Cách chọn: greedy "sớm-khả-thi" — với mỗi anchor lấy frame SỚM NHẤT có timestamp lớn
    hơn anchor trước. Đây là cách chuẩn để KIỂM khả thi của chuỗi (chọn sớm nhất chừa
    nhiều chỗ nhất cho anchor sau). Rẻ: O(tổng số frame ứng viên).
    Điểm ở đây là SƠ BỘ (dùng để xếp video) — verify chính xác để tầng graph lo.
    """
    n = len(anchor_candidates)
    if n == 0:
        return []

    # gom ứng viên từng anchor theo video, sort theo timestamp
    per_anchor_by_video: list[dict[str, list[tuple[float, str, float]]]] = []
    for cand in anchor_candidates:
        by_v: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
        for fid, sc in cand.items():
            vid, t = fmeta.resolve(fid)
            by_v[vid].append((t, fid, sc))
        for v in by_v:
            by_v[v].sort()
        per_anchor_by_video.append(by_v)

    # chỉ xét video có ứng viên cho MỌI anchor
    common = set(per_anchor_by_video[0])
    for k in range(1, n):
        common &= set(per_anchor_by_video[k])

    hits: list[TemporalHit] = []
    for v in common:
        cursor = -math.inf
        chosen: list[tuple[float, str, float]] = []
        ok = True
        for k in range(n):
            pick = next(
                ((t, fid, sc) for (t, fid, sc) in per_anchor_by_video[k][v] if t > cursor),
                None,
            )
            if pick is None:      # không còn frame nào của anchor k sau anchor k-1 -> hỏng chuỗi
                ok = False
                break
            cursor = pick[0]
            chosen.append(pick)
        if ok:
            scores = [sc for (_, _, sc) in chosen]
            agg = min(scores) if aggregate == "min" else sum(scores) / len(scores)
            hits.append(TemporalHit(v, [fid for (_, fid, _) in chosen], agg))

    hits.sort(key=lambda h: -h.score)
    return hits


# ============================================================ Pipeline chính

def run_query(
    graph: QueryGraph,
    vec: VectorIndex,
    inv: InvertedIndex,
    store: GraphStore,
    vlm: VlmRelation | None = None,
    fmeta: FrameMeta | None = None,   # cần cho query temporal (chuỗi anchor)
    *,
    coarse_k: int = 1000,          # top-K từ vector recall
    graph_budget: int = 400,       # trần frame đưa vào Neo4j (giữ traversal nhẹ)
    vlm_budget_frames: int = 15,   # SỐ frame tối đa được gọi VLM (chặn latency)
    time_budget_s: float = 4.0,    # ngân sách thời gian tổng cho 1 lượt
) -> QueryResult:
    """Thực thi 1 query graph. Trả kết quả + đo latency từng tầng + cờ degrade.

    Thứ tự BẮT BUỘC là coarse→fine: mỗi tầng chỉ chạm tập đã nhỏ hơn tầng trước.
    Rẽ nhánh theo query có chuỗi temporal (anchor A->B->C) hay không.
    """
    t_start = time.perf_counter()
    stages: dict[str, float] = {}
    vec_scores: dict[str, float] = {}
    chain = graph.temporal_chain()

    if chain and fmeta is not None:
        # ===== LUỒNG TEMPORAL: coarse-per-anchor -> temporal-join (2.5) -> graph =====
        # TẦNG 1 · coarse recall RIÊNG từng anchor (rộng rãi, recall-oriented)
        t = time.perf_counter()
        anchor_cands: list[dict[str, float]] = []
        for anchor in chain:
            hits = dict(vec.ann_search(anchor.label, top_k=coarse_k))
            anchor_cands.append(hits)
            vec_scores.update(hits)
        stages["1_vector_per_anchor"] = (time.perf_counter() - t) * 1000

        # TẦNG 2.5 · TEMPORAL-ORDERING (rẻ + chọn lọc mạnh -> TRƯỚC graph)
        t = time.perf_counter()
        thits = temporal_join(anchor_cands, fmeta)      # thuật toán anchor chuỗi
        stages["2.5_temporal"] = (time.perf_counter() - t) * 1000

        # shortlist = frame anchor của các video sống sót (đúng thứ tự), cắt theo budget
        shortlist = []
        for h in thits:
            shortlist.extend(h.chain_frames)
            if len(shortlist) >= graph_budget:
                break
        shortlist = shortlist[:graph_budget]
    else:
        # ===== LUỒNG THƯỜNG (không thứ tự): coarse gộp =====
        # TẦNG 1 · VECTOR (toàn corpus, rẻ)
        t = time.perf_counter()
        scene_text = " ".join(n.label for n in graph.scene_nodes()) or graph.raw_text
        vec_scores = dict(vec.ann_search(scene_text, top_k=coarse_k))
        stages["1_vector"] = (time.perf_counter() - t) * 1000

        # TẦNG 2 · INVERTED INDEX (giao tập, rẻ — KHÔNG Neo4j)
        t = time.perf_counter()
        keep = set(vec_scores)
        for cn in graph.count_constraints():           # ràng buộc count (vd "≥3 người")
            keep &= inv.frames_with(cn.label, min_count=cn.count or 1)
        for e in graph.spatial_edges():                # triple không gian precompute
            keep &= inv.frames_with_spatial(e.relation, _label_of(graph, e.src), _label_of(graph, e.dst))
        stages["2_inverted"] = (time.perf_counter() - t) * 1000

        shortlist = sorted(keep, key=lambda f: -vec_scores.get(f, 0.0))[:graph_budget]

    if not shortlist:
        return QueryResult(candidates=[], stages_ms=stages)

    # ---- TẦNG 3 · GRAPH (Neo4j, NEO vào shortlist) ------------------------
    t = time.perf_counter()
    cands = store.verify_structure(shortlist, graph)   # dùng CYPHER_ANCHORED
    # bơm điểm vector vào để xếp hạng sơ bộ
    for c in cands:
        c.score = c.score or vec_scores.get(c.frame_id, 0.0)
    cands.sort(key=lambda c: -c.score)
    stages["3_graph"] = (time.perf_counter() - t) * 1000

    # ---- TẦNG 4 · LAZY VLM EDGE (đắt — chặn budget, degrade trung thực) ----
    sem_edges = graph.semantic_edges()
    result = QueryResult(candidates=cands, stages_ms=stages)

    if sem_edges and vlm is not None:
        elapsed = time.perf_counter() - t_start
        # chỉ gọi VLM trên TOP frame trong hạn budget (số frame + thời gian còn lại)
        can_afford = (
            elapsed < time_budget_s
            and len(cands) <= vlm_budget_frames
        )
        if can_afford:
            t = time.perf_counter()
            for c in cands:
                for e in sem_edges:
                    s = vlm.check_edge(c.frame_id, e)   # ĐẮT: 1 lần gọi vision
                    c.trace.edge[e.id] = s
                # hợp điểm cạnh ngữ nghĩa vào tổng (min = AND nghiêm)
                if c.trace.edge:
                    c.score = min(c.score, min(c.trace.edge.values()))
            cands.sort(key=lambda c: -c.score)
            stages["4_vlm"] = (time.perf_counter() - t) * 1000
        else:
            # DEGRADE TRUNG THỰC (gap honesty, Tier B): không cố chạy rồi trễ.
            # Trả kết quả theo tín hiệu rẻ + báo cạnh nào CHƯA kiểm để người xác nhận.
            result.degraded = True
            result.unverified_edges = [e.id for e in sem_edges]
            for c in cands:
                c.trace.notes.append(
                    f"chưa kiểm {len(sem_edges)} quan hệ ngữ nghĩa (budget) — cần nhìn xác nhận"
                )

    result.stages_ms = stages
    return result


def _label_of(graph: QueryGraph, node_id: str) -> str:
    for n in graph.nodes:
        if n.id == node_id:
            return n.label
    return ""


# ============================================================ Ghi chú tinh chỉnh
# - coarse_k / graph_budget / vlm_budget_frames / time_budget_s: 4 núm chỉnh latency.
#   Chưa có data → đặt LỎNG (recall-oriented) rồi siết sau khi đo p95 thật.
# - Muốn so Neo4j vs adjacency in-memory: chỉ cần thay class implement GraphStore,
#   pipeline không đổi. Đây là lý do tách qua Protocol.
# - Cạnh ngữ nghĩa phổ biến (person-riding-vehicle) có thể precompute offline → khi đó
#   chúng thành SPATIAL-like (is_spatial=True equiv) và chạy ở tầng 2, KHÔNG rơi xuống VLM.
#   Đây là hybrid precompute-common / lazy-rare đã bàn.

if __name__ == "__main__":
    import sys
    # console Windows mặc định cp1252 -> ép UTF-8 để in được tiếng Việt
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    # ví dụ query graph q3 (không chạy thật — backend là stub) để xem cấu trúc
    demo = QueryGraph(
        raw_text="3 người bắt ếch trên xuồng chiều tối, đều đội đèn pin, một người áo cam",
        nodes=[
            GNode("n1", "entity", "người", count=3, evaluator="objects_count"),
            GNode("n2", "scene", "trên xuồng, chiều tối", evaluator="clip_search"),
            GNode("n3", "entity", "đèn pin đội đầu", evaluator="region_search"),
            GNode("n4", "entity", "áo", attributes=["màu cam"], evaluator="region_search"),
        ],
        edges=[
            GEdge("e1", "n1", "n2", "ở trên", quantifier="this", is_spatial=True),
            GEdge("e2", "n1", "n3", "đội", quantifier="universal"),      # ∀ → VLM lazy
            GEdge("e3", "n1", "n4", "mặc", quantifier="existential"),    # ∃ → VLM lazy
        ],
    )
    print("=== q3 (không temporal) ===")
    print("scene:", [n.label for n in demo.scene_nodes()])
    print("count:", [(n.label, n.count) for n in demo.count_constraints()])
    print("spatial edges (coarse, tầng 2):", [e.relation for e in demo.spatial_edges()])
    print("semantic edges (lazy VLM, tầng 4):", [e.relation for e in demo.semantic_edges()])
    print("temporal chain:", [n.label for n in demo.temporal_chain()], "(rỗng = đúng)")

    # ---- q2 (temporal): A -> B -> C, kiểm thuật toán anchor chuỗi ----
    print("\n=== q2 (temporal, anchor A->B) ===")
    q2 = QueryGraph(
        raw_text="xe bọc thép qua sông, sau đó vụ nổ bên đường",
        nodes=[
            GNode("a1", "scene", "xe bọc thép chở binh sĩ qua sông"),
            GNode("a2", "scene", "vụ nổ bên đường"),
        ],
        edges=[GEdge("t1", "a1", "a2", "sau đó", is_temporal=True)],
    )
    print("temporal chain:", [n.label for n in q2.temporal_chain()])

    # backend giả để chạy thật thuật toán temporal_join
    class _Vec:
        def ann_search(self, text, top_k):
            # a1 khớp frame ở video v_ok (t=46) và v_bad (t=155);
            # a2 khớp v_ok (t=138) và v_bad (t=52)
            if "nổ" in text:   # a2
                return [("v_ok:138", 0.74), ("v_bad:52", 0.77), ("v_none:10", 0.6)]
            return [("v_ok:46", 0.79), ("v_bad:155", 0.71)]   # a1
    class _Meta:
        def resolve(self, fid):
            vid, t = fid.split(":")
            return vid, float(t)

    cands = [dict(_Vec().ann_search("xe", 10)), dict(_Vec().ann_search("nổ", 10))]
    hits = temporal_join(cands, _Meta())
    for h in hits:
        print(f"  {h.video_id}: chain={h.chain_frames} score={h.score:.2f}")
    print("  -> v_ok GIỮ (46<138 đúng thứ tự); v_bad LOẠI (nổ 52 TRƯỚC xe 155);")
    print("     v_none LOẠI (thiếu anchor a1). Thuật toán anchor chuỗi hoạt động.")
