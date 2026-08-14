import sys
sys.path.insert(0, "share")
sys.path.insert(0, "online")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tiers.tier1_filter import by_objects
from tiers.tier2_vector import encode_query, rank
from query_planner import _apply_region_clip_rerank, _apply_secondary_entity_boost

GT_VIDEO = "L26_V246"
GT_FRAMES = [77, 78]

candidates = by_objects(must_have_labels=["Person"], min_count={"Person": 2})
print(f"tong candidates Tang 1: {len(candidates)}\n")

# ============================================================ THI NGHIEM 2: clip_text ngan gon
print("=== THI NGHIEM 2: anh huong cua do dai clip_text len rank CLIP tho ===")
variants = {
    "full (hien tai)": "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa",
    "bo phan da hard-filter/region-clip (nguoi+mau)": "đang nhìn vào bàn tay ở giữa",
    "toi gian nhat": "nhìn vào bàn tay",
}
for name, text in variants.items():
    qv = encode_query(text)
    r = rank(qv, candidates, top_k=len(candidates)).reset_index(drop=True)
    print(f"\n[{name}] text=\"{text}\"")
    for li in GT_FRAMES:
        idx = r.index[(r.video_id == GT_VIDEO) & (r.local_idx == li)]
        if len(idx):
            i = idx[0]
            print(f"  local_idx={li}: rank={i+1}/{len(r)}, score={r.loc[i,'score']:.4f}")

# ============================================================ THI NGHIEM 1: mo rong pool Region-CLIP
print("\n\n=== THI NGHIEM 1: mo rong pool Region-CLIP co cuu duoc frame 78 (rank 737) khong ===")
qv = encode_query(variants["full (hien tai)"])
r_full = rank(qv, candidates, top_k=len(candidates)).reset_index(drop=True)

attributes = [{"entity_term": "người", "attribute_text": "người mặc áo trắng và xám"}]
secondary_entities = [{"term": "bàn tay"}]

for pool_size in [400, 2000, 5000]:
    pool = r_full.head(pool_size).copy()
    reranked = _apply_region_clip_rerank(pool, attributes, top_k=pool_size, log=None)
    reranked = _apply_secondary_entity_boost(reranked, secondary_entities, log=None)
    reranked = reranked.sort_values("score", ascending=False).reset_index(drop=True)
    final_top100 = reranked.head(100)
    for li in GT_FRAMES:
        in_pool = ((pool.video_id == GT_VIDEO) & (pool.local_idx == li)).any()
        idx = reranked.index[(reranked.video_id == GT_VIDEO) & (reranked.local_idx == li)]
        in_top100 = ((final_top100.video_id == GT_VIDEO) & (final_top100.local_idx == li)).any()
        rank_after = (idx[0] + 1) if len(idx) else None
        print(f"  pool={pool_size}: local_idx={li} - vao_pool_ban_dau={in_pool}, "
              f"rank_sau_rerank={rank_after}/{len(reranked)}, lot_top100={in_top100}")
