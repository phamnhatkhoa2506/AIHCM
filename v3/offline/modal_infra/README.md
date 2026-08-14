# Hạ tầng Modal + vLLM cho Phase P1 (trích xuất relation)

Deploy **Qwen2.5-VL-7B-Instruct** qua vLLM trên Modal (A10G), expose OpenAI-compatible
endpoint — dùng để Tầng 4 (graph) gọi VLM xác nhận relation cho các cặp object đã qua gate
L0-L3 (`tiers/pair_gate.py`).

Đây **chỉ là setup hạ tầng** (deploy model + kiểm kết nối đúng schema) — batch job xử lý toàn
bộ 156,965 frame (checkpoint/resume, ghi `semantic_edges.parquet`) là bước riêng, làm sau khi
hạ tầng này chạy ổn.

## Cài đặt

```bash
pip install modal requests
modal setup          # đăng nhập/tạo token lần đầu, mở trình duyệt xác thực
```

## Chạy thử (dev, hot-reload khi sửa code)

```bash
cd v3/modal_infra
modal serve app.py
```

In ra 1 URL tạm (dạng `https://<workspace>--aic2026-qwen25vl-p1-server-dev.modal.run`) — giữ
terminal này chạy, mở terminal khác để test.

## Deploy thật (URL cố định, không tắt khi đóng terminal)

```bash
modal deploy app.py
```

## Kiểm tra hạ tầng sống (health check thuần, không cần data)

```bash
modal run app.py
```

## Test đúng schema P1 (dùng frame thật từ v3, gửi ảnh + cặp qua gate, kiểm JSON trả về)

```bash
python test_client.py <url_vua_deploy>
# hoặc chỉ định frame cụ thể:
python test_client.py <url_vua_deploy> L21_V001 7
```

Script tự đọc `objects_index.parquet` + `pair_gate.gate_pairs` để dựng request thật (không
phải request giả) — nếu chạy được và JSON trả về đúng schema (`relations: [{subj_id, obj_id,
relation, conf, inferred, evidence}, ...]`), hạ tầng coi như sẵn sàng cho bước batch job tiếp theo.

## Các tham số đáng chỉnh trong `app.py`

- `MAX_CONTAINERS` — số container (≈ số "GPU song song") chạy đồng thời. Để thấp (2-4) khi
  pilot, tăng lên khi chạy full batch (xem công thức thời gian đã bàn: tổng thời gian ≈ số
  frame × thời gian/frame / số container).
- `target_concurrency` — số request đồng thời/container trước khi Modal tự scale thêm
  container mới.
- `unauthenticated=True` — đang tắt xác thực cho dev; **bật lại trước khi để chạy lâu dài**
  nếu endpoint có thể bị truy cập từ ngoài.

## Việc CHƯA làm ở đây (thuộc phạm vi khác, không phải infra)

- `box_to_position()` — mô tả vị trí object bằng ngôn ngữ tự nhiên thay toạ độ số (tránh nhầm
  object cùng nhãn) — `test_client.py` hiện gửi thẳng object list không kèm mô tả vị trí, chỉ
  để test đường ống, chưa phải prompt P1 hoàn chỉnh.
- Batch driver xử lý toàn corpus (checkpoint/resume, ghi `semantic_edges.parquet`) — viết
  trong `v3/` chính, không phải `modal_infra/`.

## Ghi chú

Cú pháp `app.py` dựa theo `modal-examples` chính thức
(github.com/modal-labs/modal-examples/06_gpu_and_ml/llm-serving/vllm_inference.py, kiểm tra
lại 2026-08-05) — API của Modal đổi khá nhanh, nếu lệch so với thực tế thì đối chiếu lại link
đó trước khi debug thay vì đoán.
