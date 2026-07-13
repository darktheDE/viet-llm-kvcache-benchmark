# Phân tích Kết quả Thực nghiệm

> Phần này diễn giải chi tiết từng biểu đồ trong mục Kết quả: cách đọc, con số chính, nhận định, và giải thích nguyên nhân. Số liệu lấy từ `all_results_compiled.csv` (real runs trên GPU A100 80 GB, chỉ lấy trạng thái `OK`). Trọng tâm nghiên cứu: nén KV Cache khi suy luận, trọng số mô hình giữ nguyên FP16.

## Metadata

- Người thực hiện: Huỳnh Ngọc Thạch
- Team: Data & Analysis
- Vai trò: Analytics & Plotting

---

## 0. Khung phân tích chung

Bài toán benchmark nén KV Cache là bài toán **đánh đổi ba chiều**: (i) mức nén bộ nhớ, (ii) chất lượng ngôn ngữ sinh ra, (iii) tốc độ giải mã. Không có phương pháp nào tối ưu đồng thời cả ba, nên mục tiêu phân tích là xác định *mỗi phương pháp trả giá gì để đổi lấy gì*, từ đó chỉ ra cấu hình đáng triển khai nhất cho tiếng Việt. Năm phương pháp được so sánh: FP16 (mốc chuẩn không nén), FP8, PolarQuant, HQQ, TurboQuant. Cần lưu ý ngay từ đầu: trong cấu hình vLLM hiện tại, **PolarQuant chạy qua nhánh fallback `fp8`**, nên về mặt số học nó trùng với FP8 và hai điểm luôn nằm chồng nhau ở mọi biểu đồ.

---

## 1. Bộ nhớ đo được không phản ánh mức nén (Hình 1 — VRAM vs Context)

Biểu đồ cho thấy peak VRAM của cả năm phương pháp nằm trong một dải hẹp dưới 2%, bám sát đường "pre-allocated pool (~90%)" và cách xa mốc dung lượng GPU 80 GB. Giá trị đỉnh đo được là 72 612 MB, đúng bằng ~89% của 81 920 MB. Con số này là bằng chứng trực tiếp cho cơ chế của vLLM: engine **giành trước** (pre-allocate) toàn bộ pool KV Cache theo tham số `gpu_memory_utilization ≈ 0.9` ngay khi khởi động, bất kể phương pháp nén là gì. Do đó `pynvml` đo được dung lượng của *pool tĩnh*, không phải footprint thực tế của KV Cache.

Hệ quả phương pháp luận rất quan trọng: **peak VRAM đo trực tiếp không thể dùng để so sánh các phương pháp nén trên nền tảng này.** Nén không làm giảm bộ nhớ đỉnh — nó cho phép nhét nhiều token hơn vào cùng một pool. Vì vậy biểu đồ này không được diễn giải là "nén tiết kiệm/không tiết kiệm bộ nhớ"; nó được trình bày như một *phát hiện về phương pháp đo*, và lợi ích nén thực sự được lập luận bằng con số lý thuyết ở Hình 3–4. Chênh lệch nhỏ giữa các đường (HQQ hơi cao hơn) phản ánh workspace/buffer của kernel chứ không phải KV footprint.

---

## 2. Mức nén lý thuyết và thông lượng hiệu dụng (Hình 4)

Panel trái quy đổi mức nén theo độ rộng bit: FP8 và PolarQuant đạt 2×, HQQ và TurboQuant đạt 4× so với FP16. Đây là đại lượng **giải tích**, không phụ thuộc nền tảng và không bị nhiễu bởi pool tĩnh ở Hình 1, nên nó là cách hợp lệ duy nhất để phát biểu về mức nén từ bộ dữ liệu hiện có.

Panel phải trình bày *thông lượng hiệu dụng* = throughput đo được × tỷ lệ nén, một chặn trên biểu thị năng lực phục vụ trong chế độ memory-bound (cache nhỏ hơn cho phép batch lớn hơn theo tỷ lệ). Theo tiêu chí này TurboQuant dẫn đầu ở mọi mức ngữ cảnh, nhờ vừa nén 4× vừa giữ tốc độ hợp lý. Tuy nhiên phải nhấn mạnh: **metric này bỏ qua chất lượng.** HQQ cũng được thưởng hệ số 4× nên xếp hạng cao dù đầu ra của nó suy giảm nặng (xem Hình 6). Do đó thông lượng hiệu dụng chỉ nên đọc kèm biểu đồ chất lượng, không được dùng độc lập để kết luận HQQ là lựa chọn tốt.

---

## 3. Hành vi theo chiều dài ngữ cảnh (Hình 2 — Latency / Throughput / Perplexity)

Ba panel cho thấy bức tranh nhất quán khi ngữ cảnh phình từ 4k đến 16k. Về **tốc độ**, các phương pháp 4-bit phải trả giá: HQQ tệ nhất, độ trễ tăng tới ~56 ms/token và throughput rơi xuống ~18 token/s ở 16k; TurboQuant chậm hơn baseline một cách vừa phải; còn FP8/PolarQuant bám sát FP16. Điều này phản ánh chi phí tính toán của các kernel lượng tử hóa 4-bit (dequant, xoay tọa độ, bù sai số) so với đường FP8 gần như miễn phí.

Về **chất lượng**, panel perplexity (đường median, chấm mờ là từng mẫu) là điểm đáng chú ý nhất: TurboQuant giữ perplexity phẳng quanh 1.3–1.5 ở mọi chiều dài, trong khi HQQ **sập ở mốc 16k** — median nhảy từ 1.36 (8k) lên 10.17 (16k). Đây không phải outlier lẻ: median (chứ không phải mean) xác nhận đa số mẫu HQQ ở 16k đều hỏng. Như vậy HQQ có một *chế độ hỏng phụ thuộc chiều dài ngữ cảnh* — ổn ở ngữ cảnh ngắn nhưng vỡ khi ngữ cảnh dài, đúng nơi mà nén KV Cache đáng lẽ phải phát huy tác dụng nhất.

---

## 4. Đánh đổi chất lượng–kích thước, biểu đồ cốt lõi (Hình 3 — Pareto PPL vs KV-cache size)

Đây là biểu đồ trả lời trực tiếp câu hỏi nghiên cứu. Trục X là kích thước KV Cache lý thuyết (% so với FP16), trục Y là perplexity trung bình, đo tại ngữ cảnh 16k. Kết quả rõ ràng: **TurboQuant nằm trên Pareto frontier** — với cache nhỏ hơn 4×, nó đạt perplexity 1.39, ngang hoặc tốt hơn cả baseline FP16 (1.67). Ngược lại, **HQQ bị dominated hoàn toàn**: cùng ngân sách 4-bit (25% kích thước) nhưng perplexity 10.17, tức tệ hơn TurboQuant khoảng 7 lần. FP8 và PolarQuant trùng nhau tại mốc 50% với perplexity ~2.2 — nén khá nhưng chất lượng kém hơn hẳn phương pháp 4-bit TurboQuant.

Phát hiện then chốt là **cùng một mức nén 4-bit lại cho hai kết cục trái ngược**: TurboQuant tốt nhất, HQQ tệ nhất. Điều này cho thấy chất lượng nén KV Cache không quyết định bởi số bit mà bởi *thuật toán lượng tử hóa*. TurboQuant dùng phép xoay ngẫu nhiên đưa tọa độ về phân phối tập trung rồi lượng tử hóa gần tối ưu kèm bù sai số 1-bit QJL, nhờ đó bảo toàn thông tin tốt ở 4-bit; trong khi ánh xạ `int4_per_token_head` của HQQ (giải pháp thay thế Marlin trong vLLM) không đủ để giữ chất lượng ở ngữ cảnh dài của tiếng Việt.

---

## 5. Đánh đổi chất lượng–tốc độ (Hình 5 — Pareto PPL vs Throughput)

Biểu đồ này bổ sung chiều tốc độ đo thật cho Hình 3. Frontier gồm hai cực: FP8/PolarQuant (nhanh nhất ~62 token/s nhưng perplexity ~2.6) và TurboQuant (chất lượng tốt nhất 1.39 nhưng ~46 token/s); FP16 nằm giữa. HQQ bị dominated trên cả hai trục — vừa chậm nhất vừa tệ nhất. Diễn giải thực tiễn: **TurboQuant mua chất lượng ngang baseline bằng cái giá ~25% throughput** so với nhóm 8-bit. Đây là đánh đổi hợp lý cho các kịch bản ưu tiên chất lượng đầu ra; còn nếu ưu tiên thông lượng thô và chấp nhận chất lượng thấp hơn, FP8 là lựa chọn thực dụng (nhưng không nén được nhiều như TurboQuant).

---

## 6. Chất lượng ngôn ngữ tiếng Việt (Hình 6 — Gibberish vs Repetition)

Biểu đồ tách hai loại lỗi và đối chiếu với baseline FP16 (đường nét đứt). Panel **gibberish** (tỷ lệ token rác) tách bạch đúng thủ phạm: chỉ HQQ tăng vọt lên 23.6% (so với baseline 10.1%), với 18% số mẫu hỏng nặng; TurboQuant thậm chí *thấp hơn* baseline (8.0%). Panel **repetition** (tỷ lệ lặp 3-gram) lại **phẳng đều ~30–34% ở mọi phương pháp kể cả FP16 không nén** — chứng minh hiện tượng lặp là hệ quả của greedy decoding trên tác vụ tiếp nối văn bản, *không phải* do nén gây ra.

Đây là một điểm phương pháp quan trọng cần nêu trong báo cáo: cờ `repetition_flag` nhị phân ban đầu bị **bão hòa** (bắn ở ~79% số lượt, gồm cả baseline) nên gán oan cho các phương pháp tốt như TurboQuant; ta thay bằng đo tỷ lệ liên tục đối chiếu baseline mới phản ánh đúng bản chất. Kết luận: về độ tin cậy đầu ra tiếng Việt, TurboQuant an toàn nhất, HQQ rủi ro nhất.

---

## 7. Tổng hợp và khuyến nghị

Ghép năm biểu đồ lại, thứ hạng theo từng tiêu chí như sau:

| Tiêu chí | Hạng 1 | Hạng 2 | Hạng 3 |
|----------|--------|--------|--------|
| Nén lý thuyết | TurboQuant ≈ HQQ (4×) | FP8 ≈ PolarQuant (2×) | FP16 (1×) |
| Chất lượng (perplexity + gibberish) | TurboQuant ≈ FP16 | FP8 ≈ PolarQuant | HQQ |
| Tốc độ | FP8 ≈ PolarQuant ≈ FP16 | TurboQuant | HQQ |

Kết hợp lại, **TurboQuant là phương pháp đáng triển khai nhất cho tiếng Việt**: nó là điểm Pareto-optimal về mặt nén–chất lượng, giữ chất lượng ngang mô hình gốc dù cache nhỏ 4×, và chỉ đánh đổi ~25% tốc độ. **HQQ nên bị loại**: cùng mức nén nhưng vừa chậm nhất, vừa sinh nhiều rác nhất, vừa sập ở ngữ cảnh dài. **FP8/PolarQuant** là phương án trung dung an toàn khi ưu tiên tốc độ và chỉ cần nén 2×.

Thông điệp bao trùm: ở cùng một ngân sách bit, chất lượng nén KV Cache được quyết định bởi thuật toán chứ không bởi độ rộng bit — đó là đóng góp chính của khảo sát này khi áp lên đặc thù token hóa của tiếng Việt.

---

## 8. Hạn chế và hướng khắc phục

Cần nêu thẳng các hạn chế để bảo đảm tính trung thực học thuật:

1. **Peak VRAM đo được bị pool tĩnh che lấp** (Mục 1); muốn đo tiết kiệm bộ nhớ thực phải chạy lại với `gpu_memory_utilization` thấp/cố định hoặc đọc metric KV usage của vLLM.
2. **Perplexity đo trên chính văn bản model tự sinh** dưới greedy decoding, nên nó thưởng cho văn bản lặp và chỉ là chỉ báo tương đối giữa các phương pháp, không phải điểm chất lượng tuyệt đối; hướng tốt hơn là đo perplexity trên corpus tiếng Việt giữ ngoài.
3. **PolarQuant chạy fallback = FP8**, chưa được đánh giá như một kernel low-bit riêng.
4. **Cỡ mẫu nhỏ** (n = 8–15 mỗi phương pháp) và độ phủ model × method chưa đầy đủ, nên các khác biệt trong cụm FP8/PolarQuant/FP16 chưa đủ mạnh để kết luận.
5. Mốc **32k bị loại** vì vượt cửa sổ ngữ cảnh (~26k) của vài mô hình — cần chọn mô hình hỗ trợ 32k nếu muốn khảo sát giới hạn nén ở ngữ cảnh siêu dài.
6. Cần **xác nhận độ rộng bit của TurboQuant** (4-bit → 4×, 3-bit → ~5.3×) để cập nhật đúng Hình 3–4.
