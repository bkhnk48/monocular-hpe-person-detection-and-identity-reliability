# MC-MCF: Multi-Commodity Min-Cost Flow cho gán detection người 3D

## 1. Giới thiệu

Dự án xây dựng một framework tối ưu hóa việc gán các detection người 3D từ WHAM cho các đối tượng Ground Truth (GT), sử dụng mô hình **Multi-Commodity Min-Cost Flow (MC-MCF)** trên mạng thời gian mở rộng (*time-expanded network*).

Mục tiêu là lựa chọn detection phù hợp cho từng người GT qua các frame, đồng thời:

- Giảm sai số hình học giữa pose dự đoán và pose GT.
- Hạn chế bỏ sót người GT (*false negatives*).
- Phạt các detection không được sử dụng (*false positives*).
- Khuyến khích tính nhất quán theo thời gian của sai số pose.
- Ngăn một detection bị gán đồng thời cho nhiều người GT.

Khác với cách lựa chọn detection độc lập tại từng frame, framework xem quá trình gán detection là một bài toán tối ưu hóa trên toàn bộ chuỗi thời gian.

Framework hỗ trợ cả trường hợp một người GT và nhiều người GT. Với nhiều người, các đường đi của từng người được liên kết thông qua ràng buộc dung lượng detection dùng chung.

## 2. Hướng dẫn sử dụng

Hướng dẫn thực hành chi tiết về cài đặt môi trường, chuẩn bị dữ liệu, cấu hình và chạy các chương trình MC-MCF được trình bày trong tài liệu riêng:

**[MC-MCF – Hướng dẫn cài đặt và chạy chương trình](https://docs.google.com/document/d/1fUIUKLVQOlAJ9S-xI8ahBLh3dZfIocNxEP8fE2JtR_8/edit?usp=sharing)**

## 3. Ý tưởng phương pháp

### 3.1. Đối tượng và detection

Ký hiệu:

- `K`: số người GT.
- `t`: chỉ số frame.
- `d`: một detection từ WHAM.
- `k`: một người GT.
- `PA_tkd`: PA-MPJPE giữa detection `d` và người GT `k` tại frame `t`.

Mỗi detection được xác định bằng cặp `(frame, body_id)`, tương ứng với `(t, d)`.

Tại mỗi frame mà người GT `k` xuất hiện, mạng tạo các trạng thái:

1. **Detected:** gán một detection WHAM cho người GT.
2. **Missed:** không có detection nào được gán cho người GT tại frame đó.

Mỗi người GT được mô hình hóa bằng một luồng đơn vị (*unit flow*) đi từ source đến sink qua các lớp frame tương ứng.

### 3.2. Mạng time-expanded

Với mỗi người GT, mạng có cấu trúc:

```text
s_k -> V_1,k -> V_2,k -> ... -> V_T,k -> t_k
```

Trong đó:

- `s_k`: source của người GT `k`.
- `t_k`: sink của người GT `k`.
- `V_t,k`: tập trạng thái của người `k` tại frame `t`.

Các trạng thái giữa hai frame liên tiếp trong danh sách frame của người GT được kết nối bằng các cung chuyển tiếp. Mỗi cung chuyển tiếp lưu thêm khoảng cách frame gốc `Δt` giữa hai lớp (bằng 1 nếu hai frame liền nhau).

Các loại cung gồm:

| Cung | Ý nghĩa |
|---|---|
| `start_det` | Source → Detected |
| `start_miss` | Source → Missed |
| `dd` | Detected → Detected |
| `dm` | Detected → Missed |
| `md` | Missed → Detected |
| `mm` | Missed → Missed |
| `end_det` | Detected → Sink |
| `end_miss` | Missed → Sink |
| `empty` | Source → Sink (người không có frame GT nào) |

Các chi phí của nút và cung được lưu dưới dạng các đại lượng thô (PA-MPJPE, RMS residual, `Δt`, loại cung) và được tính lại bằng `apply_params()` khi thay đổi bộ tham số, không cần dựng lại mạng.

### 3.3. Ràng buộc dùng chung detection

Trong trường hợp nhiều người GT, mỗi người có một luồng riêng. Tuy nhiên, các luồng cùng sử dụng tập detection WHAM.

Một detection chỉ được gán cho tối đa một người GT:

```text
sum_k x_tkd <= 1
```

Trong đó `x_tkd = 1` nếu detection `d` tại frame `t` được gán cho người `k`.

Đây là ràng buộc liên kết các luồng và tạo nên bản chất multi-commodity của bài toán.

## 4. Hàm mục tiêu

Hàm mục tiêu kết hợp chi phí pose, false positive, false negative, tính nhất quán theo thời gian, chi phí chuyển trạng thái và chi phí biên.

### 4.1. Chi phí pose và gán detection

Với detection `d` được gán cho người GT `k` tại frame `t`, chi phí nút là:

```text
C_det(t, k, d) = α × PA_tkd / s_P - C_FP
```

Trong đó:

- `PA_tkd`: PA-MPJPE, tính bằng mm.
- `α`: trọng số pose.
- `s_P`: hệ số chuẩn hóa pose, mặc định 100 mm.
- `C_FP`: chi phí false positive, mặc định 0.8.

Chi phí trạng thái missed:

```text
C_miss = C_FN
```

Trong đó `C_FN` là chi phí false negative, mặc định bằng 1.0.

**Giải thích cách tính false positive**

Code sử dụng `-C_FP` trong chi phí của mỗi detection được gán và cộng thêm một hằng số:

```text
C_FP × N_D
```

với `N_D` là tổng số detection trong dữ liệu đầu vào.

Do đó:

```text
sum_assigned [α × PA_d / s_P - C_FP] + C_FP × N_D
= sum_assigned [α × PA_d / s_P] + C_FP × N_FP
```

với `N_FP` là số detection không được sử dụng.

Cách biểu diễn này giúp đưa chi phí false positive vào mô hình dưới dạng chi phí nút tuyến tính, trong khi vẫn giữ nguyên ý nghĩa của hàm mục tiêu.

### 4.2. Chi phí nhất quán theo thời gian

Với hai trạng thái Detected liên tiếp `a` và `b` trên đường đi của cùng một người GT, framework tính chi phí dựa trên sự thay đổi của residual sau Procrustes alignment.

Residual tại frame `t`, khớp `j`:

```text
r_t,j = X_hat_aligned_t,j - Y_GT_t,j
```

Trong đó:

- `X_hat_aligned_t,j`: khớp dự đoán sau khi căn chỉnh với GT.
- `Y_GT_t,j`: khớp tương ứng trong GT.

Độ thay đổi residual giữa hai trạng thái:

```text
RMS(r_a, r_b) = sqrt((1 / J) × sum_j ||r_b,j - r_a,j||_2^2)
```

Chỉ các khớp hợp lệ ở cả hai frame mới được sử dụng. Nếu có ít hơn 8 khớp hợp lệ, hàm trả về 0.

Chi phí temporal:

```text
C_temporal(a, b) = γ × RMS(r_a, r_b) / (s_M × (Δt_ab)^p)
```

Trong đó:

- `γ`: trọng số temporal.
- `s_M`: hệ số chuẩn hóa temporal, mặc định 0.05 m.
- `Δt_ab >= 1`: số frame gốc giữa hai lớp `a` và `b`.
- `p`: `gap_power`, mặc định 1. Với `p = 1`, chi phí đo *tốc độ* thay đổi residual trên mỗi frame, nên một khoảng trống 10 frame không bị phạt như một bước nhảy giữa hai frame liền nhau. Đặt `p = 0` để bỏ qua `Δt`.

Khi mọi frame GT liền nhau (`Δt = 1`), công thức trùng với công thức không có `Δt`. Trên các sequence đã gán nhãn (MPI S8 cam0/cam8, Harmony4D 041 cam07), mọi `Δt` đều bằng 1.

**Lưu ý về ý nghĩa của temporal cost**

Đây là chi phí đo sự thay đổi của sai số residual sau Procrustes alignment độc lập ở từng frame. Nó không trực tiếp đo vận tốc, độ dịch chuyển hay chuyển động vật lý của cơ thể.

Vì vậy, trong framework này, thuật ngữ *temporal residual consistency* được sử dụng thay cho *physical motion smoothness*.

**Độ dịch 2D giữa hai detection (`C_V`, tuỳ chọn)**

`C_temporal` không mang thông tin danh tính, vì Procrustes đã bỏ vị trí. `C_V` bổ sung chính thông tin đó: vị trí trên ảnh của hai detection liên tiếp trên đường đi của cùng một người GT. Cùng một người thì dịch ít; nhảy sang người khác thì dịch cỡ khoảng cách giữa hai người. `C_V` không dùng GT.

Mỗi detection được chiếu lên ảnh bằng đúng camera mà WHAM giả định (`f = sqrt(W² + H²)`, điểm chính ở tâm ảnh), lấy 12 khớp SMPL ứng với `LIMBS`: `u_a,j` (px). Cỡ người `S(u)` là cạnh dài hơn của khung bao các khớp, tối thiểu 40 px.

```text
D(a, b) = sqrt((1 / J) × sum_j ||u_b,j - u_a,j||_2^2) / ((S(u_a) + S(u_b)) / 2)

C_V(a, b) = η × D(a, b) / (s_V × (Δt_ab)^p)
```

Trong đó:

- `η`: trọng số độ dịch 2D, mặc định 0 (tắt; kết quả trùng hoàn toàn với khi chưa có `C_V`).
- `s_V`: hệ số chuẩn hóa, mặc định 0.05 cỡ người.
- `Δt_ab`, `p`: như của `C_temporal`.

Chỉ các khớp hợp lệ ở cả hai detection được dùng; ít hơn 8 khớp thì `D = 0`. Toạ độ ảnh được lưu ở `outputs/mapping/<sequence>/WHAM-2D.csv` (cột `frame`, `body_id`, `<khớp>_u`, `<khớp>_v`). Lần đầu tạo cần pkl của WHAM (pkl trên Drive được tải lại một lần) và khoá `"image_size": [W, H]` của sequence trong `configs/splits.json`.

### 4.3. Chi phí chuyển trạng thái

Các trạng thái trên đường đi có các chi phí chuyển tiếp:

| Chuyển trạng thái | Chi phí |
|---|---:|
| Detected → Detected | `C_temporal + C_V` |
| Detected → Missed | `C_disappear` |
| Missed → Detected | `C_reappear` |
| Missed → Missed | 0 |

Trong đó:

- `C_disappear`: chi phí khi một người chuyển từ trạng thái detected sang missed.
- `C_reappear`: chi phí khi một người chuyển từ missed sang detected.

`C_disappear` và `C_reappear` là chi phí cho một *sự kiện* đổi trạng thái nên không chia theo `Δt`.

### 4.4. Chi phí biên

Mạng cho phép cấu hình riêng chi phí tại đầu và cuối chuỗi:

```text
C_start = C_start_det, nếu trạng thái đầu là detected
C_start = C_start_miss, nếu trạng thái đầu là missed
```

Tương tự:

```text
C_end = C_end_det, nếu trạng thái cuối là detected
C_end = C_end_miss, nếu trạng thái cuối là missed
```

Các chi phí biên mặc định đều bằng 0.

### 4.5. Hàm mục tiêu tổng thể

Có thể biểu diễn hàm mục tiêu dưới dạng:

```text
min L =
    sum_t,k,d x_tkd × (α × PA_tkd / s_P - C_FP)
  + sum_t,k m_tk × C_FN
  + sum_(a,b) in E_dd (C_temporal(a,b) + C_V(a,b))
  + N_dm × C_disappear
  + N_md × C_reappear
  + C_boundary
  + C_FP × N_D
```

Trong đó:

- `m_tk = 1` nếu người `k` bị missed tại frame `t`.
- `E_dd`: tập các cung Detected → Detected được sử dụng.
- `N_dm`: số cung Detected → Missed được sử dụng.
- `N_md`: số cung Missed → Detected được sử dụng.
- `C_boundary`: tổng chi phí biên.
- `N_D`: tổng số detection đầu vào.

Hàm mục tiêu được tối thiểu hóa với các ràng buộc bảo toàn luồng và dung lượng detection dùng chung.

## 5. Mô hình tối ưu hóa MC-MCF

### 5.1. Ràng buộc bảo toàn luồng

Với mỗi người GT, luồng phải đi từ source đến sink.

Tại các nút trung gian, tổng luồng đi vào bằng tổng luồng đi ra.

Với biến luồng trên cung `a`, ký hiệu `f_a`, ràng buộc bảo toàn luồng được viết:

```text
sum_{a in δ-(v)} f_a - sum_{a in δ+(v)} f_a = b_v
```

Trong đó:

```text
b_v = -1, nếu v là source
b_v =  1, nếu v là sink
b_v =  0, với các nút khác
```

Mỗi người GT có một đơn vị luồng từ source đến sink.

### 5.2. Ràng buộc dung lượng detection

Với mỗi detection `(t, d)`:

```text
sum_k x_tkd <= 1
```

Ràng buộc này bảo đảm một detection WHAM không thể được sử dụng đồng thời bởi nhiều người GT.

### 5.3. Biến quyết định

Các biến luồng trên cung là biến nhị phân:

```text
f_a ∈ {0, 1}
```

Vì mỗi luồng đi qua một đường duy nhất trong mạng phân lớp, các biến này biểu diễn việc lựa chọn đường đi tương ứng với từng người GT.

### 5.4. Công thức bài toán MILP

Sau khi xây dựng các chi phí nút và cung, bài toán được biểu diễn dưới dạng:

```text
min_f sum_{a in A} w_a × f_a + C_FP × N_D
```

Với các ràng buộc:

```text
A × f = b
B × f <= 1
f_a ∈ {0, 1}
```

Trong đó:

- `A`: ma trận ràng buộc bảo toàn luồng.
- `B`: ma trận ràng buộc dung lượng detection.
- `w_a`: trọng số của cung, bao gồm chi phí cung và chi phí nút đích.
- `C_FP × N_D`: hằng số được cộng khi tính objective đầy đủ.

## 6. Chiến lược giải bài toán

Framework sử dụng hai phương pháp: Dynamic Programming (DP) và Mixed-Integer Linear Programming (MILP).

### 6.1. Single-commodity: Dynamic Programming

Với một người GT, bài toán trở thành tìm đường đi ngắn nhất trên mạng phân lớp dạng DAG.

Do các cung được tạo theo thứ tự topo, thuật toán có thể duyệt lần lượt các cung để cập nhật chi phí đường đi ngắn nhất.

Độ phức tạp của bước tìm đường đi tỷ lệ tuyến tính với số nút và số cung:

```text
O(|V| + |A|)
```

Đây là nghiệm tối ưu chính xác cho trường hợp một người GT trong mô hình hiện tại.

### 6.2. Multi-commodity: DP kết hợp MILP

Khi có nhiều người GT, framework trước tiên giải độc lập từng luồng bằng DP, tức tạm thời bỏ qua ràng buộc dung lượng detection dùng chung.

Sau đó, kiểm tra các đường đi vừa tìm được:

- Nếu không có detection nào được nhiều hơn một người sử dụng, nghiệm DP thỏa mãn ràng buộc dùng chung. Vì nghiệm độc lập đã tối ưu cho từng luồng và đồng thời khả thi đối với bài toán kết hợp, nghiệm này cũng là nghiệm tối ưu của bài toán MC-MCF.
- Nếu có detection bị nhiều người cùng sử dụng, framework chuyển sang giải bài toán MILP với đầy đủ ràng buộc dung lượng detection.

MILP được giải bằng `scipy.optimize.milp`, sử dụng bộ giải HiGHS. Thời gian giải mặc định được giới hạn ở 300 giây (`--time-limit`).

**Kiểm tra trạng thái tối ưu**

Sau mỗi lần giải, thông tin bộ giải được ghi vào `summary.json` (khóa `solver`):

| Khóa | Ý nghĩa |
|---|---|
| `method` | `dp (tối ưu chính xác)` hoặc `milp (HiGHS)` |
| `optimal` | `true` nếu nghiệm là tối ưu đã chứng minh (DP, hoặc HiGHS `status = 0`) |
| `milp.success`, `milp.status`, `milp.status_name`, `milp.message` | trạng thái HiGHS: 0 `optimal`, 1 `limit_reached`, 2 `infeasible`, 3 `unbounded`, 4 `other` |
| `milp.mip_gap` | khoảng cách tương đối giữa nghiệm và cận dưới |
| `milp.mip_node_count` | số nút nhánh cận đã duyệt |
| `milp.mip_dual_bound` | cận dưới tốt nhất |

Nếu MILP dừng vì hết thời gian nhưng vẫn có nghiệm khả thi, nghiệm vẫn được dùng nhưng `optimal = false`, chương trình in cảnh báo, và `pose_assoc.tune` đếm số sequence như vậy trong cột `non_optimal` của bảng kết quả. Nếu HiGHS không trả về nghiệm nào, chương trình dừng với lỗi.

## 7. Các tham số mặc định

| Tham số | Giá trị mặc định | Ý nghĩa | Khóa trong `configs/default.yaml` (mục `association`) |
|---|---:|---|---|
| `α` | 1.0 | Trọng số pose | `weights.pose` |
| `γ` | 0.5 | Trọng số temporal residual | `weights.temporal` |
| `C_FP` | 0.8 | Phạt false positive | `costs.false_positive` |
| `C_FN` | 1.0 | Phạt false negative | `costs.false_negative` |
| `C_disappear` | 0.15 | Phạt Detected → Missed | `costs.disappear` |
| `C_reappear` | 0.20 | Phạt Missed → Detected | `costs.reappear` |
| `s_P` | 100 mm | Pose normalization scale | `scales.pose_mm` |
| `s_M` | 0.05 m | Temporal normalization scale | `scales.motion_m` |
| `p` | 1.0 | Số mũ của `Δt` trong temporal cost (và `C_V`) | `temporal.gap_power` |
| `η` | 0.0 | Trọng số độ dịch 2D `C_V` (0 = tắt) | `weights.displacement` |
| `s_V` | 0.05 | Displacement normalization scale (theo cỡ người trên ảnh) | `scales.displacement` |
| `C_start_det` | 0.0 | Chi phí đầu chuỗi, detected | `boundary.start_detected` |
| `C_start_miss` | 0.0 | Chi phí đầu chuỗi, missed | `boundary.start_missed` |
| `C_end_det` | 0.0 | Chi phí cuối chuỗi, detected | `boundary.end_detected` |
| `C_end_miss` | 0.0 | Chi phí cuối chuỗi, missed | `boundary.end_missed` |

Các tham số được tổ chức trong `CostParams` (`pose_assoc/costs.py`) và được nạp từ cấu hình thông qua `from_config()`. Bảng này được kiểm tra tự động với `CostParams` bởi `tests/test_readme_sync.py`.

## 8. Dữ liệu và tiền xử lý

### 8.1. Bộ khớp sử dụng

Framework sử dụng 12 khớp chung giữa các định dạng pose:

- Left shoulder, right shoulder
- Left elbow, right elbow
- Left wrist, right wrist
- Left hip, right hip
- Left knee, right knee
- Left ankle, right ankle

Số khớp hợp lệ tối thiểu để thực hiện Procrustes alignment là 8.

### 8.2. Chuẩn hóa đơn vị

Dữ liệu GT của MPI-INF-3DHP được lưu bằng millimeter, trong khi dữ liệu WHAM được lưu bằng meter.

Trước khi tính sai số, GT được chuyển đổi sang meter:

```text
X_GT (m) = X_GT (mm) / 1000
```

PA-MPJPE được tính bằng meter trong quá trình alignment và được chuyển sang millimeter khi lưu vào mạng.

### 8.3. Procrustes alignment

Dự đoán được căn chỉnh với GT bằng phép biến đổi similarity gồm:

- Translation.
- Rotation.
- Uniform scaling.

Sau khi căn chỉnh, PA-MPJPE được tính bằng trung bình khoảng cách Euclidean giữa các khớp hợp lệ.

Residual sau alignment được lưu lại để tính temporal residual consistency cost.

## 9. Cấu trúc mã nguồn

| Module | Chức năng |
|---|---|
| `pose_assoc/costs.py` | Tham số chi phí `CostParams`, Procrustes alignment, PA-MPJPE, temporal residual cost, độ dịch 2D `C_V` (`displacement_2d()`, `displacement_cost()`) |
| `pose_assoc/network.py` | Xây dựng mạng time-expanded, các nút, cung và trọng số; `apply_params()`; `attach_positions()` gắn độ dịch 2D thô (`arc_disp`) cho các cung |
| `pose_assoc/solvers/single_mcf.py` | Giải đường đi ngắn nhất độc lập bằng dynamic programming |
| `pose_assoc/solvers/multi_mcf.py` | Kiểm tra xung đột detection, giải MILP khi cần, ghi trạng thái HiGHS |
| `pose_assoc/evaluation.py` | Bảng gán, phân rã hàm mục tiêu, chấm điểm so với mapping gán tay. Hai định dạng nhãn, không dùng lẫn: `load_truth_by_person()` cho `evaluate()` (build_network) và tune; `load_truth()` cho `evaluate_assignment()` (`pose_assoc/compare.py`). `evaluate()` từ chối nhãn sai định dạng (`tests/test_evaluation_truth.py`) |
| `pose_assoc/build_network.py` | CLI: dựng mạng, giải và xuất kết quả cho các sequence trong manifest; `--split` bỏ qua sequence `"skip": true` (`select_sequences()`); `load_positions()` đọc toạ độ ảnh WHAM khi `η > 0` |
| `pose_assoc/tune.py` | CLI: hiệu chỉnh tham số tích lũy qua nhiều lần trên tập validation; không thêm hay cập nhật sequence `"skip": true`, thêm bước kiểm tra GT đã đổi. Cache luôn có độ dịch 2D; cache dựng trước `NET_VERSION` hiện tại không được trộn vào (`load_cached()` dừng và chỉ cách `--exclude`). |
| `pose_assoc/baselines.py` | Các baseline trên cùng mạng: B6 giữ ID gốc của WHAM (đa số phiếu), B1 tham lam theo frame, B2 Hungarian theo frame, B3 đường đi ngắn nhất độc lập (không ràng buộc dùng chung detection) |
| `pose_assoc/compare.py` | CLI: so sánh B6/B1/B2/B3 với B4 (MC-MCF) trên các sequence đã gán nhãn của một tập, ghi `outputs/compare/<split>/table.csv` kèm macro-average |
| `scripts/wham_gt_mapping.py` | CLI: gán nhãn tay body_id WHAM ↔ người GT (cả theo đoạn frame); đọc GT `mpi`, `harmony4d`, `mpi_overlay`, `mpi_overlay_csv` và GT bằng fileID; `output_wham_video` nhận đường dẫn hoặc fileID (tải một lần vào `outputs/mapping/<tên>/output_wham_<id>.mp4` khi ghép video, `fetch_output_wham_video()`); `load_wham_2d()` chiếu WHAM lên ảnh (`WHAM-2D.csv`) cho `C_V` |
| `scripts/make_overlay_pair.py` | CLI: ghép 2 đoạn video MPI 1 người (`--x`/`--y START:END`, `--x-video`) thành video 2 người: X cỡ gốc đứng trước, Y thu nhỏ quanh tâm ảnh đứng sau (mặt nạ chroma key ∩ khung xương 2D); `--mode generate`/`render` để sửa tay từng frame PNG rồi ghép lại MP4 |
| `scripts/overlay_autolabel.py` | CLI: gán nhãn tự động cho video ghép theo vị trí 2D (không dùng PA-MPJPE); mẫu kiểm chứng tay theo mức che |
| `scripts/check_wham_camera.py` | CLI: so hệ toạ độ camera của WHAM (focal heuristic) với GT camera thật, trước và sau khi quy đổi; video MPI đơn (`--annot`, `--camera`) hoặc video ghép (`--meta`, GT từ `gt_3d_scene.csv`) |
| `scripts/solve_mc_mcf.py` | Script shortest-path độc lập cho trường hợp một người GT |
| `scripts/help` | In hướng dẫn nhanh: tạo mapping, tạo video overlay, hiệu chỉnh tham số, giải trên tập test (`python3 scripts/help [mapping\|overlay\|tune\|test]`); các tuỳ chọn được đối chiếu với `--help` của từng CLI trong `tests/test_help.py` |
| `configs/splits.json` | Manifest các sequence validation / test |
| `configs/default.yaml` | Tham số mặc định (mục `association`) |

`solve_mc_mcf.py` là một script giải bài toán một người GT theo cách riêng. Nó hữu ích để kiểm tra và đối chiếu với trường hợp single-commodity, nhưng không thay thế bộ xây dựng mạng và solver multi-commodity của framework.

## 10. Cài đặt và chạy

Cần Python ≥ 3.9. Các lệnh dưới đây chạy từ thư mục gốc của repo.

```bash
python3 -m pip install --user -r requirements.txt
```

Xem nhanh các bước bên dưới cùng trạng thái hiện tại (sequence nào đã gán nhãn, lần hiệu chỉnh mới nhất). Thêm `mapping`, `overlay`, `tune` hoặc `test` để chỉ in một phần:

```bash
python3 scripts/help
```

Gán nhãn tay (hỏi validation hay test, bỏ qua sequence đã có `mapping.txt`):

```bash
python3 scripts/wham_gt_mapping.py
```

Dựng mạng và giải cho một sequence hoặc cả một tập:

```bash
python3 -m pose_assoc.build_network --seq h4d_041_grappling2_cam07
```

```bash
python3 -m pose_assoc.build_network --split validation
```

Sequence có `"skip": true` trong `configs/splits.json` bị bỏ qua khi gán nhãn, khi chạy theo `--split` và khi hiệu chỉnh. Gọi đích danh bằng `--seq` thì `build_network` vẫn chạy và in lưu ý. Với `pose_assoc.tune`, `skip` chỉ ngăn thêm hoặc cập nhật sequence. Sequence đã đăng ký từ lần trước thì dữ liệu đã lưu vẫn được dùng; muốn bỏ hẳn thì dùng `--exclude <tên>`.

Kết quả nằm ở `outputs/network/<sequence>/` (`nodes.csv`, `arcs.csv`, `assignment.csv`, `summary.json`). Có thể ghi đè tham số bằng `--alpha`, `--gamma`, `--eta`, `--c-fp`, `--c-fn`, `--c-disappear`, `--c-reappear`, hoặc dùng `--config` với một file YAML.

Bật độ dịch 2D `C_V` (mục 4.2). Lần đầu mỗi sequence sẽ tạo `WHAM-2D.csv` từ pkl, nên cần `"image_size"` trong manifest; `arcs.csv` có thêm cột `disp_2d` và phân rã hàm mục tiêu có thêm mục `displacement`:

```bash
python3 -m pose_assoc.build_network --seq h4d_041_grappling2_cam07 --eta 0.5
```

Hiệu chỉnh tham số trên các sequence validation đã gán nhãn:

```bash
python3 -m pose_assoc.tune
```

Bộ tham số của mỗi lần hiệu chỉnh được ghi ở `configs/tuning/round_<n>.yaml` và có thể dùng lại qua `--config`.

Quét thêm `η` bằng `--grid eta=...`. Cache của tune luôn có độ dịch 2D. Sequence đăng ký trước khi có `C_V` được dựng lại nếu còn trong manifest; nếu chỉ còn trong sổ đăng ký thì tune dừng và in lệnh `--exclude` cho các sequence đó:

```bash
python3 -m pose_assoc.tune --grid eta=0,0.25,0.5,1,2 --grid gamma=0 --force
```

So sánh MC-MCF (B4) với các baseline trên cùng mạng và cùng nhãn tay. Chỉ chạy trên tập test sau khi đã chốt bộ tham số trên validation:

```bash
python3 -m pose_assoc.compare --split test --config configs/tuning/round_4.yaml
```

Hai chương trình dùng hai định nghĩa `accuracy` khác nhau, không so trực tiếp với nhau:

- `build_network` và `tune`: `(correct + correct_miss) / (frame × người)`.
- `compare`: `correct / (correct + wrong_id + missed)`, tức là không tính các ô bỏ trống đúng. Ví dụ trên `h4d_041_grappling2_cam07` với round 4: cùng số đếm 743 / 79 / 66 / 6, `build_network` ra 0.9195 còn `compare` ra 0.9117.

### 10.1. Video 2 người tổng hợp từ MPI-INF-3DHP

Ghép hai video 1 người X, Y theo chiều sâu: X giữ cỡ gốc và luôn đứng trước; người trong Y (có mặt nạ) được thu nhỏ hệ số `s` quanh tâm ảnh (mặc định `s = 0.5`, diện tích 1/4) và đứng sau. Mỗi người giữ nguyên chuyển động của mình; số frame là độ dài ngắn hơn trong hai video. Với cùng intrinsics, ảnh thu nhỏ quanh điểm chính `(c_x, c_y)` xấp xỉ ảnh của chính thân người Y lùi ra sau: giữ toạ độ ngang và độ cao, độ sâu `Z` thành `Z / s`. Đây không phải lùi theo tia nhìn, vì lùi theo tia nhìn thì ảnh không đổi. Sai số phối cảnh trên S8 cam0 có trung vị 1.6% chiều cao người. Vì độ cao giữ nguyên nên với camera gần nằm ngang, chân Y vẫn ở trên sàn, và người xa hơn luôn bị người gần che. Script thu nhỏ quanh tâm ảnh. Đó cũng là điểm chính mà mạng WHAM giả định, nên với WHAM phép thu nhỏ này nhất quán hoàn toàn. So với camera thật (cam0 có `c_y = 1051.4`), Y nằm cao hơn khoảng 14 px; độ lệch này là hằng số đã biết, có thể bù khi biến đổi GT.

Mỗi người lấy một đoạn frame `START:END` (không lấy `END`) của một video nguồn: `--x 0:1500 --x-video <video>` và `--y 4750:6250`. Bỏ `--y-video` thì Y lấy từ chính video của X. Subject và camera (để đọc `annot.mat`) được suy ra từ đường dẫn dạng `.../<SUBJECT>/Seq1/imageSequences/video_cam<CAMERA>.avi`. Đường dẫn khác dạng này thì chỉ rõ bằng `--x-subject`, `--x-camera`, `--y-subject`, `--y-camera`. `--annot-pattern` mặc định chỉ trỏ tới annot của Seq1, nên video của Seq khác bị từ chối, trừ khi mẫu có `{seq}`. Số frame đầu ra là độ dài ngắn hơn của hai đoạn (giới hạn thêm bằng `--frames`). Tên thư mục và `overlay_meta.json` ghi đoạn thực sự được dùng.

Mức che phụ thuộc vào việc X có đi ngang qua giữa khung hình hay không. Nên chọn frame bắt đầu bằng `--suggest`: lệnh này chỉ dùng `annot2`, không ghép video. Nó xét các vị trí bắt đầu nằm trong hai đoạn `--x`, `--y`, rồi in các cặp mà Y bị che vừa phải và nhiều lần. Các cặp bị loại khi:

- X và Y cùng dòng thời gian (cùng `annot.mat`, kể cả khác camera) mà hai đoạn nguồn giao nhau hoặc cách nhau dưới `--min-gap` frame (mặc định 100). Phần giao nhau là cùng một chuỗi động tác xuất hiện hai lần trong clip. Lệnh ghép cũng từ chối các cặp như vậy, trừ khi có `--allow-overlap`. Động tác lặp lại ở hai thời điểm không giao nhau thì chưa bị loại; muốn loại cần so độ giống chuyển động (ví dụ DTW).
- Y chạm mép khung nguồn ở quá `--max-y-cut` số frame (mặc định 0). Khi bị thu nhỏ và đặt giữa ảnh ghép, chỗ bị mép khung cắt thành một đường thẳng giữa cảnh.

```bash
python3 scripts/make_overlay_pair.py --x 0:6468 --y 0:6468 --x-video ../WHAM/inputs/mpi_inf_3dhpe/S8/Seq1/imageSequences/video_cam0.avi --frames 1500 --suggest
```

```bash
python3 scripts/make_overlay_pair.py --x 0:1500 --y 4750:6250 --x-video ../WHAM/inputs/mpi_inf_3dhpe/S8/Seq1/imageSequences/video_cam0.avi
```

Kết quả ở `outputs/overlay/<tên>/` (ví dụ trên: `ovl_S8c0f0-1500_S8c0f4750-6250_s0.5`): `composite.mp4` (chạy WHAM trên video này), `overlay_meta.json`, `overlay_frames.csv` (frame nguồn `src_x`, `src_y`, tỉ lệ Y còn thấy `vis_y`, Y bị mép khung nguồn cắt `y_cut`, độ lấp đầy mặt nạ `fill_x`, `fill_y`) và `preview/`. Đổi hệ số thu nhỏ bằng `--y-scale`. Trong `overlay_meta.json`, đường dẫn video và `annot.mat` là đường dẫn tương đối so với thư mục kết quả.

Muốn sửa tay một số frame lỗi (ví dụ mặt nạ hụt), chạy theo hai bước. Bước 1, `--mode generate` chỉ ghi từng frame ghép thành PNG `frames_raw/000000.png`, `000001.png`, …, chưa tạo MP4. Số trong tên file là thứ tự frame của video ghép, không phải frame nguồn. Mỗi frame khoảng 3 MB ở 2048×2048, tức khoảng 4.5 GB cho 1500 frame. Chế độ mặc định `--mode all` ghi thẳng MP4 và không lưu PNG.

```bash
python3 scripts/make_overlay_pair.py --x 0:1500 --y 4750:6250 --x-video ../WHAM/inputs/mpi_inf_3dhpe/S8/Seq1/imageSequences/video_cam0.avi --mode generate
```

Sửa frame cần sửa rồi lưu vào `frames_edit/`. File phải là PNG, cùng tên và cùng kích thước với file trong `frames_raw/`. Bước 2, `--mode render` ghép lại `composite.mp4`, frame nào có trong `frames_edit/` thì dùng bản đó:

```bash
python3 scripts/make_overlay_pair.py --mode render --name ovl_S8c0f0-1500_S8c0f4750-6250_s0.5
```

Lệnh render báo các tệp trong `frames_edit/` không được dùng (đuôi `.jpg`, sai tên), và dừng nếu kích thước sai. Sau khi render, `overlay_meta.json` ghi `"video": "composite.mp4"` và danh sách `edited_frames`. `overlay_frames.csv` (`vis_y`, diện tích mặt nạ) và các file GT không đổi theo frame sửa, nên chỉ nên sửa lỗi mặt nạ, không xoá hay thêm người. Nếu chạy lại `--mode generate` vào thư mục đã có `frames_edit/`, lệnh sẽ cảnh báo vì các frame sửa cũ sẽ được dùng khi render.

Mặt nạ của mỗi người là phần điểm ảnh không phải phông xanh, giao với một vùng quanh khung xương 2D. Một điểm ảnh là phông khi vừa có `G − max(R, B) > --tau` (mặc định 12), vừa xanh đậm màu: `(G − max(R, B)) / G > --min-chroma` (mặc định 0.40). Sát người, phông MPI có tỉ lệ này từ khoảng 0.42 đến 0.74. Quần áo xám ngả xanh hoặc da bị phông hắt xanh chỉ khoảng 0.2, nhưng `G − max(R, B)` vẫn khoảng 15–35, nên nếu chỉ dùng `--tau` thì chúng bị coi là phông. Ví dụ, với S4 (áo len xám-xanh), cách cũ mất khoảng 56% điểm ảnh của người. Chỗ bị mất không được dán lại khi đặt X lên trên, nên Y lộ ra đè lên X. `--min-chroma 0` cho lại cách cũ.

Cột `fill_x`, `fill_y` của `overlay_frames.csv` là tỉ lệ vùng quanh khung xương được giữ làm người; trên S8 trung vị khoảng 0.6. Cuối lệnh ghép có in trung vị này, và cảnh báo khi nó dưới 0.45 (mặt nạ hụt). Khi đó cần xem `mask_qa.jpg` và thử tăng `--min-chroma`. `overlay_meta.json` ghi `chroma_min_ratio`. Clip không có khoá này được làm theo cách cũ, và `--mask-qa` dựng lại mặt nạ của nó cũng theo cách cũ. Vùng này dày `--width` (mặc định 0.16) nhân cỡ người trên ảnh. Cỡ người tính theo độ sâu: `f · H / Z_pelvis`, với `H = --ref-height` (mặc định 1750 mm), `f = sqrt(f_x f_y)` khớp từ `annot3` và `annot2`, `Z_pelvis` lấy từ `annot3`. Cỡ này không đổi khi người đổi tư thế. Các clip cũ lấy cỡ người từ khung bao 2D; khung đó co lại khi ngồi xổm hoặc cúi sâu nên mặt nạ cắt mất đùi, bắp chân. `overlay_meta.json` ghi `ref_height_mm`; clip không có khoá này là clip theo cách cũ.

Trước khi tạo hàng loạt, nên xem mặt nạ ở các tư thế khó (nằm, cúi sâu, dang tay). Lệnh dưới đây ghi `mask_qa.jpg`, mỗi ô gồm ảnh gốc và phần được giữ (nền tím là phần bị bỏ):

```bash
python3 scripts/make_overlay_pair.py --mask-qa outputs/overlay/ovl_*
```

Ground truth được lấy lại từ `annot.mat` của video nguồn theo frame nguồn, và ghi cùng thư mục. Mỗi file có các cột `frame, person, src_frame` cùng 14 khớp (`pelvis` và 13 khớp như `data/raw/mpi_inf_3dhp/ground_truth/*.csv`):

| File | Nội dung | Dùng cho |
|---|---|---|
| `gt_3d.csv` | `annot3` gốc của từng người (mm, hệ camera của video nguồn) | Chi phí PA-MPJPE (bất biến dịch, xoay, tỉ lệ) |
| `gt_3d_scene.csv` | Pose 3D trong camera của video ghép (camera X, mm): X giữ nguyên; Y tịnh tiến cứng ra sau (độ sâu khoảng `Z / s`) để pelvis chiếu đúng chỗ đã dán | `C_root` |
| `gt_2d.csv` | Khớp 2D trên ảnh ghép (px): X là `annot2`, Y là `tâm + s·(annot2 − tâm)` | Gán nhãn tự động, `C_2D` |

Intrinsics của camera X được khớp lại từ `annot3 → annot2` (không cần file calib) và ghi vào mục `gt` của `overlay_meta.json`, kèm sai số chiếu `gt_3d_scene → gt_2d`. Trên S8 cam0, pelvis của Y lệch 0 px, các khớp khác lệch trung vị khoảng 4.5 px (phối cảnh yếu). Ghi lại GT cho các clip đã có:

```bash
python3 scripts/make_overlay_pair.py --gt-only outputs/overlay/ovl_*
```

Ba file GT ở ba hệ quy chiếu khác nhau; không trộn chúng trong cùng một phép tính:

- Hình dạng pose (PA-MPJPE): `gt_3d.csv`, hoặc `overlay_gt_poses()` mà pipeline đang dùng. Mỗi người ở hệ camera của video nguồn riêng, nên không dùng để tính khoảng cách giữa hai người.
- Vị trí và khoảng cách trong cảnh: `gt_3d_scene.csv`. Pelvis chính xác, nhưng các khớp xa thân theo chiều sâu (cổ tay, khuỷu) có thể lệch vài chục px khi chiếu, vì đây là phép dời cứng.
- Vị trí 2D trên ảnh: `gt_2d.csv` (chính xác theo phép thu nhỏ).

Lệnh `--gt-only` cũng in độ chênh intrinsics giữa hai nguồn và sai số chiếu (trung vị, p95, từng khớp). Các giá trị này được ghi vào mục `gt` của `overlay_meta.json`.

Sau khi chạy WHAM, thêm sequence vào `configs/splits.json` với `"gt": {"type": "mpi_overlay", "path": ".../outputs/overlay/<tên>/overlay_meta.json"}` rồi gán nhãn tự động:

```bash
python3 scripts/overlay_autolabel.py --seq <tên>
```

Lệnh này ghi `mapping.txt` và `mapping_frames.csv` cùng định dạng với gán tay, kèm `autolabel_review.csv` (các detection sát ngưỡng hoặc ở frame Y bị che quá nửa, cần xem lại). Kết quả đúng 9107/9107 detection trên video một người chưa chứng minh nhãn tự động đúng khi hai người chồng lên nhau, vì lúc bị che WHAM có thể "bịa" vị trí các khớp bị che. Vì vậy lệnh còn ghi `autolabel_audit.csv`, một mẫu ngẫu nhiên phân tầng theo mức che (`khong_che`, `che_vua`, `che_nang`). Điền cột `manual_person` bằng tay (tên người, hoặc `x` nếu không phải người GT), rồi chấm độ đúng theo từng mức che:

```bash
python3 scripts/overlay_autolabel.py --audit-score outputs/mapping/<tên>/autolabel_audit.csv
```

Không cần truyền `--calib`. Bản WHAM ở `../WHAM` luôn tính pose và vị trí hệ camera (`verts`, `trans`) bằng intrinsics heuristic `f = sqrt(W² + H²)` với điểm chính ở tâm ảnh, kể cả khi chạy với `--calib`; calib chỉ được đưa vào DPVO/SLAM. Kiểm trên H4D cam07 (calib `f_x = 1758.7`): `verts` chỉ khớp keypoint ViTPose khi chiếu bằng `f = 4405.7`.

Khi chạy WHAM trên video ghép, nếu truyền calib thì dùng calib của camera X (nền tĩnh mà DPVO theo dõi là của X). Không dùng calib của Y, và không có calib "tổng hợp" nào đúng cho cả hai người.

So vị trí WHAM (giữa hai hông, sau khi quy đổi camera) với GT trong cảnh. Ở video ghép, GT lấy từ `gt_3d_scene.csv` theo `(frame, person)` trong `mapping_frames.csv`, không lấy theo frame nguồn:

```bash
python3 scripts/check_wham_camera.py --pkl <wham_output.pkl> --meta outputs/overlay/<tên>/overlay_meta.json --mapping outputs/mapping/<tên>/mapping_frames.csv
```

Với video MPI đơn thì thay `--meta` bằng `--annot data/raw/mpi_inf_3dhp/S8/annot.mat --camera 0`.

## 11. Giới hạn và các giả định hiện tại

1. **GT được sử dụng làm tham chiếu khi tính chi phí pose.** Framework hiện tại đánh giá và lựa chọn detection dựa trên PA-MPJPE so với pose GT. Đây là thiết lập có giám sát, không phải phương pháp suy luận hoàn toàn không cần GT.

2. **Temporal cost không phải vận tốc vật lý.** Nó đo sự thay đổi của residual sau Procrustes alignment, không trực tiếp mô hình hóa chuyển động thực của người.

3. **Frame gap chỉ được chuẩn hóa ở temporal cost.** Temporal cost chia cho `Δt^p`. `C_disappear` và `C_reappear` vẫn tính như một bước chuyển thông thường, dù hai lớp cách nhau nhiều frame. Chưa có ngưỡng ngắt chuỗi khi `Δt` quá lớn.

4. **Temporal cost bằng 0 khi không đủ khớp hợp lệ.** Nếu số khớp hợp lệ chung giữa hai trạng thái nhỏ hơn 8, hàm `motion_rms()` trả về 0. Điều này tránh lỗi tính toán nhưng cũng có thể làm mất tín hiệu temporal ở các trường hợp thiếu dữ liệu.

5. **MILP có giới hạn thời gian.** Khi bộ giải trả về nghiệm khả thi trước khi chứng minh tối ưu, kết quả vẫn được sử dụng nhưng được đánh dấu `optimal = false` kèm `mip_gap` (xem mục 6.2).

6. **Chi phí false positive được tính trên toàn bộ detection đầu vào.** Các detection không được sử dụng đều góp vào `N_FP`, kể cả khi chúng không trở thành một nút ứng viên hợp lệ cho bất kỳ người GT nào.

7. **Chưa dùng vị trí gốc 3D và tái chiếu 2D trong chi phí đơn.** Chi phí đơn chỉ dùng PA-MPJPE, vốn đã loại bỏ vị trí gốc. Hai thành phần `C_root` (vị trí pelvis 3D) và `C_2D` (sai số tái chiếu so với GT 2D) chưa được cài đặt. Vị trí trên ảnh hiện chỉ vào mô hình qua `C_V` (độ dịch giữa hai detection liên tiếp, không dùng GT, mục 4.2).

9. **`C_V` không phạt bước nhảy đi qua trạng thái missed.** `C_V` chỉ nằm trên cung Detected → Detected. Một đường đi có thể đổi sang detection của người khác bằng cách đi qua một nút missed, khi đó chỉ trả `C_disappear + C_FN + C_reappear` (và mất phần lợi của detection), không trả `C_V`. Giống `C_temporal`, `C_V` bằng 0 khi một trong hai detection có ít hơn 8 khớp hợp lệ trên ảnh.

8. **Video ghép chỉ hợp với chi phí bất biến vị trí và tỉ lệ.** Thu nhỏ người Y trong ảnh không đổi PA-MPJPE, và GT 2D được thu nhỏ theo cùng phép biến đổi. Vị trí gốc 3D của Y trong video ghép không còn là `annot3` gốc mà xấp xỉ `(X, Y, Z / s)`, cộng thêm độ lệch đã biết khi tâm thu nhỏ khác điểm chính. Ngoài ra `verts`/`trans` của WHAM nằm trong camera heuristic (`f = sqrt(W² + H²)`), không phải camera thật, nên phải quy đổi trước khi so với GT: giữ nguyên pixel, nhân độ sâu với `f_thật / f_heuristic` (`scripts/check_wham_camera.py`). Trên S8 cam0 (6259 detection đúng, GT sâu khoảng 3.5 m), so thẳng thì vị trí giữa hai hông lệch trung vị 345 cm (độ sâu gấp đôi); sau khi quy đổi thì lệch 13 cm (độ sâu +6 cm, p95 31 cm). Vì vậy muốn đánh giá `C_root` hay `C_2D` trên video ghép thì phải biến đổi GT của Y tương ứng, không dùng thẳng `annot3`. Đây là benchmark tổng hợp có kiểm soát, đặt bên cạnh dữ liệu đa người thật (Harmony4D), không thay thế.

## 12. Định hướng phát triển

Các hướng mở rộng có thể được nghiên cứu và đánh giá thêm:

- Bổ sung chi phí vị trí gốc 3D và tái chiếu 2D vào chi phí đơn.
- Phạt độ dịch 2D giữa hai detection ở hai đầu một đoạn missed (xem giới hạn 9).
- Chuẩn hóa `C_disappear`, `C_reappear` theo `Δt`, hoặc ngắt chuỗi khi `Δt` vượt ngưỡng.
- Bổ sung các đặc trưng hình học và động học độc lập với GT.
- Đánh giá độ nhạy của nghiệm đối với các tham số pose, temporal, FP và FN.
- So sánh DP, MILP và các phương pháp tối ưu hóa khác về thời gian giải và chất lượng nghiệm.
- Đánh giá khả năng tổng quát hóa từ MPI-INF-3DHP sang dữ liệu đa người như Harmony4D.
- Nghiên cứu các ràng buộc nhận dạng và tính liên tục của người trong trường hợp nhiều người tương tác hoặc che khuất.

## 13. Tóm tắt

Framework xây dựng một mạng time-expanded với một luồng đơn vị cho mỗi người GT. Chi phí pose, temporal residual và các chi phí chuyển trạng thái được kết hợp trong một hàm mục tiêu tuyến tính.

Với một người GT, bài toán được giải chính xác bằng dynamic programming. Với nhiều người GT, framework trước tiên tìm các đường đi độc lập bằng DP; nếu các đường đi không xung đột detection, nghiệm đó được chấp nhận. Nếu xuất hiện xung đột, bài toán được giải bằng MILP với ràng buộc dung lượng detection dùng chung, và trạng thái tối ưu của HiGHS được ghi lại cùng kết quả.

Cách tiếp cận này kết hợp ưu điểm của shortest-path trên DAG và tối ưu hóa nguyên hỗn hợp để xử lý bài toán gán detection có nhiều đối tượng GT.
