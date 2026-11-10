# MC-MCF: Multi-Commodity Min-Cost Flow cho gán detection người 3D

## 0. Hướng dẫn sử dụng

Để biết cách cài đặt môi trường, chuẩn bị dữ liệu, cấu hình và
chạy các chương trình trong framework MC-MCF, xem tài liệu:
https://docs.google.com/document/d/1fUIUKLVQOlAJ9S-xI8ahBLh3dZfIocNxEP8fE2JtR_8/edit?usp=sharing
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

## 2. Ý tưởng phương pháp

### 2.1. Đối tượng và detection

Ký hiệu:

- \(K\): số người GT.
- \(t\): chỉ số frame.
- \(d\): một detection từ WHAM.
- \(k\): một người GT.
- \(PA_{tkd}\): PA-MPJPE giữa detection \(d\) và người GT \(k\) tại frame \(t\).

Mỗi detection được xác định bằng cặp:

\[
(t,d)=(\text{frame},\text{body\_id})
\]

Tại mỗi frame mà người GT \(k\) xuất hiện, mạng tạo các trạng thái:

1. **Detected:** gán một detection WHAM cho người GT.
2. **Missed:** không có detection nào được gán cho người GT tại frame đó.

Mỗi người GT được mô hình hóa bằng một luồng đơn vị (*unit flow*) đi từ source đến sink qua các lớp frame tương ứng.

### 2.2. Mạng time-expanded

Với mỗi người GT, mạng có cấu trúc:

\[
s_k \rightarrow V_{1,k}\rightarrow V_{2,k}
\rightarrow \cdots \rightarrow V_{T,k}\rightarrow t_k
\]

Trong đó:

- \(s_k\): source của người GT \(k\).
- \(t_k\): sink của người GT \(k\).
- \(V_{t,k}\): tập trạng thái của người \(k\) tại frame \(t\).

Các trạng thái giữa hai frame liên tiếp trong danh sách frame của người GT được kết nối bằng các cung chuyển tiếp. Mỗi cung chuyển tiếp lưu thêm khoảng cách frame gốc \(\Delta t\) giữa hai lớp (bằng 1 nếu hai frame liền nhau).

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

Các chi phí của nút và cung được lưu dưới dạng các đại lượng thô (PA-MPJPE, RMS residual, \(\Delta t\), loại cung) và được tính lại bằng `apply_params()` khi thay đổi bộ tham số, không cần dựng lại mạng.

### 2.3. Ràng buộc dùng chung detection

Trong trường hợp nhiều người GT, mỗi người có một luồng riêng. Tuy nhiên, các luồng cùng sử dụng tập detection WHAM.

Một detection chỉ được gán cho tối đa một người GT:

\[
\sum_k x_{tkd}\leq 1
\]

Trong đó \(x_{tkd}=1\) nếu detection \(d\) tại frame \(t\) được gán cho người \(k\).

Đây là ràng buộc liên kết các luồng và tạo nên bản chất multi-commodity của bài toán.

## 3. Hàm mục tiêu

Hàm mục tiêu kết hợp chi phí pose, false positive, false negative, tính nhất quán theo thời gian, chi phí chuyển trạng thái và chi phí biên.

### 3.1. Chi phí pose và gán detection

Với detection \(d\) được gán cho người GT \(k\) tại frame \(t\), chi phí nút là:

\[
C_{\mathrm{det}}(t,k,d)
=
\alpha\frac{PA_{tkd}}{s_P}-C_{FP}
\]

Trong đó:

- \(PA_{tkd}\): PA-MPJPE, tính bằng mm.
- \(\alpha\): trọng số pose.
- \(s_P\): hệ số chuẩn hóa pose, mặc định 100 mm.
- \(C_{FP}\): chi phí false positive, mặc định 0.8.

Chi phí trạng thái missed:

\[
C_{\mathrm{miss}}=C_{FN}
\]

Trong đó \(C_{FN}\) là chi phí false negative, mặc định bằng 1.0.

**Giải thích cách tính false positive**

Code sử dụng \(-C_{FP}\) trong chi phí của mỗi detection được gán và cộng thêm một hằng số:

\[
C_{FP}N_D
\]

với \(N_D\) là tổng số detection trong dữ liệu đầu vào.

Do đó:

\[
\begin{aligned}
&\sum_{\mathrm{assigned}\ d}
\left(
\alpha\frac{PA_d}{s_P}-C_{FP}
\right)
+C_{FP}N_D\\
&=
\sum_{\mathrm{assigned}\ d}
\alpha\frac{PA_d}{s_P}
+C_{FP}N_{FP}
\end{aligned}
\]

với \(N_{FP}\) là số detection không được sử dụng.

Cách biểu diễn này giúp đưa chi phí false positive vào mô hình dưới dạng chi phí nút tuyến tính, trong khi vẫn giữ nguyên ý nghĩa của hàm mục tiêu.

### 3.2. Chi phí nhất quán theo thời gian

Với hai trạng thái Detected liên tiếp \(a\) và \(b\) trên đường đi của cùng một người GT, framework tính chi phí dựa trên sự thay đổi của residual sau Procrustes alignment.

Residual tại frame \(t\), khớp \(j\):

\[
r_{t,j}
=
\hat{X}_{t,j}^{aligned}-Y_{t,j}^{GT}
\]

Trong đó:

- \(\hat{X}_{t,j}^{aligned}\): khớp dự đoán sau khi căn chỉnh với GT.
- \(Y_{t,j}^{GT}\): khớp tương ứng trong GT.

Độ thay đổi residual giữa hai trạng thái:

\[
RMS(r_a,r_b)
=
\sqrt{
\frac{1}{J}
\sum_{j=1}^{J}
\|r_{b,j}-r_{a,j}\|_2^2
}
\]

Chỉ các khớp hợp lệ ở cả hai frame mới được sử dụng. Nếu có ít hơn 8 khớp hợp lệ, hàm trả về 0.

Chi phí temporal:

\[
C_{\mathrm{temporal}}(a,b)
=
\gamma\frac{RMS(r_a,r_b)}{s_M\,(\Delta t_{ab})^{p}}
\]

Trong đó:

- \(\gamma\): trọng số temporal.
- \(s_M\): hệ số chuẩn hóa temporal, mặc định 0.05 m.
- \(\Delta t_{ab}\ge 1\): số frame gốc giữa hai lớp \(a\) và \(b\).
- \(p\): `gap_power`, mặc định 1. Với \(p=1\), chi phí đo *tốc độ* thay đổi residual trên mỗi frame, nên một khoảng trống 10 frame không bị phạt như một bước nhảy giữa hai frame liền nhau. Đặt \(p=0\) để bỏ qua \(\Delta t\).

Khi mọi frame GT liền nhau (\(\Delta t=1\)), công thức trùng với công thức không có \(\Delta t\). Trên các sequence đã gán nhãn (MPI S8 cam0/cam8, Harmony4D 041 cam07), mọi \(\Delta t\) đều bằng 1.

**Lưu ý về ý nghĩa của temporal cost**

Đây là chi phí đo sự thay đổi của sai số residual sau Procrustes alignment độc lập ở từng frame. Nó không trực tiếp đo vận tốc, độ dịch chuyển hay chuyển động vật lý của cơ thể.

Vì vậy, trong framework này, thuật ngữ *temporal residual consistency* được sử dụng thay cho *physical motion smoothness*.

### 3.3. Chi phí chuyển trạng thái

Các trạng thái trên đường đi có các chi phí chuyển tiếp:

| Chuyển trạng thái | Chi phí |
|---|---:|
| Detected → Detected | \(C_{\mathrm{temporal}}\) |
| Detected → Missed | \(C_{\mathrm{disappear}}\) |
| Missed → Detected | \(C_{\mathrm{reappear}}\) |
| Missed → Missed | 0 |

Trong đó:

- \(C_{\mathrm{disappear}}\): chi phí khi một người chuyển từ trạng thái detected sang missed.
- \(C_{\mathrm{reappear}}\): chi phí khi một người chuyển từ missed sang detected.

\(C_{\mathrm{disappear}}\) và \(C_{\mathrm{reappear}}\) là chi phí cho một *sự kiện* đổi trạng thái nên không chia theo \(\Delta t\).

### 3.4. Chi phí biên

Mạng cho phép cấu hình riêng chi phí tại đầu và cuối chuỗi:

\[
C_{\mathrm{start}}=
\begin{cases}
C_{\mathrm{start,det}}, & \text{nếu trạng thái đầu là detected}\\
C_{\mathrm{start,miss}}, & \text{nếu trạng thái đầu là missed}
\end{cases}
\]

Tương tự:

\[
C_{\mathrm{end}}=
\begin{cases}
C_{\mathrm{end,det}}, & \text{nếu trạng thái cuối là detected}\\
C_{\mathrm{end,miss}}, & \text{nếu trạng thái cuối là missed}
\end{cases}
\]

Các chi phí biên mặc định đều bằng 0.

### 3.5. Hàm mục tiêu tổng thể

Có thể biểu diễn hàm mục tiêu dưới dạng:

\[
\begin{aligned}
\min \mathcal{L}={}&
\sum_{t,k,d}x_{tkd}
\left(
\alpha\frac{PA_{tkd}}{s_P}-C_{FP}
\right)\\
&+\sum_{t,k}m_{tk}C_{FN}\\
&+\sum_{(a,b)\in E_{dd}}
C_{\mathrm{temporal}}(a,b)\\
&+N_{dm}C_{\mathrm{disappear}}\\
&+N_{md}C_{\mathrm{reappear}}\\
&+C_{\mathrm{boundary}}+C_{FP}N_D
\end{aligned}
\]

Trong đó:

- \(m_{tk}=1\) nếu người \(k\) bị missed tại frame \(t\).
- \(E_{dd}\): tập các cung Detected → Detected được sử dụng.
- \(N_{dm}\): số cung Detected → Missed được sử dụng.
- \(N_{md}\): số cung Missed → Detected được sử dụng.
- \(C_{\mathrm{boundary}}\): tổng chi phí biên.
- \(N_D\): tổng số detection đầu vào.

Hàm mục tiêu được tối thiểu hóa với các ràng buộc bảo toàn luồng và dung lượng detection dùng chung.

## 4. Mô hình tối ưu hóa MC-MCF

### 4.1. Ràng buộc bảo toàn luồng

Với mỗi người GT, luồng phải đi từ source đến sink.

Tại các nút trung gian, tổng luồng đi vào bằng tổng luồng đi ra.

Với biến luồng trên cung \(a\), ký hiệu \(f_a\), ràng buộc bảo toàn luồng được viết:

\[
\sum_{a\in\delta^-(v)}f_a
-
\sum_{a\in\delta^+(v)}f_a
=
b_v
\]

Trong đó:

\[
b_v=
\begin{cases}
-1,&v=\text{source}\\
1,&v=\text{sink}\\
0,&\text{các nút khác}
\end{cases}
\]

Mỗi người GT có một đơn vị luồng từ source đến sink.

### 4.2. Ràng buộc dung lượng detection

Với mỗi detection \((t,d)\):

\[
\sum_k x_{tkd}\leq 1
\]

Ràng buộc này bảo đảm một detection WHAM không thể được sử dụng đồng thời bởi nhiều người GT.

### 4.3. Biến quyết định

Các biến luồng trên cung là biến nhị phân:

\[
f_a\in\{0,1\}
\]

Vì mỗi luồng đi qua một đường duy nhất trong mạng phân lớp, các biến này biểu diễn việc lựa chọn đường đi tương ứng với từng người GT.

### 4.4. Công thức bài toán MILP

Sau khi xây dựng các chi phí nút và cung, bài toán được biểu diễn dưới dạng:

\[
\min_f \sum_{a\in A}w_af_a+C_{FP}N_D
\]

với:

\[
w_a=C_{\mathrm{arc}}(a)+C_{\mathrm{node}}(\mathrm{head}(a))
\]

subject to:

\[
Af=b
\]

\[
Bf\leq \mathbf{1}
\]

\[
f_a\in\{0,1\}
\]

Trong đó:

- \(A\): ma trận ràng buộc bảo toàn luồng.
- \(B\): ma trận ràng buộc dung lượng detection.
- \(w_a\): trọng số của cung, bao gồm chi phí cung và chi phí nút đích.
- \(C_{FP}N_D\): hằng số được cộng khi tính objective đầy đủ.

## 5. Chiến lược giải bài toán

Framework sử dụng hai phương pháp: Dynamic Programming (DP) và Mixed-Integer Linear Programming (MILP).

### 5.1. Single-commodity: Dynamic Programming

Với một người GT, bài toán trở thành tìm đường đi ngắn nhất trên mạng phân lớp dạng DAG.

Do các cung được tạo theo thứ tự topo, thuật toán có thể duyệt lần lượt các cung để cập nhật chi phí đường đi ngắn nhất.

Độ phức tạp của bước tìm đường đi tỷ lệ tuyến tính với số nút và số cung:

\[
O(|V|+|A|)
\]

Đây là nghiệm tối ưu chính xác cho trường hợp một người GT trong mô hình hiện tại.

### 5.2. Multi-commodity: DP kết hợp MILP

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

## 6. Các tham số mặc định

| Tham số | Giá trị mặc định | Ý nghĩa | Khóa trong `configs/default.yaml` (mục `association`) |
|---|---:|---|---|
| \(\alpha\) | 1.0 | Trọng số pose | `weights.pose` |
| \(\gamma\) | 0.5 | Trọng số temporal residual | `weights.temporal` |
| \(C_{FP}\) | 0.8 | Phạt false positive | `costs.false_positive` |
| \(C_{FN}\) | 1.0 | Phạt false negative | `costs.false_negative` |
| \(C_{\mathrm{disappear}}\) | 0.15 | Phạt Detected → Missed | `costs.disappear` |
| \(C_{\mathrm{reappear}}\) | 0.20 | Phạt Missed → Detected | `costs.reappear` |
| \(s_P\) | 100 mm | Pose normalization scale | `scales.pose_mm` |
| \(s_M\) | 0.05 m | Temporal normalization scale | `scales.motion_m` |
| \(p\) | 1.0 | Số mũ của \(\Delta t\) trong temporal cost | `temporal.gap_power` |
| \(C_{\mathrm{start,det}}\) | 0.0 | Chi phí đầu chuỗi, detected | `boundary.start_detected` |
| \(C_{\mathrm{start,miss}}\) | 0.0 | Chi phí đầu chuỗi, missed | `boundary.start_missed` |
| \(C_{\mathrm{end,det}}\) | 0.0 | Chi phí cuối chuỗi, detected | `boundary.end_detected` |
| \(C_{\mathrm{end,miss}}\) | 0.0 | Chi phí cuối chuỗi, missed | `boundary.end_missed` |

Các tham số được tổ chức trong `CostParams` (`pose_assoc/costs.py`) và được nạp từ cấu hình thông qua `from_config()`. Bảng này được kiểm tra tự động với `CostParams` bởi `tests/test_readme_sync.py`.

## 7. Dữ liệu và tiền xử lý

### 7.1. Bộ khớp sử dụng

Framework sử dụng 12 khớp chung giữa các định dạng pose:

- Left shoulder, right shoulder
- Left elbow, right elbow
- Left wrist, right wrist
- Left hip, right hip
- Left knee, right knee
- Left ankle, right ankle

Số khớp hợp lệ tối thiểu để thực hiện Procrustes alignment là 8.

### 7.2. Chuẩn hóa đơn vị

Dữ liệu GT của MPI-INF-3DHP được lưu bằng millimeter, trong khi dữ liệu WHAM được lưu bằng meter.

Trước khi tính sai số, GT được chuyển đổi sang meter:

\[
X_{GT}^{(m)}=\frac{X_{GT}^{(mm)}}{1000}
\]

PA-MPJPE được tính bằng meter trong quá trình alignment và được chuyển sang millimeter khi lưu vào mạng.

### 7.3. Procrustes alignment

Dự đoán được căn chỉnh với GT bằng phép biến đổi similarity gồm:

- Translation.
- Rotation.
- Uniform scaling.

Sau khi căn chỉnh, PA-MPJPE được tính bằng trung bình khoảng cách Euclidean giữa các khớp hợp lệ.

Residual sau alignment được lưu lại để tính temporal residual consistency cost.

## 8. Cấu trúc mã nguồn

| Module | Chức năng |
|---|---|
| `pose_assoc/costs.py` | Tham số chi phí `CostParams`, Procrustes alignment, PA-MPJPE, temporal residual cost |
| `pose_assoc/network.py` | Xây dựng mạng time-expanded, các nút, cung và trọng số; `apply_params()` |
| `pose_assoc/solvers/single_mcf.py` | Giải đường đi ngắn nhất độc lập bằng dynamic programming |
| `pose_assoc/solvers/multi_mcf.py` | Kiểm tra xung đột detection, giải MILP khi cần, ghi trạng thái HiGHS |
| `pose_assoc/evaluation.py` | Bảng gán, phân rã hàm mục tiêu, chấm điểm so với mapping gán tay |
| `pose_assoc/build_network.py` | CLI: dựng mạng, giải và xuất kết quả cho các sequence trong manifest |
| `pose_assoc/tune.py` | CLI: hiệu chỉnh tham số tích lũy qua nhiều lần trên tập validation |
| `scripts/wham_gt_mapping.py` | CLI: gán nhãn tay body_id WHAM ↔ người GT (cả theo đoạn frame) |
| `scripts/solve_mc_mcf.py` | Script shortest-path độc lập cho trường hợp một người GT |
| `configs/splits.json` | Manifest các sequence validation / test |
| `configs/default.yaml` | Tham số mặc định (mục `association`) |

`solve_mc_mcf.py` là một script giải bài toán một người GT theo cách riêng. Nó hữu ích để kiểm tra và đối chiếu với trường hợp single-commodity, nhưng không thay thế bộ xây dựng mạng và solver multi-commodity của framework.

## 9. Cài đặt và chạy

Cần Python ≥ 3.9. Các lệnh dưới đây chạy từ thư mục gốc của repo.

```bash
python3 -m pip install --user -r requirements.txt
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

Kết quả nằm ở `outputs/network/<sequence>/` (`nodes.csv`, `arcs.csv`, `assignment.csv`, `summary.json`). Có thể ghi đè tham số bằng `--alpha`, `--gamma`, `--c-fp`, `--c-fn`, `--c-disappear`, `--c-reappear`, hoặc dùng `--config` với một file YAML.

Hiệu chỉnh tham số trên các sequence validation đã gán nhãn:

```bash
python3 -m pose_assoc.tune
```

Bộ tham số của mỗi lần hiệu chỉnh được ghi ở `configs/tuning/round_<n>.yaml` và có thể dùng lại qua `--config`.

## 10. Giới hạn và các giả định hiện tại

1. **GT được sử dụng làm tham chiếu khi tính chi phí pose.** Framework hiện tại đánh giá và lựa chọn detection dựa trên PA-MPJPE so với pose GT. Đây là thiết lập có giám sát, không phải phương pháp suy luận hoàn toàn không cần GT.

2. **Temporal cost không phải vận tốc vật lý.** Nó đo sự thay đổi của residual sau Procrustes alignment, không trực tiếp mô hình hóa chuyển động thực của người.

3. **Frame gap chỉ được chuẩn hóa ở temporal cost.** Temporal cost chia cho \(\Delta t^{p}\). \(C_{\mathrm{disappear}}\) và \(C_{\mathrm{reappear}}\) vẫn tính như một bước chuyển thông thường, dù hai lớp cách nhau nhiều frame. Chưa có ngưỡng ngắt chuỗi khi \(\Delta t\) quá lớn.

4. **Temporal cost bằng 0 khi không đủ khớp hợp lệ.** Nếu số khớp hợp lệ chung giữa hai trạng thái nhỏ hơn 8, hàm `motion_rms()` trả về 0. Điều này tránh lỗi tính toán nhưng cũng có thể làm mất tín hiệu temporal ở các trường hợp thiếu dữ liệu.

5. **MILP có giới hạn thời gian.** Khi bộ giải trả về nghiệm khả thi trước khi chứng minh tối ưu, kết quả vẫn được sử dụng nhưng được đánh dấu `optimal = false` kèm `mip_gap` (xem mục 5.2).

6. **Chi phí false positive được tính trên toàn bộ detection đầu vào.** Các detection không được sử dụng đều góp vào \(N_{FP}\), kể cả khi chúng không trở thành một nút ứng viên hợp lệ cho bất kỳ người GT nào.

7. **Chưa dùng vị trí gốc 3D và tái chiếu 2D.** Chi phí đơn chỉ dùng PA-MPJPE, vốn đã loại bỏ vị trí gốc. Hai thành phần \(C_{\mathrm{root}}\) (vị trí pelvis 3D) và \(C_{2D}\) (sai số tái chiếu dùng calib camera) chưa được cài đặt.

## 11. Định hướng phát triển

Các hướng mở rộng có thể được nghiên cứu và đánh giá thêm:

- Bổ sung chi phí vị trí gốc 3D và tái chiếu 2D vào chi phí đơn.
- Chuẩn hóa \(C_{\mathrm{disappear}}\), \(C_{\mathrm{reappear}}\) theo \(\Delta t\), hoặc ngắt chuỗi khi \(\Delta t\) vượt ngưỡng.
- Bổ sung các đặc trưng hình học và động học độc lập với GT.
- Đánh giá độ nhạy của nghiệm đối với các tham số pose, temporal, FP và FN.
- So sánh DP, MILP và các phương pháp tối ưu hóa khác về thời gian giải và chất lượng nghiệm.
- Đánh giá khả năng tổng quát hóa từ MPI-INF-3DHP sang dữ liệu đa người như Harmony4D.
- Nghiên cứu các ràng buộc nhận dạng và tính liên tục của người trong trường hợp nhiều người tương tác hoặc che khuất.

## 12. Tóm tắt

Framework xây dựng một mạng time-expanded với một luồng đơn vị cho mỗi người GT. Chi phí pose, temporal residual và các chi phí chuyển trạng thái được kết hợp trong một hàm mục tiêu tuyến tính.

Với một người GT, bài toán được giải chính xác bằng dynamic programming. Với nhiều người GT, framework trước tiên tìm các đường đi độc lập bằng DP; nếu các đường đi không xung đột detection, nghiệm đó được chấp nhận. Nếu xuất hiện xung đột, bài toán được giải bằng MILP với ràng buộc dung lượng detection dùng chung, và trạng thái tối ưu của HiGHS được ghi lại cùng kết quả.

Cách tiếp cận này kết hợp ưu điểm của shortest-path trên DAG và tối ưu hóa nguyên hỗn hợp để xử lý bài toán gán detection có nhiều đối tượng GT.
