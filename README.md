# Monocular HPE: Phát hiện người và độ tin cậy của định danh

Dự án nghiên cứu các lỗi phát hiện người và sự không nhất quán về định danh trong các mô hình ước lượng tư thế người 3D từ video đơn camera (Monocular 3D Human Pose Estimation – HPE). Dự án tập trung khảo sát mô hình WHAM và phát triển một khuôn khổ tối ưu hóa luồng chi phí cực tiểu đa hàng hóa (Multi-Commodity Min-Cost Flow – MC-MCF).

## 1. Động cơ nghiên cứu

Các mô hình Monocular 3D Human Pose Estimation ước lượng tư thế người trong không gian 3D từ các chuỗi video. Tuy nhiên, kết quả dự đoán của những mô hình này có thể chứa các sai sót mà những chỉ số đánh giá tư thế truyền thống như MPJPE hoặc PA-MPJPE chưa phản ánh đầy đủ.

Trong các tình huống có nhiều người, mô hình có thể gặp những vấn đề sau:

- Phát hiện một người không tương ứng với bất kỳ người nào trong dữ liệu chuẩn (Ground Truth – GT).
- Không phát hiện được một người thực sự xuất hiện trong GT.
- Gán các phát hiện thuộc về nhiều người thực tế khác nhau vào cùng một định danh dự đoán.
- Chia các phát hiện thuộc về cùng một người thực tế thành nhiều định danh dự đoán khác nhau.
- Phát hiện và theo dõi người không nhất quán giữa các khung hình liên tiếp.

Những sai sót này ảnh hưởng đến cả độ tin cậy trong phát hiện người và tính nhất quán của định danh theo thời gian.

Dự án hướng tới việc khảo sát có hệ thống các hiện tượng trên, đồng thời nghiên cứu những phương pháp dựa trên tối ưu hóa nhằm phân tích và cải thiện quá trình liên kết giữa các tư thế người được mô hình dự đoán với những người thực tế.

## 2. Các câu hỏi nghiên cứu

Dự án tập trung vào bốn câu hỏi nghiên cứu chính:

- **RQ1 – Độ tin cậy của phát hiện người (Detection Reliability):** Các mô hình Monocular HPE tạo ra phát hiện dương tính giả (False Positive) và âm tính giả (False Negative) với tần suất như thế nào?
- **RQ2 – Tính nhất quán của định danh (Identity Consistency):** Định danh người do mô hình dự đoán nhất quán với người thực tế đến mức nào qua các khung hình video?
- **RQ3 – Liên kết theo thời gian (Temporal Association):** Thông tin về thời gian và hình học có thể cải thiện việc liên kết các phát hiện tư thế ở từng khung hình với định danh người trong GT hay không?
- **RQ4 – Khuôn khổ tối ưu hóa (Optimization Framework):** Mô hình hóa bài toán dưới dạng Multi-Commodity Min-Cost Flow có thể tạo ra kết quả gán định danh toàn cục đáng tin cậy hơn các phương pháp đối sánh độc lập theo từng khung hình hay không?

## 3. Bộ dữ liệu và mô hình nghiên cứu

### 3.1. MPI-INF-3DHP

Bộ dữ liệu được sử dụng trong giai đoạn thực nghiệm ban đầu là MPI-INF-3DHP.

Bộ dữ liệu cung cấp các chú thích tư thế người 3D và thông tin camera, cho phép đánh giá kết quả dự đoán của các mô hình Monocular HPE.

Các thí nghiệm ban đầu sử dụng:

- Kết quả dự đoán của WHAM được tạo từ video MPI-INF-3DHP.
- Dữ liệu tư thế chuẩn MPI-INF-3DHP (Ground Truth).
- Thông tin hiệu chuẩn camera (Camera Calibration).
- Định danh ở cấp độ khung hình và cấp độ người.

Phiên bản triển khai đầu tiên tập trung xử lý và đối sánh đúng định dạng CSV thực tế được tạo ra từ kết quả dự đoán WHAM và dữ liệu GT của MPI-INF-3DHP.

### 3.2. WHAM

WHAM được sử dụng làm mô hình Monocular HPE đầu tiên để khảo sát.

Các body ID do WHAM tạo ra được giữ nguyên như kết quả đầu ra của mô hình. Không được mặc định rằng những ID này chính là định danh người trong Ground Truth.

Mỗi phát hiện được biểu diễn bằng một quan sát tại khung hình t, ký hiệu là d(t,i), trong đó:

- t là chỉ số khung hình.
- i là chỉ số của một phát hiện trong khung hình đó.

Body ID do WHAM tạo ra được lưu giữ như một thuộc tính đi kèm với từng phát hiện.

### 3.3. Mở rộng trong tương lai: Harmony4D

Harmony4D được dự kiến bổ sung như một bộ dữ liệu đánh giá tiếp theo.

Kiến trúc nạp dữ liệu và xử lý tọa độ được thiết kế để có thể hỗ trợ nhiều bộ dữ liệu khác nhau. Tuy nhiên, phiên bản triển khai đầu tiên và quá trình tối ưu tham số sẽ tập trung vào MPI-INF-3DHP.

## 4. Mô hình hóa bài toán

Tại mỗi khung hình t, xác định:

- G(t): Tập hợp những người xuất hiện trong Ground Truth.
- D(t): Tập hợp các phát hiện người do mô hình tạo ra.
- g(t,j): Người thứ j trong Ground Truth.
- d(t,i): Phát hiện thứ i do mô hình tạo ra.

Bài toán liên kết cần xác định mỗi phát hiện dự đoán có tương ứng với một người trong Ground Truth hay không.

Mô hình hóa bài toán phải cho phép các tình huống sau:

1. Một người trong Ground Truth không có phát hiện tương ứng tại một khung hình.
2. Một phát hiện dự đoán không tương ứng với bất kỳ người nào trong Ground Truth.
3. Một người trong Ground Truth có thể được liên kết với các model ID khác nhau ở những khung hình khác nhau.
4. Một model ID có thể chứa các phát hiện tương ứng với nhiều người khác nhau trong Ground Truth.

Cách tiếp cận này tránh giả định rằng các định danh ban đầu của mô hình luôn chính xác.

## 5. Khuôn khổ MC-MCF đề xuất

Dự án nghiên cứu mô hình hóa bài toán liên kết định danh toàn cục theo thời gian bằng Multi-Commodity Min-Cost Flow (MC-MCF).

Mỗi định danh người trong Ground Truth được xem như một luồng hàng hóa (Commodity) riêng biệt.

Quá trình tối ưu hóa hoạt động trên các phát hiện ở cấp độ khung hình, thay vì mặc định rằng các track do WHAM tạo ra đã chính xác.

### 5.1. Các biến quyết định

Sử dụng ba nhóm biến quyết định:

**Biến x(t,j,i):**

- Nhận giá trị 1 nếu phát hiện thứ i tại khung hình t được gán cho người thứ j trong Ground Truth.
- Nhận giá trị 0 trong trường hợp ngược lại.

**Biến m(t,j):**

- Nhận giá trị 1 nếu người thứ j trong Ground Truth bị bỏ sót tại khung hình t.
- Nhận giá trị 0 trong trường hợp ngược lại.

**Biến u(t,i):**

- Nhận giá trị 1 nếu phát hiện thứ i tại khung hình t không được gán cho bất kỳ người nào trong Ground Truth.
- Nhận giá trị 0 trong trường hợp ngược lại.

Các ràng buộc gán định danh:

- Mỗi người trong Ground Truth phải được gán đúng một phát hiện hoặc được đánh dấu là bị bỏ sót.
- Mỗi phát hiện phải được gán cho đúng một người trong Ground Truth hoặc được đánh dấu là không có đối tượng tương ứng.

Có thể biểu diễn hai ràng buộc trên bằng cách diễn giải:

**Ràng buộc 1 – Đối với mỗi người trong GT:**

Tổng số phát hiện được gán cho người đó cộng với biến đánh dấu bỏ sót phải bằng 1.

**Ràng buộc 2 – Đối với mỗi phát hiện:**

Tổng số người trong GT được gán với phát hiện đó cộng với biến đánh dấu không đối tượng phải bằng 1.

Hai ràng buộc này cho phép mô hình biểu diễn tường minh cả trường hợp bỏ sót người và trường hợp phát hiện không có đối tượng tương ứng.

### 5.2. Chi phí đối sánh đơn (Unary Matching Cost)

Chi phí đối sánh đơn kết hợp ba thành phần hình học:

- **C_pose:** Sai khác giữa tư thế 3D tương đối so với điểm gốc (Root-relative 3D Pose) hoặc hình dạng bộ xương.
- **C_root:** Sai khác vị trí điểm gốc (Root Position) trong không gian 3D.
- **C_2D:** Sai khác giữa các điểm khớp 2D quan sát được và điểm khớp 2D thu được từ phép chiếu lại (2D Reprojection).

Chi phí tổng hợp được tính theo công thức:

C_U = α × C_pose + β × C_root + γ × C_2D

Trong đó:

- α, β, γ là các trọng số không âm.
- Mỗi thành phần chi phí cần được chuẩn hóa hoặc đưa về thang đo có thể so sánh trước khi kết hợp.

Thử nghiệm tìm kiếm tham số ban đầu sẽ sử dụng điều kiện:

α + β + γ = 1

Các tham số sẽ được lựa chọn trên tập validation, không sử dụng tập test để tối ưu.

Khả năng sử dụng thành phần chi phí tái chiếu 2D phụ thuộc vào việc có sẵn dữ liệu quan sát 2D phù hợp và thông tin hiệu chuẩn camera hay không.

### 5.3. Chi phí theo thời gian (Temporal Cost)

Chi phí theo thời gian đo lường mức độ nhất quán của các phép gán định danh giữa những khung hình liên tiếp.

Hàm mục tiêu tổng quát được xây dựng dựa trên ba thành phần:

**Tổng chi phí =**

- Tổng chi phí đối sánh đơn và chi phí bỏ sót người.
- Cộng với λ_T nhân tổng chi phí chuyển tiếp theo thời gian.
- Cộng với c_FP nhân số lượng phát hiện không có đối tượng tương ứng.

Trong đó:

- C_U: Chi phí đối sánh đơn, bao gồm cả chi phí bỏ sót người.
- C_T: Chi phí chuyển tiếp theo thời gian.
- λ_T: Trọng số của chi phí thời gian.
- c_FP: Mức phạt đối với phát hiện không có người tương ứng.

Chi phí bỏ sót người được tính trong thành phần chi phí đơn khi một người trong Ground Truth được gán vào trạng thái bị bỏ sót.

Cấu trúc mạng tối ưu hóa và các ràng buộc chi tiết sẽ được tài liệu hóa trong quá trình phát triển và hoàn thiện hệ thống.

## 6. Các phương pháp đối chứng thực nghiệm (Baselines)

Phương pháp MC-MCF đề xuất cần được so sánh với những chiến lược liên kết đơn giản hơn.

| Phương pháp | Mô tả |
|---|---|
| B1: Greedy Matching | Đối sánh tại từng khung hình bằng chiến lược tham lam. |
| B2: Hungarian Matching | Đối sánh toàn cục tại từng khung hình bằng thuật toán Hungarian, có bổ sung các lựa chọn giả (Dummy Assignment) cho trường hợp bỏ sót và phát hiện không có đối tượng tương ứng. |
| B3: Temporal Optimization | Liên kết định danh ở cấp độ chuỗi video, có sử dụng thông tin theo thời gian. |
| B4: MC-MCF | Phương pháp tối ưu hóa đa hàng hóa được đề xuất, với các ràng buộc dùng chung về năng lực sử dụng phát hiện. |
| B5: Oracle Assignment | Phương pháp tham chiếu hoặc thử nghiệm giới hạn trên có sử dụng thông tin GT, nếu cung cấp thông tin bổ sung so với B2. |
| B6: Original WHAM IDs | Đánh giá trực tiếp các định danh ban đầu của WHAM mà không thực hiện điều chỉnh. |

Mỗi phương pháp đối chứng phải được triển khai với chi phí, ràng buộc và nguồn thông tin đầu vào được mô tả rõ ràng.

Nếu hai phương pháp sử dụng cùng ma trận chi phí và cùng các ràng buộc gán định danh thì cần xem xét chúng là cùng một phương pháp, thay vì trình bày như hai đóng góp nghiên cứu độc lập.

## 7. Các chỉ số đánh giá

### 7.1. Chỉ số đánh giá phát hiện người (Detection Metrics)

- Precision: Độ chính xác của các phát hiện.
- Recall: Khả năng phát hiện đầy đủ người thực tế.
- F1-score: Trung bình điều hòa giữa Precision và Recall.
- False-positive detections: Số lượng phát hiện không có người tương ứng trong GT.
- Missed detections: Số lượng người trong GT không được mô hình phát hiện.

### 7.2. Chỉ số đánh giá định danh (Identity Metrics)

- IDTP: Số lượng phát hiện được gán đúng định danh.
- IDFP: Số lượng phát hiện bị gán sai định danh.
- IDFN: Số lượng phát hiện đúng bị bỏ sót trong quá trình gán định danh.
- IDF1: Chỉ số F1 đánh giá độ chính xác của định danh người.
- Identity Purity: Mức độ thuần nhất của một định danh dự đoán, phản ánh tỷ lệ phát hiện thuộc về cùng một người thực tế.
- Identity Contamination: Mức độ một định danh dự đoán chứa các phát hiện thuộc về nhiều người thực tế khác nhau.
- Identity Fragmentation: Mức độ một người thực tế bị chia thành nhiều định danh dự đoán khác nhau.

Các định danh ban đầu của WHAM sẽ được đánh giá riêng với kết quả gán định danh giữa GT và các phát hiện sau tối ưu hóa.

### 7.3. Chỉ số đánh giá tư thế (Pose Metrics)

- MPJPE: Sai số vị trí trung bình của các khớp 3D.
- PA-MPJPE: Sai số MPJPE sau khi căn chỉnh Procrustes.
- Root-position Error: Sai số vị trí điểm gốc của cơ thể.
- Acceleration Error: Sai số gia tốc của chuyển động.

Các chỉ số phát hiện và định danh đánh giá khả năng phát hiện và liên kết đúng người của mô hình.

Các chỉ số tư thế đánh giá độ chính xác hình học của bộ xương dự đoán.

Cần báo cáo riêng các nhóm chỉ số này vì độ chính xác tư thế tốt không đồng nghĩa với khả năng phát hiện người hoặc duy trì định danh đáng tin cậy.

## 8. Tối ưu hóa siêu tham số (Hyperparameter Optimization)

Thí nghiệm tối ưu hóa đầu tiên tập trung tìm các giá trị phù hợp cho ba trọng số:

- α: Trọng số của chi phí tư thế 3D.
- β: Trọng số của chi phí vị trí điểm gốc 3D.
- γ: Trọng số của chi phí tái chiếu 2D.

Quy trình thực hiện:

1. Nạp kết quả dự đoán WHAM từ các file CSV.
2. Nạp dữ liệu Ground Truth MPI-INF-3DHP từ các file CSV.
3. Ánh xạ các khớp tương ứng giữa hai hệ biểu diễn.
4. Kiểm tra hệ tọa độ, đơn vị tọa độ, thông tin camera và chỉ số khung hình.
5. Tính toán riêng từng thành phần chi phí tư thế, vị trí điểm gốc và tái chiếu.
6. Chuẩn hóa các thành phần chi phí.
7. Tìm kiếm các tổ hợp giá trị ứng viên của α, β và γ.
8. Chạy phương pháp liên kết định danh với từng tổ hợp tham số.
9. Đánh giá kết quả gán định danh bằng các chỉ số trên tập validation.
10. Lựa chọn bộ tham số theo một hàm mục tiêu validation được xác định trước.

Trong giai đoạn khảo sát ảnh hưởng của α, β và γ, các tham số khác như trọng số thời gian và mức phạt đối với phát hiện không có đối tượng tương ứng hoặc bỏ sót người nên được giữ cố định.

Hàm mục tiêu lựa chọn tham số nên dựa trên một chỉ số đánh giá như IDF1 hoặc một điểm số liên kết được xác định trước, thay vì đơn thuần lựa chọn nghiệm có chi phí tối ưu hóa nhỏ nhất.

Việc chia dữ liệu nên được thực hiện theo đối tượng (Subject) hoặc chuỗi video (Sequence) khi có thể, thay vì chia ngẫu nhiên các khung hình riêng lẻ.

## 9. Cấu trúc dự án

```text
monocular-hpe-person-detection-and-identity-reliability/
│
├── data/
│   ├── raw/
│   │   └── mpi_inf_3dhp/
│   │       ├── wham_predictions/
│   │       └── ground_truth/
│   │
│   ├── processed/
│   │   └── mpi_inf_3dhp/
│   │
│   └── calibration/
│       └── mpi_inf_3dhp/
│
├── src/
│   ├── data/
│   │   ├── wham_loader.py
│   │   ├── mpi_gt_loader.py
│   │   ├── mpi_joint_mapping.py
│   │   ├── harmony4d_loader.py
│   │   └── coordinate_transform.py
│   │
│   ├── costs/
│   │   ├── root_cost.py
│   │   ├── pose_cost.py
│   │   ├── reprojection_cost.py
│   │   ├── transition_cost.py
│   │   ├── unary_cost.py
│   │   └── cost_matrix.py
│   │
│   ├── optimization/
│   │   ├── network_builder.py
│   │   ├── mc_mcf_model.py
│   │   ├── constraints.py
│   │   └── solver.py
│   │
│   ├── tuning/
│   │   ├── validation.py
│   │   ├── parameter_search.py
│   │   └── objective.py
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── identity_metrics.py
│   │   ├── identity_purity.py
│   │   ├── contamination.py
│   │   └── pose_metrics.py
│   │
│   └── main.py
│
├── experiments/
│   ├── configs/
│   │   └── mpi_validation.yaml
│   ├── tune_mpi.py
│   └── run_experiment.py
│
├── outputs/
│   ├── parameter_search/
│   ├── assignments/
│   ├── metrics/
│   └── visualizations/
│
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## 10. Tổ chức dữ liệu

Các file dữ liệu gốc cần được lưu trữ tách biệt với dữ liệu đã qua xử lý.

Ví dụ:

```text
data/raw/mpi_inf_3dhp/
│
├── wham_predictions/
│   ├── RA-WHAM_thongTinVideo_cam1.csv
│   ├── RA-WHAM_thongTinVideo_cam2.csv
│   └── ...
│
└── ground_truth/
    ├── thongTinVideo_cam1.csv
    ├── thongTinVideo_cam2.csv
    └── ...
```

Dữ liệu đã xử lý, kết quả gán định danh trung gian và đầu ra thực nghiệm không được ghi đè lên các file CSV gốc.

Các bộ dữ liệu dung lượng lớn và kết quả được tạo ra trong quá trình thực nghiệm nhìn chung không nên đưa trực tiếp vào Git.

Vị trí lưu trữ, hướng dẫn tải xuống và quy trình tạo dữ liệu cần được mô tả riêng để bảo đảm khả năng tái lập thí nghiệm.

## 11. Khả năng tái lập thực nghiệm (Reproducibility)

Mỗi thí nghiệm cần ghi lại đầy đủ các thông tin sau:

- Định danh bộ dữ liệu và chuỗi video.
- Định danh camera.
- Mô hình và cấu hình suy luận (Inference Configuration).
- Cấu hình ánh xạ khớp (Joint Mapping).
- Quy ước về hệ tọa độ và đơn vị đo.
- Phương pháp chuẩn hóa chi phí.
- Giá trị của α, β và γ.
- Trọng số thời gian và mức phạt đối với các phát hiện không được gán hoặc người bị bỏ sót.
- Bộ giải tối ưu hóa (Solver) và cấu hình của bộ giải.
- Các chỉ số validation và evaluation.

Kết quả tìm kiếm tham số cần được xuất ra định dạng CSV để thuận tiện cho việc phân tích tiếp theo.

Ví dụ:

```text
outputs/
└── parameter_search/
    └── mpi_validation_results.csv
```

## 12. Các giai đoạn phát triển hiện tại

Việc triển khai dự án được thực hiện theo từng giai đoạn, từ tổ chức dữ liệu đến xây dựng và đánh giá phương pháp tối ưu hóa.

| Giai đoạn | Công việc | Trạng thái |
|---|---|---|
| 1 | Tổ chức dữ liệu WHAM và GT của MPI-INF-3DHP | Planned |
| 2 | Xây dựng bộ nạp dữ liệu WHAM CSV | Planned |
| 3 | Xây dựng bộ nạp dữ liệu MPI Ground Truth CSV | Planned |
| 4 | Ánh xạ khớp và kiểm tra hệ tọa độ | Planned |
| 5 | Xây dựng các hàm tính chi phí thành phần | Planned |
| 6 | Xây dựng mạng MC-MCF và các ràng buộc | Planned |
| 7 | Tìm kiếm tham số α, β và γ | Planned |
| 8 | Đánh giá phương pháp và kết quả thực nghiệm | Planned |

---

**Mục tiêu dài hạn:** Xây dựng một khuôn khổ nghiên cứu có khả năng đánh giá, phân tích và cải thiện độ tin cậy trong phát hiện người và duy trì định danh người của các mô hình Monocular 3D Human Pose Estimation, thông qua các phương pháp tối ưu hóa có xét đến thông tin hình học và tính liên tục theo thời gian.