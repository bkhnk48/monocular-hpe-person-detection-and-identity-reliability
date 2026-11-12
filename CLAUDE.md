# Quy tắc cho Claude trong repo này

- Khi sửa `pose_assoc/` (tham số, công thức chi phí, bộ giải, CLI) phải cập nhật README.md
  trong cùng commit: công thức (mục 3), bảng tham số (mục 6), cấu trúc mã (mục 8), lệnh chạy (mục 9).
- Không đặt lại tên/ý nghĩa ký hiệu đã có trong README (α = pose, γ = temporal, ...).
  Thêm tham số mới thì thêm ký hiệu mới và ghi vào README.
- Trước khi báo xong: chạy `python3 -m pytest -q tests/test_readme_sync.py` và báo kết quả.
- Dùng `/usr/bin/python3` (Python 3.9, không dùng cú pháp `X | Y` ở chú thích kiểu lúc chạy).
- Khi đổi chi phí/bộ giải: chạy lại ít nhất một sequence đã gán nhãn
  (`python3 -m pose_assoc.build_network --seq h4d_041_grappling2_cam07`) và so hàm mục tiêu, độ đúng với lần trước.
