# Hướng dẫn nộp bài (tiếng Việt)

## Đề đang yêu cầu gì?

Bạn nộp hai trường: **Project Title** (tên dự án) và **URL** (đường dẫn GitHub repository public). Người khác sẽ kiểm tra 12 tiêu chí trong README, gồm API / model / logging tests, test script một lệnh, performance monitor, tách dữ liệu test, dự đoán theo quốc gia/toàn cầu, ingestion, so sánh nhiều model, EDA, Docker và biểu đồ so với baseline.

- **Project Title:** `AAVAIL Revenue Forecasting – End-to-End AI Business Solution`
- **Repository dự kiến:** `aavail-revenue-forecasting` — URL chỉ điền sau khi repo thực sự được tạo và public.

## Chạy nhanh

```powershell
python -m pip install -r requirements.txt
python run_tests.py
python -m src.train
python -m src.plots
python -m src.app
```

Mở `http://127.0.0.1:8000/`. Dataset kèm theo là **dữ liệu mô phỏng**, nên kết quả test/demo chỉ chứng minh pipeline chạy được, không thể gọi là kết quả đánh giá dữ liệu IBM thật.

## Trước khi nộp bài chính thức

1. Tải data gốc bằng `python scripts/fetch_official_data.py` hoặc tải từ GitHub nguồn nếu script lỗi mạng.
2. Chạy `python -m src.train --data-dir data/official` rồi `python -m src.plots --data-dir data/official`.
3. Chạy `python run_tests.py`, `python -m src.monitor`, kiểm tra biểu đồ PNG và `reports/metrics.json`.
4. Kiểm tra image Docker thực sự build/chạy, rồi tạo repo GitHub **public**; chỉ sau đó mới có URL nộp.
5. Đọc và hiểu kiến trúc, mô hình, tập holdout, sai số. Hãy trình bày đúng những phần mình đã kiểm chứng, đặc biệt đừng nói đã deploy cloud nếu chỉ chạy Docker local.

## Trả lời nhanh các câu hỏi

- **Business problem:** Dự đoán doanh thu 30 ngày nhằm hỗ trợ lập kế hoạch tài chính theo thị trường.
- **Baseline:** Lấy doanh thu của đúng thứ đó tuần trước.
- **Models:** Ridge và Random Forest, chọn model có MAE trên khoảng dữ liệu cuối chưa dùng để train nhỏ nhất.
- **MAE:** Sai số tuyệt đối trung bình; số càng nhỏ càng tốt.
- **Leakage:** Không cho model nhìn dữ liệu tương lai trong lúc đánh giá trên holdout.
- **Unit tests:** Tự động kiểm tra chức năng nhỏ, có API integration test gọi máy chủ HTTP thật.
- **Logs:** Ghi lại lịch sử dự đoán, bảo vệ endpoint đọc log bằng token.
- **Docker:** Đóng gói môi trường và code để chạy nhất quán; chưa có bằng chứng chạy trên cloud.
