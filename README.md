# Dự báo rời bỏ khách hàng Telco

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Notebook](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Workflow](https://img.shields.io/badge/Workflow-Reproducible%20notebook-2ea44f)](Customer_Churn_Colab_Standalone.ipynb)

Mô hình dự báo xác suất một khách hàng viễn thông sẽ rời bỏ dịch vụ (*customer churn*). Dự án thực hiện khám phá dữ liệu, so sánh ba mô hình phân loại và xuất danh sách khách hàng cần ưu tiên giữ chân để hỗ trợ đội ngũ kinh doanh.

## Tổng quan

Churn làm giảm doanh thu và thường tốn kém hơn nhiều so với việc giữ chân khách hàng hiện tại. Mô hình này nhận diện sớm nhóm có nguy cơ rời bỏ từ thông tin dịch vụ, hợp đồng, thời gian gắn bó và chi phí; ngưỡng dự đoán được chọn theo điểm F2 để ưu tiên **recall** cho các chiến dịch can thiệp giữ chân.

Nguồn dữ liệu là bộ [Telco Customer Churn](https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113) công khai trên Kaggle. Notebook sẽ tự tải tệp Excel khi chạy trên Colab có Internet.

```mermaid
flowchart LR
    A[Dữ liệu Telco Customer Churn] --> B[Kiểm tra dữ liệu & EDA]
    B --> C[Lọc biến rò rỉ dữ liệu]
    C --> D[Chia stratified<br/>Train 60% · Validation 20% · Test 20%]
    D --> E[Tiền xử lý<br/>impute · scale · one-hot]
    E --> F[So sánh mô hình<br/>Logistic Regression · Decision Tree · Random Forest]
    F --> G[Chọn ngưỡng theo F2]
    G --> H[Đánh giá trên test chưa thấy]
    H --> I[Danh sách ưu tiên giữ chân]
```

## Cấu trúc thư mục

```text
.
├── Customer_Churn_Colab_Standalone.ipynb  # Toàn bộ quy trình EDA, huấn luyện và dự báo
├── requirements.txt                       # Các thư viện Python cần thiết
├── LICENSE                                # Giấy phép MIT
└── outputs/                               # Báo cáo, chỉ số và danh sách hành động sinh ra khi chạy
    ├── validation_model_comparison.csv
    ├── threshold_selection.csv
    ├── test_metrics.csv
    ├── classification_report.csv
    └── retention_priority_list.csv
```

- `data/` được notebook tạo tự động và chứa `Telco_customer_churn.xlsx` tải từ Kaggle; thư mục này không được đưa vào repository.
- `outputs/` lưu các bảng kết quả và, sau khi chạy notebook, các biểu đồ EDA/đánh giá (`01_...png` đến `05_test_evaluation.png`).
- Notebook thay thế cho các thư mục `src/`, `notebooks/` và `models/`: toàn bộ pipeline được đóng gói trong một tệp Colab tự chạy.

## Cài đặt

### Chạy trên máy cục bộ

```bash
git clone https://github.com/Tomato2312/Customer-Churn-Prediction-Using-Supervised-Learning
cd PROJECTS-DATAMINING
python -m venv .venv
```

Kích hoạt môi trường ảo:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Cài các thư viện:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Chạy trên Google Colab

1. Tải `Customer_Churn_Colab_Standalone.ipynb` lên Google Colab hoặc mở tệp từ GitHub.
2. Chọn **Runtime → Run all**. Cell đầu tiên cài thư viện, sau đó notebook tải dữ liệu Kaggle công khai và chạy toàn bộ pipeline.
3. Cell cuối tạo `customer_churn_outputs.zip` để tải các kết quả về máy.

## Sử dụng

### Huấn luyện và đánh giá

Notebook là điểm chạy chính. Sau khi đã cài môi trường cục bộ, khởi động Jupyter:

```bash
jupyter notebook Customer_Churn_Colab_Standalone.ipynb
```

Trong giao diện Jupyter, chọn **Run All Cells**. Hoặc chạy không giao diện để tái tạo kết quả:

```bash
jupyter nbconvert --to notebook --execute --inplace Customer_Churn_Colab_Standalone.ipynb
```

Lệnh trên bao gồm huấn luyện, chọn mô hình/ngưỡng và đánh giá; dữ liệu được tải tự động nếu chưa có `data/Telco_customer_churn.xlsx`.

### Dự đoán và ưu tiên giữ chân

Phiên bản hiện tại không có `inference.py` độc lập: cell cuối của notebook chính là bước suy luận. Sau khi chạy, mở danh sách dự đoán đã sắp theo xác suất churn giảm dần:

```python
import pandas as pd

priority = pd.read_csv("outputs/retention_priority_list.csv")
priority.head(20)
```

Các khách hàng có `Action = "Ưu tiên liên hệ giữ chân"` là những trường hợp vượt ngưỡng F2 tối ưu. Để dự báo dữ liệu mới trong phiên bản hiện tại, hãy thay dữ liệu nguồn bằng tệp có cùng schema, rồi chạy lại toàn bộ notebook; pipeline sẽ tự huấn luyện lại và xuất danh sách mới.

## Kết quả

Mô hình tốt nhất trên validation là **Random Forest**. Với ngưỡng F2 tối ưu **0.37** trên tập test chưa thấy, mô hình đạt ROC-AUC **0.836**, PR-AUC **0.635** và bắt được **90.6%** khách hàng thực sự rời bỏ. Accuracy thấp hơn là đánh đổi có chủ đích để giảm nguy cơ bỏ sót khách hàng churn.

| Chỉ số test | Giá trị |
| --- | ---: |
| ROC-AUC | 0.836 |
| PR-AUC | 0.635 |
| Accuracy | 0.627 |
| Precision (churn) | 0.408 |
| Recall (churn) | 0.906 |
| F1-score (churn) | 0.563 |
| F2-score (churn) | 0.729 |

Ma trận nhầm lẫn tại ngưỡng 0.37:

| Thực tế \ Dự đoán | Ở lại | Rời bỏ |
| --- | ---: | ---: |
| Ở lại | 544 | 491 |
| Rời bỏ | 35 | 339 |

Sau khi chạy notebook, biểu đồ tổng hợp confusion matrix, ROC curve và Precision–Recall curve được lưu tại [`outputs/05_test_evaluation.png`](outputs/05_test_evaluation.png). Các bảng số liệu có thể kiểm tra trực tiếp trong [`outputs/`](outputs/).

## Giấy phép

Dự án được phát hành theo [MIT License](LICENSE). Bạn có thể sử dụng, chỉnh sửa và phân phối mã nguồn, miễn là giữ lại thông báo bản quyền và giấy phép.
