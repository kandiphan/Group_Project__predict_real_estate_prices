# Dự đoán giá Bất động sản

Hệ thống dự đoán giá bất động sản sử dụng mô hình CatBoost, được huấn luyện trên dữ liệu từ Chợ Tốt.

## Cài đặt

1. Clone repository:

```bash
git clone <repository-url>
cd Final
```

2. Tạo và kích hoạt môi trường ảo Python:

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

1. Huấn luyện mô hình:

```bash
python all.py
```

2. Chạy ứng dụng web:

```bash
streamlit run save.py
```

## Cấu trúc dự án

- `all.py`: Pipeline huấn luyện mô hình
- `model.py`: Định nghĩa và lưu trữ mô hình
- `web.py`: Giao diện web Streamlit
- `data.py`: Thu thập và xử lý dữ liệu
- `encoding_scaling_transformation.py`: Biến đổi dữ liệu
- `tests/`: Unit tests
- `model_artifacts/`: Thư mục chứa mô hình đã huấn luyện

## Đặc điểm

- Log transform cho biến price và size
- CatBoost Regressor
- Streamlit UI
- MongoDB integration
- Unit tests

## Lưu ý

- Cần có kết nối MongoDB để thu thập dữ liệu
- Dự đoán giá trong khoảng hợp lý
- Xử lý ngoại lệ đầy đủ
