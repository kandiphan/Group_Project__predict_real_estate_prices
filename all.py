from sklearn.model_selection import train_test_split
from data import *
from pre_process import *
from encoding_scaling_transformation import *
from model import *


# Tiny transformer used to inverse log1p-transformed targets when saving artifacts.
class YLog1pTransformer:
    """Tiny transformer with inverse_transform compatible interface.

    inverse_transform(X) -> expm1(X)
    """
    def inverse_transform(self, arr):
        import numpy as _np
        a = _np.asarray(arr)
        # ensure 2D shape for compatibility with sklearn-style transformers
        if a.ndim == 1:
            a = a.reshape(-1, 1)
            return _np.expm1(a).ravel()
        return _np.expm1(a)

if __name__ == "__main__":
    # Thông tin MongoDB
    MONGO_CONNECTION = "mongodb+srv://vhyjjj:vhyjjj@ck.usqytco.mongodb.net/?retryWrites=true&w=majority&appName=CK"
    DB_NAME = "data"
    COLLECTION_NAME = "CK"

    # ========================================
    # 1. LOAD DỮ LIỆU
    # ========================================
    df = run_full_pipeline(
        connection_string=MONGO_CONNECTION,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        skip_crawl=True,
        skip_upload=True
    )

    # Basic preprocessing
    df = basic_preprocessing(df)
    X = df.drop(columns=['price', 'price_string'], errors='ignore')
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ========================================
    # 2. CLEANING & OUTLIER FILTERING
    # ========================================
    X_train_clean, X_test_clean, y_train_clean, y_test_clean, artifacts = initial_cleaning_and_outlier_filtering(
        X_train, X_test, y_train, y_test,
        target_name='price',
        category_col='category_name'
    )

    # ========================================
    # 3. FINAL PIPELINE: ÁP DỤNG LOG1P CHO PRICE & SIZE → ENCODE + SCALE
    # ========================================
    X_train_final, X_test_final, y_train_final, y_test_final, preprocessor, artifacts = \
        final_transformation_pipeline(
            X_train_clean,
            X_test_clean,
            y_train_clean,
            y_test_clean
        )

    # Nếu pipeline đã áp dụng log1p lên target, chuẩn bị một y_transformer
    # tương thích cho việc lưu artifacts và cho hàm make_prediction.
    if artifacts.get('y_transform') == 'log1p':
        y_transformer_to_save = YLog1pTransformer()
    else:
        y_transformer_to_save = None

    # ========================================
    # 4. HUẤN LUYỆN CATBOOST TRÊN TARGET (đã transform nếu có)
    # ========================================
    best_model = train_and_save_model(
        X_train_final=X_train_final,
        y_train_final=y_train_final,  # Đã được transform (log1p) nếu pipeline yêu cầu
        preprocessor=preprocessor,
        y_transformer=y_transformer_to_save,
        artifacts=artifacts
    )

    print("\n\n--- QUY TRÌNH CHÍNH HOÀN TẤT ---")
    print("Mô hình CatBoost đã được huấn luyện và lưu trữ.")
    print("Mô hình được huấn luyện trên target đã transform (nếu có).")
    print("Khi dùng model để dự đoán, hãy dùng hàm inverse phù hợp (ví dụ expm1 cho log1p).")