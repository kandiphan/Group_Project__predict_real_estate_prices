import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from catboost import CatBoostRegressor
from encoding_scaling_transformation import final_transformation_pipeline

# --- CẤU HÌNH LƯU TRỮ ---
ARTIFACTS_DIR = 'model_artifacts'
MODEL_FILENAME = 'catboost_model.joblib'
PREPROCESSOR_FILENAME = 'preprocessor.joblib'
Y_TRANSFORMER_FILENAME = 'y_transformer.joblib'
ARTIFACTS_FILENAME = 'artifacts.joblib'


def train_and_save_model(
    X_train_final: pd.DataFrame,
    y_train_final: pd.Series,
    preprocessor: ColumnTransformer,
    y_transformer: Any,
    artifacts: Dict[str, Any] = None,
    model_name: str = 'CatBoost'
) -> CatBoostRegressor:
    """
    Huấn luyện mô hình CatBoost trên dữ liệu đã xử lý và lưu trữ
    mô hình cùng các bộ biến đổi.
    """
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        
    print(f"\n--- Bắt đầu huấn luyện mô hình {model_name} ---")
    
    # Khởi tạo mô hình CatBoost
    model = CatBoostRegressor(
        random_state=42, 
        verbose=100, # In log mỗi 100 lần lặp
        loss_function='RMSE', 
        # CatBoost có thể tận dụng GPU nếu có, và tối ưu hóa cho dữ liệu của bạn
    )
    
    # Huấn luyện mô hình
    model.fit(X_train_final, y_train_final)
    
    print("✅ Huấn luyện hoàn tất. Bắt đầu lưu trữ artifacts...")
    
    # --- LƯU TRỮ ARTIFACTS ---
    try:
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, MODEL_FILENAME))
        joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILENAME))
        # y_transformer may be None or a small wrapper (e.g. YLog1pTransformer)
        joblib.dump(y_transformer, os.path.join(ARTIFACTS_DIR, Y_TRANSFORMER_FILENAME))
        # save artifacts dict if provided so downstream apps know transforms
        if artifacts is not None:
            joblib.dump(artifacts, os.path.join(ARTIFACTS_DIR, ARTIFACTS_FILENAME))

        print(f"✅ Đã lưu trữ mô hình và transformers tại thư mục '{ARTIFACTS_DIR}'")
    except Exception as e:
        print(f"❌ Lỗi khi lưu artifacts: {e}")
        
    return model

# --------------------------------------------------------------------------
# CÁC HÀM TIỆN ÍCH CHO ỨNG DỤNG STREAMLIT
# --------------------------------------------------------------------------

def load_artifacts() -> Dict[str, Any]:
    """Tải mô hình và các bộ biến đổi đã lưu trữ."""
    
    print(f"\n--- Tải Artifacts từ thư mục '{ARTIFACTS_DIR}' ---")
    
    try:
        loaded_artifacts = {
            'model': joblib.load(os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)),
            'preprocessor': joblib.load(os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILENAME)),
            'y_transformer': joblib.load(os.path.join(ARTIFACTS_DIR, Y_TRANSFORMER_FILENAME)),
        }
        # artifacts may be optional
        artifacts_path = os.path.join(ARTIFACTS_DIR, ARTIFACTS_FILENAME)
        if os.path.exists(artifacts_path):
            loaded_artifacts['artifacts'] = joblib.load(artifacts_path)
        else:
            loaded_artifacts['artifacts'] = {}
        print("✅ Tải mô hình và transformers thành công.")
        return loaded_artifacts
    except Exception as e:
        print(f"❌ Lỗi khi tải artifacts: {e}")
        print("Đảm bảo bạn đã chạy hàm train_and_save_model trước đó.")
        return {}


def make_prediction(
    new_data_raw: pd.DataFrame,
    model: CatBoostRegressor,
    preprocessor: ColumnTransformer,
    y_transformer: Any = None,
    artifacts: Dict[str, Any] = None
) -> Tuple[float, float]:
    """
    Thực hiện dự đoán trên dữ liệu thô mới và đưa kết quả về thang đo gốc (VNĐ).
    
    Parameters:
    - new_data_raw: DataFrame 1 hàng chứa dữ liệu thô (chưa scale/encode).
    """
    
    # 1. TIỀN XỬ LÝ (TRANSFORM) DỮ LIỆU MỚI
    # Áp dụng preprocessor đã fit trên X_train
    # If artifacts indicate X transforms (e.g. size: log1p), apply them to new_data_raw
    new_data = new_data_raw.copy()
    try:
        if artifacts and isinstance(artifacts, dict):
            x_trans = artifacts.get('X_transforms', {})
            if x_trans.get('size') == 'log1p' and 'size' in new_data.columns:
                new_data['size'] = np.log1p(new_data['size'].clip(lower=0))

        X_new_processed = preprocessor.transform(new_data)
    except ValueError as e:
        # Xảy ra nếu có lỗi về feature names hoặc columns không khớp
        print(f"❌ Lỗi tiền xử lý dữ liệu mới: {e}")
        return 0.0, 0.0

    # 2. DỰ ĐOÁN
    y_pred_transformed = model.predict(X_new_processed)

    # 3. INVERSE TRANSFORM (Đảo ngược về VNĐ gốc)
    if y_transformer is not None:
        try:
            # Support transformers that accept 1d or 2d arrays
            y_pred_original = y_transformer.inverse_transform(
                np.asarray(y_pred_transformed).reshape(-1, 1)
            ).flatten()
        except Exception:
            # fallback: if transformer expects 1d
            y_pred_original = np.asarray(y_pred_transformed).ravel()
    else:
        # No transformer: assume model predicts VNĐ directly
        y_pred_original = np.asarray(y_pred_transformed).ravel()

    # Trả về giá trị dự đoán
    predicted_price = float(y_pred_original[0])
    
    # Giá trị tham khảo (ví dụ: là log(price_million_per_m2) * 100)
    # Ta lấy giá trị dự đoán chính (VNĐ) làm kết quả cuối cùng.
    return predicted_price, predicted_price # Trả về predicted_price (VNĐ) 2 lần cho đơn giản