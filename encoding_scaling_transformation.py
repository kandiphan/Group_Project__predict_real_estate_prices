"""encoding_scaling_transformation.py
Final transformation pipeline that applies log1p to the target (price) and
to the 'size' feature, then performs standard scaling and one-hot encoding
for other features.

This module provides:
- final_transformation_pipeline(...)
- inverse_transform_predictions(...)

"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats


def transform_for_prediction(
    df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None
) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, Any]]:
    """
    Version đơn giản của final_transformation_pipeline cho prediction.
    Nhận vào một DataFrame và áp dụng các biến đổi cần thiết.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần transform
    preprocessor : ColumnTransformer, optional
        Preprocessor đã fit. Nếu None, một preprocessor mới sẽ được tạo và fit.

    Returns:
    --------
    transformed_df : pd.DataFrame
        DataFrame sau khi đã transform
    preprocessor : ColumnTransformer
        Preprocessor đã fit
    artifacts : Dict[str, Any]
        Thông tin về các biến đổi đã áp dụng
    """
    df = df.copy()
    artifacts = {}
    
    # Kiểm tra và thêm các cột thiếu với giá trị None
    if preprocessor is not None:
        # Lấy danh sách các cột từ preprocessor đã fit
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][2]
        required_columns = list(numeric_features) + list(categorical_features)
        
        # Thêm các cột thiếu
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                print(f"Added missing column: {col}")

    # Transform size with log1p if present
    if 'size' in df.columns:
        try:
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            df['size'] = np.log1p(df['size'].clip(lower=0))
            artifacts.setdefault('X_transforms', {})['size'] = 'log1p'
        except Exception as e:
            print(f"Error transforming size column: {str(e)}")
            df['size'] = None

    # Convert categorical columns to string safely
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        try:
            df[col] = df[col].fillna('unknown').astype(str)
        except Exception as e:
            print(f"Error converting column {col} to string: {str(e)}")
            df[col] = 'unknown'

    # If no preprocessor provided, create and fit a new one
    if preprocessor is None:
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = cat_cols

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        preprocessor.fit(df)

    # Transform the data
    df_transformed = preprocessor.transform(df)

    # Get feature names
    if len(cat_cols) > 0:
        onehot_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        final_feature_names = numeric_features + list(onehot_names)
    else:
        final_feature_names = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    df_final = pd.DataFrame(df_transformed, columns=final_feature_names)

    return df_final, preprocessor, artifacts

def final_transformation_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None,
    y_test: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, Dict[str, Any]]:
    """
    Final pipeline that:
    - Applies log1p to the target (y)
    - Applies log1p to the 'size' feature in X
    - Applies StandardScaler to numeric features and OneHotEncoder to categorical

    Returns transformed X_train, X_test, transformed y_train, y_test,
    fitted preprocessor and an artifacts dict describing transforms.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    artifacts: Dict[str, Any] = {}

    print("=" * 80)
    print("FINAL TRANSFORMATION PIPELINE (LOG1P for price & size)")
    print("=" * 80 + "\n")

    # ---------------------------
    # Transform target with log1p
    # ---------------------------
    y_train_transformed = None
    y_test_transformed = None

    if y_train is not None:
        y_train_safe = y_train.clip(lower=0)
        y_train_transformed = np.log1p(y_train_safe)
        if y_test is not None:
            y_test_safe = y_test.clip(lower=0)
            y_test_transformed = np.log1p(y_test_safe)

        print("Applied log1p to target (price)")
        print(f"  Original target: min={y_train.min():,.0f}, max={y_train.max():,.0f}, mean={y_train.mean():,.0f}")
        print(f"  Transformed target (log1p): min={y_train_transformed.min():.4f}, max={y_train_transformed.max():.4f}, mean={y_train_transformed.mean():.4f}")
        print(f"  Skewness (original): {stats.skew(y_train):.2f}, Skewness (log1p): {stats.skew(y_train_transformed):.2f}\n")

    artifacts['y_transform'] = 'log1p'

    # ---------------------------
    # Transform size with log1p
    # ---------------------------
    print("Applying log1p to feature 'size' (if present)")
    if 'size' in X_train.columns:
        orig_size = X_train['size'].copy()
        print(f"  Original Size: min={orig_size.min():.1f}, max={orig_size.max():.1f}, mean={orig_size.mean():.1f}")
        print(f"  Original Skewness: {stats.skew(orig_size.dropna()):.2f}")

        X_train['size'] = np.log1p(X_train['size'].clip(lower=0))
        X_test['size'] = np.log1p(X_test['size'].clip(lower=0))

        trans_size = X_train['size']
        print(f"  Transformed Size (log1p): min={trans_size.min():.4f}, max={trans_size.max():.4f}, mean={trans_size.mean():.4f}")
        print(f"  Skewness (log1p): {stats.skew(trans_size.dropna()):.2f}\n")
        artifacts.setdefault('X_transforms', {})['size'] = 'log1p'
    else:
        print("  No 'size' column found; skipping size transform.\n")

    # ---------------------------
    # Categorical conversion
    # ---------------------------
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print(f"  Converted {len(cat_cols)} categorical columns to string.\n")

    # ---------------------------
    # Encoding & scaling
    # ---------------------------
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = cat_cols

    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # FIT/TRANSFORM
    print("  Fitting preprocessor on X_train...")
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # feature names
    if len(categorical_features) > 0:
        onehot_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        final_feature_names = numeric_features + list(onehot_names)
    else:
        final_feature_names = numeric_features

    X_train_final = pd.DataFrame(X_train_transformed, columns=final_feature_names)
    X_test_final = pd.DataFrame(X_test_transformed, columns=final_feature_names)

    print(f"  Total features after encoding: {X_train_final.shape[1]}\n")

    print("=" * 80)
    print("TRANSFORMATION PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Train shape: {X_train_final.shape}")
    print(f"Test shape: {X_test_final.shape}")

    if y_train_transformed is not None:
        print(f"\nOriginal target summary: min={y_train.min():,.0f}, max={y_train.max():,.0f}, mean={y_train.mean():,.0f}")

    print(f"\nArtifacts: {artifacts}\n")

    return X_train_final, X_test_final, y_train_transformed, y_test_transformed, preprocessor, artifacts


def inverse_transform_predictions(y_pred_transformed: np.ndarray) -> float:
    """
    Inverse the model output which was produced on log1p(price).

    Accepts a scalar or array-like. Returns a float (price in original VNĐ scale).
    """
    arr = np.asarray(y_pred_transformed)
    inv = np.expm1(arr)
    if inv.size > 1:
        return float(inv.ravel()[0])
    else:
        return float(inv)
