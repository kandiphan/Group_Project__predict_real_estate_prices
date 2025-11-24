import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# ============================================================================
# PHẦN 1: CLEANING, OUTLIER FILTERING, VÀ IMPUTATION (ĐÃ SỬA LỖI GROUP IMPUTATION VÀ LỌC TEST)
# ============================================================================
# ============================================================================
# HÀM 1: TIỀN XỬ LÝ CƠ BẢN (Toàn bộ dataset - trước khi chia train/test)
# ============================================================================

def basic_preprocessing(df):
    """
    Tiền xử lý CƠ BẢN - KHÔNG GÂY LEAKAGE
    - Loại bỏ duplicates
    - Drop cột dư thừa
    - Ép kiểu dữ liệu
    - Chuẩn hóa categorical cố định
    """
    print("="*80)
    print("📦 BẮT ĐẦU TIỀN XỬ LÝ CƠ BẢN")
    print("="*80 + "\n")

    df = df.copy()

    # 1️⃣ Loại bỏ trùng list_id
    if 'list_id' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset='list_id', keep='first')
        print(f"🧹 Loại bỏ {before - len(df)} bản ghi trùng list_id\n")


    # 3️⃣ Ép kiểu số
    numeric_cols = ['width', 'length', 'rooms', 'size', 'price_million_per_m2', 'price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"🔢 Ép kiểu số cho các cột: {[c for c in numeric_cols if c in df.columns]}\n")


    # 4️⃣ Chuẩn hóa is_main_street
    if 'is_main_street' in df.columns:
        df['is_main_street'] = df['is_main_street'].replace(
            {'True': 1, 'False': 0, True: 1, False: 0, '': np.nan}
        ).astype('float')
        print("✅ Chuẩn hóa is_main_street thành 0/1\n")


    # 5️⃣ Xử lý legal_doc_text
    if 'legal_doc_text' in df.columns:
        df['legal_doc_text'] = df['legal_doc_text'].fillna('Không rõ')

        mapping = {
            "Không rõ": "Thiếu/Không rõ",
            "Khác": "Thiếu/Không rõ",
            "Giấy tay / Chưa có sổ": "Giấy tay/Chưa có sổ",
            "Hợp đồng mua bán": "Hợp đồng mua bán",
            "Đang chờ sổ": "Đang chờ sổ",
            "Sổ hồng / Sổ đỏ đầy đủ": "Sổ hồng / Sổ đỏ đầy đủ"
        }
        df['legal_doc_text'] = df['legal_doc_text'].replace(mapping)

        ordinal_map = {
            "Thiếu/Không rõ": 0,
            "Giấy tay/Chưa có sổ": 1,
            "Hợp đồng mua bán": 2,
            "Đang chờ sổ": 3,
            "Sổ hồng / Sổ đỏ đầy đủ": 4
        }
        df['legal_doc_encoded'] = df['legal_doc_text'].map(ordinal_map)
        print("✅ Encode legal_doc_text\n")

    print(f"📦 Kích thước sau tiền xử lý: {df.shape}")
    print("="*80 + "\n")

    return df

def initial_cleaning_single(
    df: pd.DataFrame,
    target_name: str = 'price',
    category_col: str = 'category_name'
) -> pd.DataFrame:
    """
    Version đơn giản cho prediction - chỉ nhận vào một DataFrame
    """
    df = df.copy()
    
    # Không lọc outliers cho prediction
    # Chỉ áp dụng các bước cleaning cơ bản
    
    # Đảm bảo các kiểu dữ liệu đúng
    if 'rooms' in df.columns:
        df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    
    # Fill NA values
    df = df.fillna({
        'direction_text': 'Không có thông tin',
        'legal_doc_text': 'Không có thông tin',
        'region_name': 'Không có thông tin',
        'area_name': 'Không có thông tin',
        'ward_name': 'Không có thông tin'
    })
    
    return df

def initial_cleaning_and_outlier_filtering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str = 'price',
    category_col: str = 'category_name'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:

    # Bắt buộc: Copy và Reset Index ban đầu để đảm bảo các Series/DataFrame khớp nhau.
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    artifacts = {}
    print("="*80)
    print("🧹 CLEANING & OUTLIER FILTERING (BY CATEGORY)")
    print("="*80 + "\n")

    # ----------------------------------------
    # BƯỚC 0: TIỀN XỬ LÝ CHUẨN BỊ (Lọc cột category nếu không có)
    # ----------------------------------------
    use_category = category_col in X_train.columns
    if not use_category:
        print(f"⚠️ Cảnh báo: Không tìm thấy cột '{category_col}'. Xử lý outliers trên toàn bộ dữ liệu.")

    # ----------------------------------------
    # BƯỚC 1: XỬ LÝ OUTLIERS (THEO CATEGORY HOẶC TOÀN BỘ)
    # ----------------------------------------
    print("📌 BƯỚC 1: Lọc Outliers")
    print("-" * 40)

    outlier_cols = ['size', 'price_million_per_m2', 'width', 'length', 'rooms']

    def get_outlier_bounds(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    category_outlier_bounds = {}


    # Khởi tạo mask toàn cục cho Train và Test (dùng index để áp dụng)
    train_indices = X_train.index.copy()
    test_indices = X_test.index.copy()

    # 1.1 Lọc Outlier Target (Price)
    # Tính ngưỡng trên tập Train ban đầu
    lower_y, upper_y = get_outlier_bounds(y_train)

    # Lọc Train
    mask_y_train = (y_train >= lower_y) & (y_train <= upper_y)
    train_indices = train_indices[mask_y_train]

    # Lọc Test
    mask_y_test = (y_test.reindex(X_test.index) >= lower_y) & (y_test.reindex(X_test.index) <= upper_y) # Reindex an toàn
    test_indices = test_indices[mask_y_test]

    artifacts['target_outlier_bounds'] = {'lower': lower_y, 'upper': upper_y}
    print(f"   Lọc Outlier (Price): Loại {len(X_train) - len(train_indices)} (Train) | Loại {len(X_test) - len(test_indices)} (Test)")


    # 1.2 Lọc Outliers từ FEATURES (Tương tự, dùng ngưỡng chung từ Train)
    feature_outlier_bounds = {}

    if use_category:
        # Nếu dùng category, cần tính bounds cho mỗi category và áp dụng cho cả X/Y
        # Logic này quá phức tạp và dễ lỗi. Đề xuất Dùng Outlier Capping (giới hạn) thay vì Lọc
        # Tạm thời, ta dùng logic lọc toàn bộ đơn giản (fallback) để giữ cho hàm này là bước lọc.
        categories = X_train[category_col].unique()

        # NOTE: Đối với Outlier theo nhóm, nên dùng CAPPING (giới hạn giá trị) trong Pipeline
        # thay vì Filtering, vì Filtering theo nhóm rất dễ lỗi và làm mất dữ liệu.
        # Ở đây, ta chỉ lọc chung theo ngưỡng sau khi price đã lọc.

    for col in outlier_cols:
        if col in X_train.columns and X_train.loc[train_indices, col].notna().sum() > 0:
            # Tính ngưỡng trên tập Train đã lọc Outlier Price
            lower, upper = get_outlier_bounds(X_train.loc[train_indices, col].dropna())
            feature_outlier_bounds[col] = {'lower': lower, 'upper': upper}

            # Lọc Train
            mask_train = ((X_train[col] >= lower) & (X_train[col] <= upper)) | X_train[col].isna()
            train_indices = train_indices[mask_train.loc[train_indices]]

            # Lọc Test
            mask_test = ((X_test[col] >= lower) & (X_test[col] <= upper)) | X_test[col].isna()
            test_indices = test_indices[mask_test.loc[test_indices]]

            print(f"   Lọc Outlier ({col}): Train size còn {len(train_indices)} | Test size còn {len(test_indices)}")

    artifacts['feature_outlier_bounds'] = feature_outlier_bounds

    # ÁP DỤNG LỌC CUỐI CÙNG
    X_train = X_train.loc[train_indices]
    y_train = y_train.loc[train_indices]
    X_test = X_test.loc[test_indices]
    y_test = y_test.loc[test_indices]

    print(f"\n✅ Train size cuối sau lọc: {X_train.shape}")
    print(f"✅ Test size cuối sau lọc: {X_test.shape}\n")

    # ----------------------------------------
    # BƯỚC 2: XỬ LÝ MISSING VALUES (Imputation)
    # ----------------------------------------
    print("="*80)
    print("📌 BƯỚC 2: Xử lý Missing Values (Imputation)")
    print("="*80 + "\n")

    train_missing_before = X_train.isna().sum().sum()

    if train_missing_before > 0 or X_test.isna().sum().sum() > 0:
        fill_values = {}

        # 2.1 is_main_street (Fill bằng 0/False)
        if 'is_main_street' in X_train.columns:
            fill_val = 0.0
            X_train.loc[:, 'is_main_street'] = X_train['is_main_street'].fillna(fill_val)
            X_test.loc[:, 'is_main_street'] = X_test['is_main_street'].fillna(fill_val)
            fill_values['is_main_street'] = fill_val
            print("✅ Filled 'is_main_street' with 0.0\n")

        # 2.2 width, length, rooms theo Group Mean/Overall Mean
        fill_cols = ['width', 'length', 'rooms']
        group_cols = ['region_name', 'area_name']
        available_groups = [c for c in X_train.columns if c in group_cols]

        if available_groups:
            print(f"🔄 Group Imputation theo: {available_groups}")

        for col in fill_cols:
            if col in X_train.columns:
                missing_train = X_train[col].isna().sum()
                missing_test = X_test[col].isna().sum()

                if missing_train == 0 and missing_test == 0:
                    continue

                # Tính Overall Mean TỪ TRAIN
                overall_mean = X_train[col].mean()

                if available_groups and not np.isnan(overall_mean):
                    # Tính group means TỪ TRAIN (bỏ qua NaN để tính mean đúng)
                    train_group_means = X_train.groupby(available_groups)[col].mean()
                    train_group_means_dict = train_group_means.to_dict()

                    # Hàm điền khuyết
                    def group_imputer(row):
                        if pd.isna(row[col]):
                            key = tuple(row[g] for g in available_groups)
                            # Sử dụng group mean. Nếu group mean là NaN, fallback về Overall Mean
                            group_val = train_group_means_dict.get(key, overall_mean)
                            return group_val if not np.isnan(group_val) else overall_mean
                        return row[col]

                    # Áp dụng cho TRAIN và TEST
                    X_train.loc[:, col] = X_train.apply(group_imputer, axis=1)
                    X_test.loc[:, col] = X_test.apply(group_imputer, axis=1)

                # BƯỚC CUỐI: FILL BẤT KỲ NaN CÒN LẠI BẰNG OVERALL MEAN (sau group imputation)
                X_train.loc[:, col] = X_train[col].fillna(overall_mean)
                X_test.loc[:, col] = X_test[col].fillna(overall_mean)

                fill_values[col] = overall_mean
                print(f"  ✅ {col}: Filled {missing_train} (train) + {missing_test} (test) | Final Mean: {overall_mean:.2f}")

        # 2.3 Các cột số khác (Median)
        numeric_cols_rest = [c for c in X_train.select_dtypes(include=['float64', 'int64']).columns if c not in fill_cols and c != 'is_main_street']

        if numeric_cols_rest:
            print(f"\n🔢 Median Imputation cho các cột còn lại:")

        for col in numeric_cols_rest:
            missing_train = X_train[col].isna().sum()
            if missing_train > 0 or X_test[col].isna().sum() > 0:
                median_val = X_train[col].median()
                X_train.loc[:, col] = X_train[col].fillna(median_val)
                X_test.loc[:, col] = X_test[col].fillna(median_val)
                fill_values[col] = median_val
                print(f"  ✅ {col}: Filled {missing_train} (train) | Median: {median_val:.2f}")

        # 2.4 Categorical (Mode)
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) > 0:
            print(f"\n📋 Mode Imputation cho categorical:")

        for col in cat_cols:
            missing_train = X_train[col].isna().sum()
            if missing_train > 0 or X_test[col].isna().sum() > 0:
                mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
                X_train.loc[:, col] = X_train[col].fillna(mode_val)
                X_test.loc[:, col] = X_test[col].fillna(mode_val)
                fill_values[col] = mode_val
                print(f"  ✅ {col}: Filled {missing_train} (train) | Mode: {mode_val}")

        artifacts['fill_values'] = fill_values

    # ----------------------------------------
    # KIỂM TRA CUỐI CÙNG
    # ----------------------------------------
    train_missing_after = X_train.isna().sum().sum()
    test_missing_after = X_test.isna().sum().sum()

    print(f"\n{'='*80}")
    print("✅ TỔNG KẾT")
    print(f"{'='*80}")
    print(f"Missing sau imputation:")
    print(f"  Train: {train_missing_after} (Phải là 0)")
    print(f"  Test: {test_missing_after} (Phải là 0)")

    if train_missing_after > 0 or test_missing_after > 0:
        print("\n⚠️ VẪN CÒN MISSING VALUES SAU IMPUTATION! Cần kiểm tra lại các cột.")

    print(f"\nKích thước cuối:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"\nArtifacts: {list(artifacts.keys())}\n")

    return X_train, X_test, y_train, y_test, artifacts