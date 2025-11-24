"""
app.py
======
Streamlit app cho dự đoán giá BĐS với ML pipeline tích hợp
"""

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import warnings 
warnings.filterwarnings('ignore')

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================

st.set_page_config(
    page_title="Dự đoán giá BĐS",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN")
st.markdown("---")

# ============================================================================
# HELPER FUNCTIONS - DATA PROCESSING
# ============================================================================

def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing steps"""
    df = df.copy()
    
    # Xử lý missing values
    df['direction'] = df['direction'].fillna('unknown')
    df['direction_text'] = df['direction'].fillna('unknown')
    df['property_legal_document'] = df['property_legal_document'].fillna('unknown')
    df['legal_doc_text'] = df['property_legal_document'].fillna('unknown')
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    
    # Chuyển đổi kiểu dữ liệu
    for col in ['width', 'length', 'size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def transform_for_prediction(
    df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    artifacts: Optional[Dict] = None
) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, Any]]:
    """Transform data for prediction"""
    df = df.copy()
    if artifacts is None:
        artifacts = {}

    # Transform size with log1p
    if 'size' in df.columns:
        df['size'] = np.log1p(df['size'].clip(lower=0))
        artifacts['X_transforms'] = {'size': 'log1p'}

    # Prepare numeric and categorical features
    numeric_features = ['size', 'width', 'length', 'rooms', 'is_main_street']
    categorical_features = ['category_name', 'direction', 'property_legal_document', 
                          'region_name', 'area_name', 'ward_name']
                          
    # Convert categorical columns to string
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)
            
    # If no preprocessor is provided, create a new one
    if preprocessor is None:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

    # Transform data
    try:
        # Thử transform trước (nếu preprocessor đã fit)
        df_transformed = preprocessor.transform(df)
    except (AttributeError, NotFittedError):
        # Nếu chưa fit, thực hiện fit_transform
        df_transformed = preprocessor.fit_transform(df)
    
    try:
        # Lấy feature names từ categorical transformer
        onehot_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(onehot_features)
    except (AttributeError, KeyError):
        # Fallback nếu không thể lấy feature names
        feature_names = [f"feature_{i}" for i in range(df_transformed.shape[1])]
    
    # Create DataFrame with feature names
    df_final = pd.DataFrame(df_transformed, columns=feature_names)
    
    return df_final, preprocessor, artifacts

# ============================================================================
# ML PIPELINE
# ============================================================================

@st.cache_resource
def train_model():
    """Train the model and return necessary artifacts"""
    
    # Thông tin MongoDB
    MONGO_CONNECTION = "mongodb+srv://vhyjjj:vhyjjj@ck.usqytco.mongodb.net/?retryWrites=true&w=majority&appName=CK"
    DB_NAME = "data"
    COLLECTION_NAME = "CK"
    
    # Load và xử lý dữ liệu
    with st.spinner("Đang tải và xử lý dữ liệu..."):
        from pymongo import MongoClient
        
        # Kết nối MongoDB
        client = MongoClient(MONGO_CONNECTION)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Load data
        data = list(collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        
        # Basic preprocessing
        df = basic_preprocessing(df)
        
        # Prepare X and y
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'].notna()]  # Remove rows with invalid price
        
        # Keep only rows with valid size
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df = df[df['size'].notna() & (df['size'] > 0)]
        
        # Convert numeric columns including binary features
        numeric_cols = ['width', 'length', 'rooms', 'is_main_street']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == 'is_main_street':
                df[col] = df[col].fillna(0).astype(float)  # Default to 0 for missing values
        
        # Convert categorical columns to string
        cat_cols = ['category_name', 'direction', 'property_legal_document', 
                   'region_name', 'area_name', 'ward_name']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
        
        # Keep a copy of the full DataFrame for suggestions
        df_full = df.copy()
        
        # Prepare X and y
        X = df.drop(columns=['price_string', 'description'], errors='ignore')
        y = np.log1p(df['price'])  # Log transform target
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize và fit preprocessor
        numeric_features = ['size', 'width', 'length', 'rooms', 'is_main_street']
        categorical_features = ['category_name', 'direction', 'property_legal_document', 
                              'region_name', 'area_name', 'ward_name']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        # Ensure 'size' uses the same transformation as prediction (log1p)
        if 'size' in X_train.columns:
            X_train = X_train.copy()
            X_train['size'] = np.log1p(X_train['size'].clip(lower=0))

        # Fit preprocessor trên toàn bộ data và transform training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        # Lưu thông tin features
        artifacts = {
            'features': {
                'numeric': numeric_features,
                'categorical': categorical_features
            }
        }
        
        # Train model
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            verbose=False
        )
        
        model.fit(X_train_transformed, y_train)
        
        return model, preprocessor, artifacts, df_full

# Train model and get artifacts
model, preprocessor, artifacts, df = train_model()

if model is None:
    st.error("❌ Không thể train model!")
    st.stop()

st.success("✅ Đã train xong model!")

# ------------------
# Compute defaults for optional inputs
# ------------------
numeric_default_cols = ['width', 'length', 'rooms']
categorical_default_cols = ['category_name', 'direction', 'property_legal_document',
                            'region_name', 'area_name', 'ward_name']
binary_default_cols = ['is_main_street']

# numeric medians
numeric_medians = {}
for col in numeric_default_cols:
    if col in df.columns and not df[col].dropna().empty:
        try:
            numeric_medians[col] = float(df[col].median())
        except Exception:
            numeric_medians[col] = 0.0
    else:
        numeric_medians[col] = 0.0

# categorical modes
categorical_modes = {}
for col in categorical_default_cols:
    if col in df.columns and not df[col].dropna().empty:
        try:
            categorical_modes[col] = str(df[col].mode().iloc[0])
        except Exception:
            categorical_modes[col] = 'unknown'
    else:
        categorical_modes[col] = 'unknown'

# binary defaults (most frequent)
binary_modes = {}
for col in binary_default_cols:
    if col in df.columns and not df[col].dropna().empty:
        try:
            binary_modes[col] = int(df[col].mode().iloc[0])
        except Exception:
            binary_modes[col] = 0
    else:
        binary_modes[col] = 0

# Mappings for direction and legal docs (same as crawler mapping)
direction_map = {
    1: "Đông", 2: "Tây", 3: "Nam", 4: "Bắc",
    5: "Đông-Bắc", 6: "Tây-Bắc", 7: "Đông-Nam", 8: "Tây-Nam"
}
legal_doc_map = {
    1: "Sổ hồng / Sổ đỏ đầy đủ", 2: "Giấy tay / Chưa có sổ",
    3: "Đang chờ sổ", 4: "Hợp đồng mua bán", 5: "Khác"
}

def build_direction_options(df, option_none="Không có thông tin"):
    # collect unique codes present in df['direction'] (as numbers or strings) or fallback to all
    codes = set()
    if 'direction' in df.columns:
        codes.update([int(x) for x in pd.to_numeric(df['direction'], errors='coerce').dropna().unique()])
    # if none found, use keys from mapping
    if not codes:
        codes = set(direction_map.keys())

    # build display options as '1 - Đông'
    opts = [option_none]
    for c in sorted(codes):
        label = direction_map.get(int(c), str(c))
        opts.append(f"{int(c)} - {label}")
    return opts

def parse_direction_selection(sel):
    # sel like '1 - Đông' or option_none
    if sel is None or sel == "Không có thông tin":
        return None
    try:
        return str(int(str(sel).split('-')[0].strip()))
    except Exception:
        return str(sel)

def build_legal_options(df, option_none="Không có thông tin"):
    codes = set()
    if 'property_legal_document' in df.columns:
        codes.update([int(x) for x in pd.to_numeric(df['property_legal_document'], errors='coerce').dropna().unique()])
    if not codes:
        codes = set(legal_doc_map.keys())
    opts = [option_none]
    for c in sorted(codes):
        label = legal_doc_map.get(int(c), str(c))
        opts.append(f"{int(c)} - {label}")
    return opts

def parse_legal_selection(sel):
    if sel is None or sel == "Không có thông tin":
        return None
    try:
        return str(int(str(sel).split('-')[0].strip()))
    except Exception:
        return str(sel)


# ============================================================================
# HELPER FUNCTIONS - UI
# ============================================================================

def get_area_options(df, region_name, option_none="Không có thông tin"):
    """Lấy danh sách quận/huyện theo tỉnh/thành"""
    if region_name == option_none or region_name is None:
        areas = df['area_name'].dropna().unique()
    else:
        areas = df[df['region_name'] == region_name]['area_name'].dropna().unique()
    return [option_none] + sorted(areas.tolist())

def get_ward_options(df, region_name, area_name, option_none="Không có thông tin"):
    """Lấy danh sách phường/xã theo quận/huyện và tỉnh/thành"""
    df_filtered = df.copy()
    
    if region_name != option_none and region_name is not None:
        df_filtered = df_filtered[df_filtered['region_name'] == region_name]
    
    if area_name != option_none and area_name is not None:
        df_filtered = df_filtered[df_filtered['area_name'] == area_name]
    
    wards = df_filtered['ward_name'].dropna().unique()
    return [option_none] + sorted(wards.tolist())

def format_price(price):
    """Format giá tiền VNĐ"""
    if price >= 1_000_000_000:
        return f"{price/1_000_000_000:.2f} tỷ VNĐ"
    else:
        return f"{price/1_000_000:.0f} triệu VNĐ"

# ============================================================================
# SIDEBAR - THÔNG TIN
# ============================================================================

with st.sidebar:
    st.header("ℹ️ Thông tin")
    st.markdown("""
    ### Hướng dẫn sử dụng:
    1. Nhập các thông tin BĐS
    2. Các trường có (*) là bắt buộc
    3. Các trường trống sẽ được điền vào bằng
        * Med cho các cột số
        * Mode cho các cột phân loại \n
    _(giả định: đây là các yếu tố phù hợp với đại đa số người dùng nhất)_
    
    4. Nhấn "DỰ ĐOÁN GIÁ" để xem kết quả
    ### Thống kê dataset:
    """)
    st.metric("Tổng số BĐS", f"{len(df):,}")
    st.metric("Số loại hình", df['category_name'].nunique())
    st.metric("Số tỉnh/thành", df['region_name'].nunique())
    
    st.markdown("---")
    st.markdown("© 2025 - Real Estate Price Prediction")

# ============================================================================
# FORM NHẬP LIỆU
# ============================================================================

st.header("📝 Nhập thông tin Bất động sản")

with st.form("input_form"):
    
    # Chuẩn bị options
    option_none = "Không có thông tin"
    
    opt_category = sorted(df[
        df['category_name'] != option_none
    ]['category_name'].dropna().unique())
    
    # Convert to string before sorting to avoid type mixing
    opt_rooms = [option_none] + sorted([str(x) for x in df['rooms'].dropna().unique()])
    # Build human-readable direction and legal options (display labels), parse back to codes later
    opt_direction = build_direction_options(df, option_none)
    opt_legal = build_legal_options(df, option_none)
    opt_region = [option_none] + sorted([str(x) for x in df['region_name'].dropna().unique()])
    
    # ========================================
    # THÔNG TIN BẮT BUỘC
    # ========================================
    st.subheader("Thông tin bắt buộc (*)")
    
    col1, col2 = st.columns(2)
    with col1:
        category_name = st.selectbox(
            "Loại hình BĐS (*)", 
            options=opt_category,
            help="Chọn loại hình bất động sản"
        )
    with col2:
        size = st.number_input(
            "Diện tích (m²) (*)", 
            min_value=1.0, 
            max_value=10000.0,
            value=50.0, 
            step=5.0,
            help="Nhập diện tích đất/nhà"
        )
    
    # ========================================
    # VỊ TRÍ (LỌC ĐỘNG)
    # ========================================
    st.subheader("Vị trí (Tùy chọn)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_name = st.selectbox(
            "Tỉnh/Thành phố", 
            options=opt_region,
            help="Chọn tỉnh/thành phố"
        )
    
    with col2:
        opt_area = get_area_options(df, region_name, option_none)
        area_name = st.selectbox(
            "Quận/Huyện", 
            options=opt_area,
            help="Chọn quận/huyện (tự động lọc theo tỉnh)"
        )
    
    with col3:
        opt_ward = get_ward_options(df, region_name, area_name, option_none)
        ward_name = st.selectbox(
            "Phường/Xã", 
            options=opt_ward,
            help="Chọn phường/xã (tự động lọc theo quận)"
        )
    
    # ========================================
    # CHI TIẾT
    # ========================================
    st.subheader("Chi tiết (Tùy chọn)")
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input(
            "Chiều rộng (m)", 
            min_value=0.0, 
            max_value=1000.0,
            value=5.0, 
            step=0.5,
            help="Chiều rộng mặt tiền"
        )
        length = st.number_input(
            "Chiều dài (m)", 
            min_value=0.0, 
            max_value=1000.0,
            value=10.0, 
            step=0.5,
            help="Chiều dài đất"
        )
        rooms = st.selectbox(
            "Số phòng", 
            options=opt_rooms,
            help="Số phòng ngủ"
        )
        
    with col2:
        direction_text = st.selectbox(
            "Hướng nhà", 
            options=opt_direction,
            help="Hướng nhà/đất"
        )
        legal_doc_text = st.selectbox(
            "Giấy tờ pháp lý", 
            options=opt_legal,
            help="Tình trạng pháp lý"
        )
        is_main_street = st.checkbox(
            "Mặt tiền", 
            value=True,
            help="BĐS có nằm trên đường/phố chính không"
        )
    
    # Nút submit
    st.markdown("---")
    submit_button = st.form_submit_button(
        "🔮 DỰ ĐOÁN GIÁ",
        use_container_width=True,
        type="primary"
    )

# ============================================================================
# XỬ LÝ KHI SUBMIT
# ============================================================================

if submit_button:
    
    st.markdown("---")
    st.header("📊 KẾT QUẢ DỰ ĐOÁN")
    
    # ========================================
    # 1. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
    # ========================================
    
    # Xử lý giá trị "Không có thông tin"
    def to_str_safe(val):
        return None if val == "Không có thông tin" else val
    
    # Tạo DataFrame input và điền giá trị mặc định cho các trường optional
    # Numeric defaults: median of full dataset
    width_val = width if width > 0 else numeric_medians.get('width', 0.0)
    length_val = length if length > 0 else numeric_medians.get('length', 0.0)
    if rooms != option_none and str(rooms).replace('.','').isdigit():
        rooms_val = float(rooms)
    else:
        rooms_val = numeric_medians.get('rooms', 0.0)

    # Categorical defaults: mode of full dataset
    def cat_default(col_name, user_val):
        if user_val is None or user_val == option_none:
            return categorical_modes.get(col_name, 'unknown')
        return user_val

    # parse displayed selections back to codes (strings) expected by preprocessor
    parsed_direction = parse_direction_selection(direction_text)
    if parsed_direction is None:
        direction_val = categorical_modes.get('direction', 'unknown')
    else:
        direction_val = parsed_direction

    parsed_legal = parse_legal_selection(legal_doc_text)
    if parsed_legal is None:
        legal_doc_val = categorical_modes.get('property_legal_document', 'unknown')
    else:
        legal_doc_val = parsed_legal
    region_val = cat_default('region_name', region_name)
    area_val = cat_default('area_name', area_name)
    ward_val = cat_default('ward_name', ward_name)

    # Binary default: use checkbox value directly (checked=1, unchecked=0)
    # Note: checkbox cannot express "no input", so we treat unchecked as explicit 0
    is_main_street_val = 0.0 if is_main_street else 1.0

    input_data = {
        'category_name': category_name,
        'size': size,
        'width': width_val,
        'length': length_val,
        'rooms': rooms_val,
        'direction': direction_val,
        'property_legal_document': legal_doc_val,
        'region_name': region_val,
        'area_name': area_val,
        'ward_name': ward_val,
        'is_main_street': is_main_street_val
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Hiển thị thông tin đã nhập
    with st.expander("📋 Xem thông tin đã nhập", expanded=False):
        st.write(df_input)
    
    # ========================================
    # 2. DỰ ĐOÁN GIÁ
    # ========================================
    
    try:
        with st.spinner("🔄 Đang dự đoán giá..."):
            # Transform input data using preprocessor đã fit
            df_input_transformed, _, _ = transform_for_prediction(
                df_input,
                preprocessor=preprocessor,
                artifacts=artifacts
            )
            
            # Predict (log scale)
            y_pred_log = model.predict(df_input_transformed)[0]
            
            # Inverse transform về VNĐ
            y_pred_vnd = np.expm1(y_pred_log)
            
            # Kiểm tra giá trị hợp lệ
            if np.isinf(y_pred_vnd) or np.isnan(y_pred_vnd):
                st.error("❌ Lỗi: Giá dự đoán không hợp lệ")
                st.stop()
            
            # Tính giá/m²
            price_per_m2 = y_pred_vnd / size
            
        # Hiển thị kết quả
        st.success("✅ Dự đoán thành công!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="💰 Giá dự đoán", 
                value=format_price(y_pred_vnd),
                help="Giá dự đoán của BĐS"
            )
        with col2:
            st.metric(
                label="📐 Giá/m²", 
                value=f"{price_per_m2/1_000_000:.2f} triệu/m²",
                help="Đơn giá trên mỗi m²"
            )
        with col3:
            st.metric(
                label="📏 Diện tích", 
                value=f"{size:.1f} m²",
                help="Tổng diện tích"
            )
        
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
        st.write("Debug info:")
        st.write("Input data:", df_input.to_dict())
        if 'df_input_transformed' in locals():
            st.write("Transformed data:", df_input_transformed.to_dict())
        st.stop()
    
    # ========================================
    # 3. GỢI Ý BẤT ĐỘNG SẢN TƯƠNG TỰ
    # ========================================
    
    st.markdown("---")
    st.subheader("🏘️ Gợi ý Bất động sản tương tự")
    
    with st.spinner("🔍 Đang tìm BĐS tương tự..."):
        # Filter BĐS tương tự
        df_suggest = df[
            (df['category_name'] == category_name) &
            (df['size'].between(size * 0.7, size * 1.3))  # Size trong khoảng ±30%
        ].copy()
        
        if region_name != option_none:
            df_suggest = df_suggest[df_suggest['region_name'] == region_name]
            
            if area_name != option_none:
                df_suggest = df_suggest[df_suggest['area_name'] == area_name]
        
        # Tính giá/m²
        df_suggest['price_per_m2'] = df_suggest['price'] / df_suggest['size']
        
        # Sắp xếp theo giá gần với giá dự đoán nhất
        df_suggest['price_diff'] = abs(df_suggest['price'] - y_pred_vnd)
        df_suggest = df_suggest.sort_values('price_diff').head()
    
    # Hiển thị kết quả
    if df_suggest.empty:
        st.info("Không tìm thấy BĐS tương tự 😢")
        
    else:
        # Format columns
        df_display = df_suggest.copy()
        
        # Chuyển đổi các cột sang numeric nếu cần
        df_display['price'] = pd.to_numeric(df_display['price'], errors='coerce')
        df_display['size'] = pd.to_numeric(df_display['size'], errors='coerce')
        
        # Format display values
        df_display.loc[:, 'price'] = df_display['price'].apply(format_price)
        df_display.loc[:, 'price_per_m2'] = df_display['price_per_m2'].apply(
            lambda x: f"{x/1_000_000:.2f} triệu/m²"
        )
        df_display.loc[:, 'size'] = df_display['size'].apply(
            lambda x: f"{x:.1f} m²" if pd.notnull(x) else "N/A"
        )
        
        # Format additional columns
        df_display.loc[:, 'rooms'] = df_display['rooms'].fillna('N/A').astype(str)
        df_display.loc[:, 'direction'] = df_display['direction'].map(direction_map).fillna('Không có thông tin')
        df_display.loc[:, 'property_legal_document'] = df_display['property_legal_document'].map(legal_doc_map).fillna('Không có thông tin')
        df_display.loc[:, 'is_main_street'] = df_display['is_main_street'].map({1: 'Có', 0: 'Không'}).fillna('Không')
        
        # Create detail link with HTML
        df_display['Chi tiết'] = df_display.apply(
            lambda row: f'https://www.nhatot.com/mua-ban-bat-dong-san/{row.name}.htm' 
            #"https://gateway.chotot.com/v1/public/ad-listing/{rơw.name}"
            if pd.notnull(row.name) else "N/A", 
            axis=1
        )
        
        # Select columns to display
        columns_to_show = [
            'category_name', 'size', 'price', 'price_per_m2',
            'rooms', 'direction', 'property_legal_document', 'is_main_street',
            'region_name', 'area_name', 'ward_name', 'Chi tiết'
        ]
        
        # Rename original columns to Vietnamese labels before display
        df_display = df_display.rename(columns={
            'category_name': 'Loại hình',
            'size': 'Diện tích',
            'price': 'Giá tiền',
            'price_per_m2': 'Giá/m2',
            'rooms': 'Số phòng',
            'direction': 'Hướng nhà',
            'property_legal_document': 'Giấy tờ pháp lý',
            'is_main_street': 'Mặt tiền',
            'region_name': 'Tỉnh/Thành phố',
            'area_name': 'Quận/Huyện',
            'ward_name': 'Phường/Xã'
        })

        # Final columns to show (Vietnamese labels)
        columns_to_show = [
            'Loại hình', 'Diện tích', 'Giá tiền', 'Giá/m2',
            'Số phòng', 'Hướng nhà', 'Giấy tờ pháp lý', 'Mặt tiền',
            'Tỉnh/Thành phố', 'Quận/Huyện', 'Phường/Xã', 'Chi tiết'
        ]

        # Use pandas Styler to set column display properties (widths, nowrap)
        styler = df_display[columns_to_show].style
        # Prevent wrapping and set min-width for important columns
        try:
            styler = styler.set_properties(**{'white-space': 'nowrap'})
            styler = styler.set_properties(**{'min-width': '140px'}, subset=['Diện tích'])
            styler = styler.set_properties(**{'min-width': '180px'}, subset=['Giá tiền'])
        except Exception:
            # set_properties may fail on older pandas versions; ignore and continue
            pass

        # Render using st.dataframe which accepts a Styler object in recent Streamlit versions
        try:
            st.dataframe(styler, use_container_width=True)
        except Exception:
            # Fallback: show plain dataframe if Styler not supported by Streamlit
            st.dataframe(df_display[columns_to_show], use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p style='margin: 5px;'>🏠 <b>Ứng dụng dự đoán giá Bất động sản</b></p>
    <p style='margin: 5px;'>Sử dụng mô hình CatBoost với dữ liệu từ Chợ Tốt</p>
    <p style='margin: 5px;'>© 2025 - Real Estate Price Prediction System</p>
</div>
""", unsafe_allow_html=True)