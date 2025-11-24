"""
data_pipeline.py
================
Module tự động hóa quy trình:
1. Crawl dữ liệu BĐS từ Chợ Tốt
2. Upload lên MongoDB
3. Load dữ liệu từ MongoDB về DataFrame

Author: Your Name
Date: 2024
"""

import requests
import pandas as pd
import time
from tqdm import tqdm
from pymongo import MongoClient
import certifi

# ============================================================================
# PHẦN 1: CRAWL DỮ LIỆU TỪ CHỢ TỐT
# ============================================================================

def crawl_chotot_data(
    start_page=0,
    max_pages=None,
    save_every=500,
    sleep_time=1,
    region_id=None,
    save_csv=True,
    csv_filename=None
):
    """
    Crawl dữ liệu BĐS từ Chợ Tốt - đầy đủ thông tin, tự động mapping hướng & pháp lý.

    Parameters
    ----------
    start_page : int, optional
        Trang bắt đầu crawl (mặc định là 0).
    max_pages : int hoặc None, optional
        Số lượng trang tối đa muốn crawl (mặc định là None - không giới hạn).
    save_every : int, optional
        Tự động LƯU file CSV tạm sau mỗi N tin (mặc định là 500).
    sleep_time : float, optional
        Thời gian (giây) nghỉ giữa mỗi lần gọi API.
    region_id : str hoặc None, optional
        Mã vùng (region) muốn lọc.
        - "0" hoặc None: Toàn quốc (mặc định).
        - "13000": Chỉ TP. Hồ Chí Minh.
        - "12000": Chỉ Hà Nội.
    save_csv : bool, optional
        Có lưu file CSV cuối cùng không (mặc định True).
    csv_filename : str, optional
        Tên file CSV tùy chỉnh (nếu không có sẽ tự động đặt tên).

    Returns
    -------
    pandas.DataFrame
        DataFrame chứa toàn bộ dữ liệu đã thu thập và làm sạch.
    """

    # =========================
    # CẤU HÌNH
    # =========================
    BASE_URL = "https://gateway.chotot.com/v1/public/ad-listing"

    KEYS_TO_EXTRACT = [
        "list_id", "status", "price", "price_string", "price_million_per_m2",
        "size", "width", "length", "rooms", "direction", "property_legal_document",
        "region_name", "area_name", "ward_name", "category_name", "is_main_street",
        "number_of_images"
    ]

    all_data = []
    all_ids = set()
    file_counter = 1
    page = start_page
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # =========================
    # HÀM PHỤ TRỢ
    # =========================
    def safe_get(ad, key, default=''):
        value = ad.get(key, default)
        return value if value is not None else default

    def map_values(df):
        direction_map = {
            1: "Đông", 2: "Tây", 3: "Nam", 4: "Bắc",
            5: "Đông-Bắc", 6: "Tây-Bắc", 7: "Đông-Nam", 8: "Tây-Nam"
        }
        legal_doc_map = {
            1: "Sổ hồng / Sổ đỏ đầy đủ", 2: "Giấy tay / Chưa có sổ",
            3: "Đang chờ sổ", 4: "Hợp đồng mua bán", 5: "Khác"
        }
        df["direction_text"] = df["direction"].map(direction_map)
        df["legal_doc_text"] = df["property_legal_document"].map(legal_doc_map)
        return df

    def save_snapshot(data_list, file_num, current_page):
        if not data_list:
            return
        df_temp = pd.DataFrame(data_list)
        filename = f"chotot_part{file_num}_{len(data_list)}tin_p{current_page}.csv"
        df_temp.to_csv(filename, index=False, encoding='utf-8-sig')
        tqdm.write(f"\n💾 [Snapshot {file_num}] Đã lưu: {filename} ({len(data_list)} tin)")

    # =========================
    # VÒNG LẶP CRAWL
    # =========================
    region_text = "TOÀN QUỐC 🇻🇳" if region_id in [None, "0"] else f"Region {region_id}"
    print(f"🚀 Bắt đầu crawl từ trang {start_page} | Vùng: {region_text}")
    print(f"💾 Tự động lưu snapshot mỗi {save_every} tin")
    print(f"🛑 Dừng nếu gặp {MAX_CONSECUTIVE_ERRORS} lỗi liên tiếp.\n")

    pbar = tqdm(total=max_pages, desc="Crawling", unit="page")

    while True:
        if max_pages is not None and (page - start_page) >= max_pages:
            tqdm.write(f"\n⏹️ Đã đạt giới hạn {max_pages} trang. Dừng crawl.")
            break

        offset = page * 25
        params = {"cg": 1000, "o": offset, "st": "s,k", "limit": 25}
        if region_id and region_id != "0":
            params["region_v2"] = region_id

        try:
            r = session.get(BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            consecutive_errors = 0

        except requests.exceptions.HTTPError as e:
            tqdm.write(f"\n⚠️ Lỗi HTTP trang {page}: {e}")
            if r.status_code == 404:
                tqdm.write("Lỗi 404, đã hết trang. Dừng...")
                break

            consecutive_errors += 1
            tqdm.write(f"Lỗi liên tiếp: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                tqdm.write("🛑 Đạt tối đa lỗi liên tiếp. Dừng crawl.")
                break

            time.sleep(5)
            continue

        except Exception as e:
            tqdm.write(f"\n⚠️ Lỗi kết nối trang {page}: {e}")
            consecutive_errors += 1
            tqdm.write(f"Lỗi liên tiếp: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                tqdm.write("🛑 Đạt tối đa lỗi liên tiếp. Dừng crawl.")
                break

            time.sleep(5)
            continue

        ads = data.get('ads', [])

        if not ads:
            tqdm.write(f"\n✅ API trả về rỗng tại trang {page}. Dừng crawl.")
            break

        new_records_on_page = 0
        for ad in ads:
            list_id = safe_get(ad, 'list_id')
            if not list_id or list_id in all_ids:
                continue

            all_ids.add(list_id)
            new_records_on_page += 1
            record = {key: safe_get(ad, key) for key in KEYS_TO_EXTRACT}
            all_data.append(record)

        if new_records_on_page == 0:
            tqdm.write(f"\n🛑 Trang {page} không có tin mới. Dừng crawl.")
            break

        page += 1
        pbar.update(1)
        pbar.set_postfix({"Tổng tin": len(all_data)})

        if len(all_data) >= file_counter * save_every:
            save_snapshot(all_data, file_counter, page)
            file_counter += 1

        time.sleep(sleep_time)

    pbar.close()

    # =========================
    # HOÀN THÀNH
    # =========================
    if not all_data:
        print("\n❌ Crawl hoàn tất nhưng không có dữ liệu.")
        return pd.DataFrame()

    print(f"\n🎉 Crawl hoàn tất! Tổng cộng {len(all_data)} tin.")

    df_final = pd.DataFrame(all_data)
    df_final = map_values(df_final)

    column_order = [
        "list_id", "status", "price", "price_string", "price_million_per_m2",
        "size", "width", "length", "rooms", "direction", "direction_text",
        "property_legal_document", "legal_doc_text", "region_name", "area_name",
        "ward_name", "category_name", "is_main_street", "number_of_images",
    ]
    df_final = df_final.reindex(columns=column_order)

    # Lưu file CSV
    if save_csv:
        if csv_filename is None:
            region_suffix = "TOANQUOC" if region_id in [None, "0"] else f"region{region_id}"
            csv_filename = f"chotot_FINAL_{region_suffix}_{len(df_final)}tin_p{start_page}-p{page-1}.csv"

        print(f"💾 Đang lưu file: {csv_filename}...")
        df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu file thành công!")

    return df_final


# ============================================================================
# PHẦN 2: UPLOAD LÊN MONGODB
# ============================================================================

def upload_to_mongodb(
    df,
    connection_string,
    db_name,
    collection_name,
    drop=0
):
    """
    Đẩy DataFrame lên MongoDB collection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần upload.
    connection_string : str
        Chuỗi kết nối MongoDB.
    db_name : str
        Tên database.
    collection_name : str
        Tên collection.
    drop : int, optional
        - drop = 0: Giữ dữ liệu cũ, chèn thêm (mặc định).
        - drop = 1: Xóa tất cả dữ liệu cũ trước khi chèn.

    Returns
    -------
    bool
        True nếu thành công, False nếu có lỗi.
    """

    if df.empty:
        print("⚠️ DataFrame rỗng, không có gì để upload.")
        return False

    client = None

    try:
        # Kết nối MongoDB
        ca = certifi.where()
        client = MongoClient(connection_string, tls=True, tlsCAFile=ca)

        db = client[db_name]
        collection = db[collection_name]
        print(f"✅ Đã kết nối tới DB: '{db_name}', Collection: '{collection_name}'")

        # Xóa dữ liệu cũ nếu drop=1
        if drop == 1:
            print("🗑️  Đang xóa dữ liệu cũ (drop=1)...")
            result_delete = collection.delete_many({})
            print(f"✅ Đã xóa {result_delete.deleted_count} document cũ.")
        else:
            print("➕ Chèn nối tiếp (drop=0)...")

        # Chuyển DataFrame sang list dict
        data_to_insert = df.to_dict("records")

        # Upload
        result_insert = collection.insert_many(data_to_insert)
        print(f"✅ Đã upload thành công {len(result_insert.inserted_ids)} document!\n")

        return True

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        if "SSL" in str(e) or "timeout" in str(e):
            print("⚠️ GỢI Ý: Kiểm tra IP Access List trên MongoDB Atlas!")
        return False

    finally:
        if client:
            client.close()


# ============================================================================
# PHẦN 3: LOAD DỮ LIỆU TỪ MONGODB
# ============================================================================

def load_from_mongodb(
    connection_string,
    db_name,
    collection_name,
    query_filter={},
    remove_id=True
):
    """
    Kết nối MongoDB và tải dữ liệu về DataFrame.

    Parameters
    ----------
    connection_string : str
        Chuỗi kết nối MongoDB.
    db_name : str
        Tên database.
    collection_name : str
        Tên collection.
    query_filter : dict, optional
        Bộ lọc MongoDB (mặc định {} = lấy tất cả).
    remove_id : bool, optional
        Nếu True, tự động loại bỏ cột "_id" (mặc định True).

    Returns
    -------
    pandas.DataFrame
        DataFrame chứa dữ liệu. Trả về DataFrame rỗng nếu có lỗi.
    """

    df_from_mongo = pd.DataFrame()
    client = None

    try:
        # Kết nối MongoDB
        ca = certifi.where()
        client = MongoClient(connection_string, tls=True, tlsCAFile=ca)

        db = client[db_name]
        collection = db[collection_name]

        print(f"📥 Đang tải dữ liệu từ '{db_name}.{collection_name}'...")

        # Projection để loại bỏ _id
        projection = {"_id": 0} if remove_id else None

        # Truy vấn
        cursor = collection.find(query_filter, projection)
        data_list = list(cursor)

        if data_list:
            df_from_mongo = pd.DataFrame(data_list)
            print(f"✅ Tải thành công {len(df_from_mongo)} document.\n")
        else:
            print("⚠️ Collection rỗng hoặc không có dữ liệu.\n")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        if "SSL" in str(e) or "timeout" in str(e):
            print("⚠️ GỢI Ý: Kiểm tra IP Access List trên MongoDB Atlas!")

    finally:
        if client:
            client.close()

    return df_from_mongo


# ============================================================================
# PHẦN 4: PIPELINE TỰ ĐỘNG (ALL-IN-ONE)
# ============================================================================

def run_full_pipeline(
    # Crawl params
    max_pages=100,
    region_id=None,
    sleep_time=1,
    # MongoDB params
    connection_string=None,
    db_name=None,
    collection_name=None,
    drop_before_upload=0,
    # Options
    skip_crawl=False,
    skip_upload=False,
    skip_load=False
):
    """
    Chạy toàn bộ pipeline: Crawl → Upload → Load

    Parameters
    ----------
    max_pages : int
        Số trang tối đa để crawl.
    region_id : str, optional
        Mã vùng (None = toàn quốc).
    sleep_time : float
        Thời gian chờ giữa các request.
    connection_string : str
        Connection string MongoDB.
    db_name : str
        Tên database MongoDB.
    collection_name : str
        Tên collection MongoDB.
    drop_before_upload : int
        0 = append, 1 = drop trước khi upload.
    skip_crawl : bool
        Nếu True, bỏ qua bước crawl (dùng data có sẵn).
    skip_upload : bool
        Nếu True, bỏ qua bước upload.
    skip_load : bool
        Nếu True, bỏ qua bước load.

    Returns
    -------
    pandas.DataFrame
        DataFrame cuối cùng (từ MongoDB hoặc từ crawl).
    """

    print("="*80)
    print("🚀 BẮT ĐẦU FULL PIPELINE")
    print("="*80 + "\n")

    df_final = None

    # ========================================
    # BƯỚC 1: CRAWL DỮ LIỆU
    # ========================================
    if not skip_crawl:
        print("\n" + "="*80)
        print("📡 BƯỚC 1: CRAWL DỮ LIỆU TỪ CHỢ TỐT")
        print("="*80 + "\n")

        df_crawled = crawl_chotot_data(
            max_pages=max_pages,
            region_id=region_id,
            sleep_time=sleep_time,
            save_csv=True
        )

        if df_crawled.empty:
            print("❌ Crawl không có dữ liệu. Dừng pipeline.")
            return pd.DataFrame()

        df_final = df_crawled

    else:
        print("\n⏭️  Bỏ qua bước crawl (skip_crawl=True)")

    # ========================================
    # BƯỚC 2: UPLOAD LÊN MONGODB
    # ========================================
    if not skip_upload and df_final is not None:
        if connection_string is None or db_name is None or collection_name is None:
            print("\n⚠️ Thiếu thông tin MongoDB. Bỏ qua upload.")
        else:
            print("\n" + "="*80)
            print("☁️  BƯỚC 2: UPLOAD LÊN MONGODB")
            print("="*80 + "\n")

            success = upload_to_mongodb(
                df_final,
                connection_string,
                db_name,
                collection_name,
                drop=drop_before_upload
            )

            if not success:
                print("⚠️ Upload thất bại nhưng tiếp tục pipeline.")
    else:
        print("\n⏭️  Bỏ qua bước upload")

    # ========================================
    # BƯỚC 3: LOAD TỪ MONGODB
    # ========================================
    if not skip_load:
        if connection_string is None or db_name is None or collection_name is None:
            print("\n⚠️ Thiếu thông tin MongoDB. Bỏ qua load.")
        else:
            print("\n" + "="*80)
            print("📥 BƯỚC 3: LOAD DỮ LIỆU TỪ MONGODB")
            print("="*80 + "\n")

            df_loaded = load_from_mongodb(
                connection_string,
                db_name,
                collection_name
            )

            if not df_loaded.empty:
                df_final = df_loaded
            else:
                print("⚠️ Load không có dữ liệu.")
    else:
        print("\n⏭️  Bỏ qua bước load")

    # ========================================
    # HOÀN THÀNH
    # ========================================
    print("\n" + "="*80)
    print("🎉 HOÀN TẤT FULL PIPELINE")
    print("="*80)

    if df_final is not None and not df_final.empty:
        print(f"\n📊 Kết quả cuối cùng: {df_final.shape[0]} dòng, {df_final.shape[1]} cột")
        print("\n📌 5 dòng đầu:")
        print(df_final.head())
    else:
        print("\n⚠️ Không có dữ liệu cuối cùng.")

    return df_final


# # ========================================
    # # CÁCH 1: CHẠY FULL PIPELINE
    # # ========================================
    # print("\n🔥 CHẠY FULL PIPELINE (Crawl → Upload → Load)\n")

    # df_result = run_full_pipeline(
    #     max_pages=10,  # Crawl 10 trang để test
    #     region_id=None,  # None = toàn quốc
    #     sleep_time=1,
    #     connection_string=MONGO_CONNECTION,
    #     db_name=DB_NAME,
    #     collection_name=COLLECTION_NAME,
    #     drop_before_upload=0  # 0 = append, 1 = drop trước khi upload
    # )

    # ========================================
    # CÁCH 2: CHẠY TỪNG BƯỚC RIÊNG LẺ
    # ========================================

    # # BƯỚC 1: Chỉ crawl
    # df_crawled = crawl_chotot_data(max_pages=10, region_id="13000")

    # # BƯỚC 2: Chỉ upload
    # upload_to_mongodb(df_crawled, MONGO_CONNECTION, DB_NAME, COLLECTION_NAME, drop=1)

    # # BƯỚC 3: Chỉ load
    # df_loaded = load_from_mongodb(MONGO_CONNECTION, DB_NAME, COLLECTION_NAME)

    # ========================================
    # CÁCH 3: CHỈ LOAD DỮ LIỆU CÓ SẴN
    # ========================================

    # df_from_db = run_full_pipeline(
    #     connection_string=MONGO_CONNECTION,
    #     db_name=DB_NAME,
    #     collection_name=COLLECTION_NAME,
    #     skip_crawl=True,  # Bỏ qua crawl
    #     skip_upload=True  # Bỏ qua upload
    # )