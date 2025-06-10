import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "init")  # 原始 CSV 資料夾
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")  # 處理後結果存放
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 讀取原始 CSV
train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"), parse_dates=["date"])
test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"), parse_dates=["date"])
oil = pd.read_csv(os.path.join(RAW_DIR, "oil.csv"), parse_dates=["date"])
holidays = pd.read_csv(
    os.path.join(RAW_DIR, "holidays_events.csv"), parse_dates=["date"]
)
txn = pd.read_csv(os.path.join(RAW_DIR, "transactions.csv"), parse_dates=["date"])
stores = pd.read_csv(os.path.join(RAW_DIR, "stores.csv"))

# 3. 補齊油價缺失
oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

# 4. 節日處理：去除 Transfer，並標記 is_holiday
holidays = holidays[holidays["type"] != "Transfer"].copy()
holidays["is_holiday"] = 1
holidays = holidays[["date", "is_holiday"]]

# 5. 聚合每日交易量
txn_agg = txn.groupby(["date", "store_nbr"])["transactions"].sum().reset_index()


# 6. 定義合併函式
def merge_all(df):
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays, on="date", how="left")
    df = df.merge(txn_agg, on=["date", "store_nbr"], how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0)
    df["transactions"] = df["transactions"].fillna(0)
    return df


train = merge_all(train)
test = merge_all(test)


# 7. 加入時間特徵
def add_time_features(df):
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)


for df in (train, test):
    add_time_features(df)

# 8. 類別欄位編碼
le_family = LabelEncoder().fit(train["family"])
le_type = LabelEncoder().fit(stores["type"])
le_cluster = LabelEncoder().fit(stores["cluster"])

for df in (train, test):
    df["family_enc"] = le_family.transform(df["family"])
    df["type_enc"] = le_type.transform(df["type"])
    df["cluster_enc"] = le_cluster.transform(df["cluster"])

# 9. 轉換目標變數 (train)
train["sales_log"] = np.log1p(train["sales"])

# 10. 輸出結果 (Parquet 格式)
train.to_parquet(os.path.join(OUTPUT_DIR, "train_basic.parquet"), index=False)
test.to_parquet(os.path.join(OUTPUT_DIR, "test_basic.parquet"), index=False)

print("Preprocessing complete! Files saved in:", OUTPUT_DIR)

# 11. 印出最後幾筆做確認
print("\n===== Train 最後 5 筆 =====")
print(train.tail())

print("\n===== Test 最後 5 筆 =====")
print(test.tail())
