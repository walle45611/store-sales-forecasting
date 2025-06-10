import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "init")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
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

# 4. 節日處理：去除 Transfer
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

# 8. 計算 sales_log 與 lag 特徵
train["sales_log"] = np.log1p(train["sales"])
print("Adding lagged sales features...")
full_df = pd.concat([train, test], ignore_index=True)
full_df = full_df.sort_values(by=["store_nbr", "family", "date"])

for lag in [1, 7, 14, 28]:
    full_df[f"sales_lag_{lag}"] = full_df.groupby(["store_nbr", "family"])[
        "sales_log"
    ].shift(lag)

# 9. 類別編碼
le_family = LabelEncoder().fit(full_df["family"])
le_type = LabelEncoder().fit(full_df["type"])
le_cluster = LabelEncoder().fit(full_df["cluster"])

full_df["family_enc"] = le_family.transform(full_df["family"])
full_df["type_enc"] = le_type.transform(full_df["type"])
full_df["cluster_enc"] = le_cluster.transform(full_df["cluster"])

# 10. 分離 train/test 並重置欄位順序
train_output = full_df[full_df["id"].isin(train["id"])].copy()
test_output = full_df[full_df["id"].isin(test["id"])].copy()

COMMON_FEATURES = [
    "id",
    "date",
    "store_nbr",
    "family",
    "onpromotion",
    "dcoilwtico",
    "transactions",
    "is_holiday",
    "type",
    "cluster",
    "city",
    "state",
    "dow",
    "month",
    "year",
    "weekofyear",
    "is_weekend",
    "family_enc",
    "type_enc",
    "cluster_enc",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
]

train_output = train_output[COMMON_FEATURES + ["sales", "sales_log"]]
test_output = test_output[COMMON_FEATURES]

# 11. 輸出 parquet
train_output.to_parquet(os.path.join(OUTPUT_DIR, "train_basic.parquet"), index=False)
test_output.to_parquet(os.path.join(OUTPUT_DIR, "test_basic.parquet"), index=False)

print("Preprocessing complete! Files saved in:", OUTPUT_DIR)
print(f"Train shape: {train_output.shape}")
print(f"Test shape:  {test_output.shape}\n")

MODEL_FEATURES_FOR_TRAIN = COMMON_FEATURES[4:]  # 去掉id,date,store_nbr,family
print(f"===== 模型特徵 ({len(MODEL_FEATURES_FOR_TRAIN)}個) =====")
for feature in MODEL_FEATURES_FOR_TRAIN:
    print(f"- {feature}")

# 12. 顯示前五筆資料
print("\n===== Train 前五筆 =====")
print(train_output.head())

print("\n===== Test 前五筆 =====")
print(test_output.head())
