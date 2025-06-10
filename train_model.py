import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# ---------------------------
# 1. 路徑設定
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")  # 已處理資料目錄
OUTPUT_DIR = BASE_DIR

# ---------------------------
# 2. 讀取處理後資料
# ---------------------------
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_basic.parquet"))
test_df = pd.read_parquet(os.path.join(DATA_DIR, "test_basic.parquet"))

# ---------------------------
# 3. 定義特徵與目標，填補缺失並標準化
# ---------------------------
FEATURES = [
    "onpromotion",
    "dcoilwtico",
    "transactions",
    "is_holiday",
    "dow",
    "month",
    "year",
    "weekofyear",
    "is_weekend",
    "family_enc",
    "type_enc",
    "cluster_enc",
]
TARGET = "sales_log"

# 補缺失值
train_df[FEATURES] = train_df[FEATURES].fillna(0)
test_df[FEATURES] = test_df[FEATURES].fillna(0)

# 標準化特徵
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])


# ---------------------------
# 4. 定義序列資料集
# ---------------------------
class SeqSalesDataset(Dataset):
    def __init__(self, df, features, target=None, seq_len=30):
        # 將資料依 store_nbr, family, date 排序後，轉為數組
        data = df.sort_values(["store_nbr", "family", "date"])
        arr = data[features + ([target] if target else [])].values
        group_arr = data["store_nbr"].values

        self.X = []
        self.y = []
        self.groups = []

        for i in range(len(arr) - seq_len):
            seq_x = arr[i : i + seq_len, : len(features)]
            self.X.append(seq_x.astype(np.float32))
            if target:
                seq_y = arr[i + seq_len, len(features)]
                self.y.append(np.float32(seq_y))
            self.groups.append(int(group_arr[i + seq_len]))

        self.X = np.stack(self.X)
        if target:
            self.y = np.array(self.y, dtype=np.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.y is not None:
            y = torch.tensor(self.y[idx])
            return x, y
        return x


# ---------------------------
# 5. 定義 LSTM 模型
# ---------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        output = self.fc(last)
        return output.squeeze(1)


# ---------------------------
# 6. 設備與參數
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seq_len = 30
epochs = 5
batch_size = 512
learning_rate = 1e-3

# 準備資料集和交叉驗證
dataset = SeqSalesDataset(train_df, FEATURES, TARGET, seq_len)
kfold = GroupKFold(n_splits=5)
models = []

for fold, (train_idx, val_idx) in enumerate(
    kfold.split(dataset.X, y=dataset.y, groups=dataset.groups), start=1
):
    print(f"=== Fold {fold} Training ===")

    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    model = LSTMForecaster(input_dim=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = model.state_dict()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                val_preds.append(out)
                val_targets.append(yb.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        print(f"Fold {fold} Epoch {epoch} RMSE(log): {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = model.state_dict()

    print(f"Fold {fold} best RMSE(log): {best_rmse:.4f}\n")
    model.load_state_dict(best_state)
    models.append(model)

# ---------------------------
# 7. 在測試集上預測
# ---------------------------
combined = pd.concat([train_df.tail(seq_len), test_df], ignore_index=True)
test_dataset = SeqSalesDataset(combined, FEATURES, target=None, seq_len=seq_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_fold_preds = []
for model in models:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
    all_fold_preds.append(np.concatenate(preds))

mean_log_preds = np.mean(all_fold_preds, axis=0)
pred_sales = np.expm1(mean_log_preds)
pred_sales = np.nan_to_num(pred_sales, nan=0.0, posinf=0.0, neginf=0.0)

submission = pd.DataFrame({"id": test_df["id"], "sales": pred_sales})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print("Submission saved to", os.path.join(OUTPUT_DIR, "submission.csv"))
