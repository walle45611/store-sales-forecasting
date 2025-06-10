import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def load_data():
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_basic.parquet"))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_basic.parquet"))
    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
    return train_df, test_df

train_df, test_df = load_data()

# 2. 特徵與目標
FEATURES = [
    "onpromotion", "dcoilwtico", "transactions", "is_holiday",
    "dow", "month", "year", "weekofyear", "is_weekend",
    "family_enc", "type_enc", "cluster_enc",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
]
TARGET = "sales_log"

# 3. 填補缺失並標準化
train_df[FEATURES] = train_df[FEATURES].fillna(0)
test_df[FEATURES]  = test_df[FEATURES].fillna(0)
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES]  = scaler.transform(test_df[FEATURES])

# 4. 序列資料集
class SeqSalesDataset(Dataset):
    def __init__(self, df, features, target=None, seq_len=7):
        data = df.sort_values(["store_nbr", "family", "date"])
        X = data[features].values.astype(np.float32)
        grp = data["store_nbr"].values
        self.X, self.y, self.groups = [], [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i : i + seq_len])
            if target and target in data.columns:
                self.y.append(data[target].iloc[i + seq_len])
            self.groups.append(int(grp[i + seq_len]))
        self.X = np.stack(self.X)
        self.y = np.array(self.y, dtype=np.float32) if self.y else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        if self.y is not None:
            return x, torch.tensor(self.y[idx])
        return x

# 5. LSTM 模型
class LSTMForecaster(nn.Module):
    def __init__(self, in_dim, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

# 6. 訓練、驗證與推論
def train_and_predict(train_df, test_df, features, target,
                      seq_len=7, folds=5, epochs=20,
                      batch_size=512, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_ds = SeqSalesDataset(train_df, features, target, seq_len)
    kf = GroupKFold(n_splits=folds)
    models = []

    # 5-fold CV
    for fold, (tr_idx, va_idx) in enumerate(
        kf.split(train_ds.X, y=train_ds.y, groups=train_ds.groups), start=1
    ):
        print(f"\n=== Fold {fold} ===")
        tr_dl = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size, shuffle=True)
        va_dl = DataLoader(Subset(train_ds, va_idx), batch_size=batch_size)
        model = LSTMForecaster(len(features)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_rmse, best_state = float("inf"), None

        for ep in range(1, epochs + 1):
            epoch_start = time.time()

            # 訓練
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            # 驗證
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    p = model(xb)
                    preds.append(p.cpu().numpy()); trues.append(yb.cpu().numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            rmse = np.sqrt(((preds - trues) ** 2).mean())

            epoch_time = time.time() - epoch_start
            print(f"Epoch {ep:02d} — Val RMSE: {rmse:.4f} — time: {epoch_time:.1f}s")

            if rmse < best_rmse:
                best_rmse, best_state = rmse, model.state_dict()

        model.load_state_dict(best_state)
        models.append(model)
        print(f"Fold {fold} Best RMSE: {best_rmse:.4f}")

    # 推論
    print("\nStart inference on test set...")
    inf_start = time.time()

    test_ds = SeqSalesDataset(test_df, features, target=None, seq_len=seq_len)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    all_fold_preds = []

    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for xb in test_dl:
                xb = xb.to(device)
                p = model(xb).cpu().numpy()
                preds.append(p)
        all_fold_preds.append(np.concatenate(preds))

    y_pred_log = np.mean(all_fold_preds, axis=0)
    y_pred     = np.expm1(y_pred_log)

    inf_time = time.time() - inf_start
    print(f"Inference done — total time: {inf_time:.1f}s")

    submission = pd.DataFrame({
        "id": test_df["id"].values[seq_len:],  
        "sales": y_pred
    })
    return models, submission

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, submission = train_and_predict(
        train_df, test_df,
        FEATURES, TARGET,
        seq_len=7, folds=5, epochs=20,
        batch_size=512, lr=1e-3,
        device=device
    )
    submission.to_csv("submission.csv", index=False)
    print("Done! Submission saved to submission.csv")
