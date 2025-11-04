# comsol_surrogate_train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1️⃣ 定义三隐藏层神经网络（LeakyReLU + BatchNorm）
# ============================================================

class SurrogateNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2️⃣ 定义加载已训练模型函数
# ============================================================

def load_surrogate_model(checkpoint_path: str, input_dim: int, output_dim: int, device='cpu'):
    """
    加载已经训练好的 surrogate 模型和归一化器
    """
    model = SurrogateNN(input_dim, output_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 恢复归一化器参数
    x_scaler = StandardScaler()
    x_scaler.mean_ = checkpoint['x_scaler_mean']
    x_scaler.scale_ = checkpoint['x_scaler_scale']

    y_scalers = []
    for mean, scale in zip(checkpoint['y_scalers_mean'], checkpoint['y_scalers_scale']):
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        y_scalers.append(scaler)

    return model, x_scaler, y_scalers


# ============================================================
# 3️⃣ 可选：训练与验证代码（仅在独立运行时执行）
# ============================================================

if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import matplotlib.pyplot as plt

    # ========== 数据读取与预处理 ==========
    data = pd.read_csv("comsol_data.csv")

    input_cols = ['base_x', 'base_z', 'base_y', 'g_1', 'g_2', 'g_3',
                  'thick_copper', 'w_1', 'w_2', 'core_y', 'r', 'n', 'I']
    output_cols = ['LCoil_uH', 'Lmut_uH', 'CPL_ripple', 'CPL_volume', 'ht_QInt_mW', 'ht_hf1_Tave_C']

    X = data[input_cols].values
    y = data[output_cols].values

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    y_scalers = []
    y_scaled_list = []
    for i in range(y.shape[1]):
        scaler = StandardScaler()
        y_scaled_list.append(scaler.fit_transform(y[:, i].reshape(-1, 1)))
        y_scalers.append(scaler)
    y_scaled = np.hstack(y_scaled_list)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    n_total = len(X_tensor)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_data, val_data, test_data = random_split(TensorDataset(X_tensor, y_tensor), [n_train, n_val, n_test])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # ========== 模型初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateNN(len(input_cols), len(output_cols)).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, threshold=1e-4)

    n_epochs = 300
    train_losses, val_losses = [], []

    # ========== 训练循环 ==========
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")


    # ========== 绘制训练与验证损失曲线 ==========
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300)
    plt.show()




    # ========== 保存模型 ==========
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_scaler_mean': x_scaler.mean_,
        'x_scaler_scale': x_scaler.scale_,
        'y_scalers_mean': [sc.mean_ for sc in y_scalers],
        'y_scalers_scale': [sc.scale_ for sc in y_scalers],
    }, 'surrogate_model_optimized.pth')
    print("\n🎯 模型已保存为 surrogate_model_optimized.pth")
