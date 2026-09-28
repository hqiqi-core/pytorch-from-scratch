import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split

class RegressionDataset(Dataset):

    def __init__(self, num_samples):
        self.samples_x = torch.randn(num_samples)  # torch.randn 返回的类型就是tensor
        self.samples_y = self.samples_x * 3 +2

    def __len__(self):
        return len(self.samples_x)

    def __getitem__(self, index):
        x = self.samples_x[index]   # 通过索引取单个元素 → 0 维 tensor   如果想要看到 [1] 这样的形状 可以用切片或 unsqueeze
        y = self.samples_y[index]
        return x,y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        x = self.network(x)
        return x

model = MLP()
dataset = RegressionDataset(100)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# ===== 按 7:1:2 划分 =====
total = len(dataset)             # 100
train_size = int(0.7 * total)    # 70
val_size   = int(0.1 * total)    # 10
test_size  = total - train_size - val_size  # 20（用减法保证总数对齐）

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size]
)

# ===== 为每个子集创建 DataLoader =====
batch_size = 10
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)

for epoch in range(20):
    model.train()
    total_train_loss = 0
    for x_batch,y_batch in train_loader:
        x_batch = x_batch.unsqueeze(1)
        y_batch = y_batch.unsqueeze(1)
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()  # 不等价于with torch.no_grad() 只是现在进入评估模式。主要影响一些具有训练/评估不同状态的层
    total_val_loss = 0
    with torch.no_grad():# 有了eval还是要加这个  是告诉pytorch下面这些计算不需要建立梯度计算图。
        for x_val_batch,y_val_batch in val_loader:
            x_val_batch = x_val_batch.unsqueeze(1)
            y_val_batch = y_val_batch.unsqueeze(1)
            prediction = model(x_val_batch)
            val_loss = criterion(prediction,y_val_batch)
            total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
    print(f"Train loss:{avg_train_loss},Val loss:{avg_val_loss}")

model.eval()
total_test_loss = 0
with torch.no_grad():# 有了eval还是要加这个
    for x_test_batch,y_test_batch in test_loader:
        x_test_batch= x_test_batch.unsqueeze(1)
        y_test_batch = y_test_batch.unsqueeze(1)
        prediction= model(x_test_batch)
        test_loss = criterion(prediction,y_test_batch)
        total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss/len(test_loader)
    print(f"Test loss:{avg_test_loss}")