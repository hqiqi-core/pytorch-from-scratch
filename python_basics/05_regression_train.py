import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split

model = nn.Linear(1, 1)

print(model)
print(model.weight)
print(model.bias)

# 目前x的形状是[8] 但是model输入要求[8,1] 这就要用到unsqueeze

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

print("weight_before_train:", model.weight)
print("bias_before_train:", model.bias)
for epoch in range(20):
    model.train()  # 模型训练
    total_train_loss = 0
    for x_batch, y_batch in train_loader:   # 另一种写法  for batch_idx, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.unsqueeze(1)  # 从[8]变成[8,1]
        y_batch = y_batch.unsqueeze(1)
        prediction = model(x_batch)
        loss = criterion(
            prediction,
            y_batch
        )
        loss.backward()  #反向传播计算梯度
        optimizer.step()  #更新参数
        optimizer.zero_grad()  # 清空梯度
        total_train_loss += loss.item()   # 累积 loss 时不带计算图，省内存  .item() 会切断计算图 所以要在反向传播以后

    avg_train_loss = total_train_loss / len(train_loader)

    #验证集
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.unsqueeze(1)
            y_val_batch = y_val_batch.unsqueeze(1)
            prediction = model(x_val_batch)
            val_loss = criterion(
                prediction,
                y_val_batch
            )
            total_loss += val_loss.item() 
        avg_loss = total_loss / len(val_loader)

    print(f"epoch={epoch}, loss={avg_train_loss.item():.4f},val loss:,{avg_loss.item():.4f}") # .item() 的作用是：把只含一个元素的张量，转成普通的 Python 数字（float 或 int）

print("weight:", model.weight)
print("bias:", model.bias)
# 训练结束进行测试
model.eval()
total_test_loss = 0
with torch.no_grad():  # 验证的时候不需要计算梯度
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch = x_test_batch.unsqueeze(1)
        y_test_batch = y_test_batch.unsqueeze(1)
        prediction = model(x_test_batch)
        test_loss = criterion(
            prediction,
            y_test_batch
        )
        total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss/len(test_loader)
    print(f"Test MSE:{avg_test_loss.item():.4f}") # .item() 的作用是：把只含一个元素的张量，转成普通的 Python 数字（float 或 int）