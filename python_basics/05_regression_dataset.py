import torch
from torch.utils.data import Dataset, DataLoader

x = [1, 2, 3, 4, 5]
y = [5, 8, 11, 14, 17]
samples = [(1.0,5.0),(2.0,8.0),(3.0,11.0),(4.0,14.0),(5.0,17.0)]

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

print("dataset length:", len(dataset))

x, y = dataset[0]

print("x:", x)
print("y:", y)
print("x shape:", x.shape)  # torch.Size([]) 是 PyTorch 中 0 维张量（标量） 
print("y shape:", y.shape)

loader = DataLoader(
    dataset,
    batch_size=8,  # 每次取8个样本
    shuffle=True,
    drop_last=True
)

for x_batch, y_batch in loader:

    print("x_batch:", x_batch)
    print("y_batch:", y_batch)

    print("x_batch.shape:", x_batch.shape)
    print("y_batch.shape:", y_batch.shape)

    break

