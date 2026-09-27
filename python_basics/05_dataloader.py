from torch.utils.data import Dataset
import torch
samples = [(1.0,5.0),(2.0,8.0),(3.0,11.0),(4.0,14.0),(5.0,17.0)]  # 里面是元组的列表  输出不是tensor
# 最终返回的 x 和 y 都要是 Tensor
x = [1, 2, 3, 4, 5]
y = [5, 8, 11, 14, 17]
# 以上三个变量 里面的元素本身都是 Python int  Dataset 不负责自动转换  所以返回什么完全取决于__getitem__怎么写
"""
Python 数据   (可以在 Dataset 里面转换  torch.tensor(x[i], dtype=torch.float32))
    ↓
Dataset
    ↓
Tensor
    ↓
DataLoader
    ↓
Batch Tensor
"""

class SimpleDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.samples = []
        for i in range(len(x)):
            self.samples.append((x[i],y[i]))
    def __len__(self):
        return len(self.samples)
    #def __getitem__(self, index):
        #return self.samples[index]  # 这样返回不是tensor 还是python类型
    def __getitem__(self, index):
        x, y = self.samples[index]

        x = torch.tensor(x, dtype=torch.float32)  # 手动转换类型
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

# dataset = SimpleDataset(x,y)

# print(len(dataset))
# print(dataset[0]) 
# print(dataset[1]) 

# print(type(dataset[0][0]))
# print(type(dataset[0][1]))

# print(dataset[0][0].dtype)
# print(dataset[0][1].dtype)

# 训练模型 一般需要一批数据一批数据输入 而不是一个数据一个数据  这就引入了batch
from torch.utils.data import DataLoader
dataset = SimpleDataset(x, y)
loader = DataLoader(   # 从 Dataset 中按照 batch_size 取样本，并把多个样本组织成 Batch，同时还可以 shuffle
    dataset,
    batch_size=2,   # 每次从 Dataset 中取 2 个样本
    shuffle=True,   # 设置为True的话就可以在每个 epoch 开始时把 Dataset 的样本顺序打乱  训练集通常True 验证和测试False
    drop_last=True  # 把不完整 Batch 丢掉  5/2 多了一个 设置这个True的话就会把最后一个丢掉
)
for batch in loader:
    x_batch, y_batch = batch

    print("x_batch:", x_batch)
    print("x_batch.shape:", x_batch.shape)

    print("y_batch:", y_batch)
    print("y_batch.shape:", y_batch.shape)

# 模型处理一次 Batch = 1 iteration    整个 Dataset 完整遍历一次 = 1 epoch
