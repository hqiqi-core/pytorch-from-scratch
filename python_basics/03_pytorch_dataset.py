import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

samples = [
    "image_001.jpg",
    "image_002.jpg",
    "image_003.jpg",
    "image_004.jpg",
]

class SimpleDataset(Dataset):
    def __init__(self,samples):
        super().__init__()  # 调用父类 Dataset 的初始化方法  super() = 调用父类的东西
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]

dataset = SimpleDataset(samples)

print(len(dataset))
print(dataset[0])
print(dataset[2])

for i in range(len(dataset)):
    print(i, dataset[i])

loader = DataLoader(dataset,batch_size=2,shuffle=False)
for batch in loader:
    print(batch)

