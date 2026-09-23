samples = [
    "image_001.jpg",
    "image_002.jpg",
    "image_003.jpg",
    "image_004.jpg"
]

from torch.utils.data import Dataset   # 
class SimpleDataset(Dataset):   # 为什么 PyTorch 要规定 __len__ 和 __getitem__  因为 PyTorch 的 DataLoader 需要知道①一共有多少样本②第 i 个样本是什么 

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]

# 只要你的 Dataset 告诉 PyTorch  我有多少数据+ 第 i 个数据怎么拿 后面的 DataLoader 就可以接手

dataset = SimpleDataset(samples)

print(len(dataset))  # Python 就会自动调用 dataset.__len__() 
print(dataset[0])   # 等价于dataset.__getitem__(0)
print(dataset[1]) 

