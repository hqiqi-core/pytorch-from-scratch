import torch
# reshape 重塑形状
x = torch.arange(12)  # 一行12个元素
print(x)
print(x.shape)
y = x.reshape(3, 4)   # reshape成(3x4)矩阵
print(y)
print(y.shape)
y = x.reshape(2, 6) 
print(y)
print(y.shape)
y = x.reshape(4, 3) 
print(y)
print(y.shape)
y = x.reshape(3, 2, 2) 
print(y)
print(y.shape)

# unsqueeze：增加一个维度  很多 PyTorch 操作要求输入必须具有特定的维度
x = torch.tensor([1, 2, 3])
print(x.shape)  # torch.Size([3])
x = x.unsqueeze(0)   # 在第 0 个位置插入一个新维度
print(x)  # tensor([[1, 2, 3]])
print(x.shape)  # torch.Size([1, 3])  # 并没有增加数据，只是增加了一个维度 

x = torch.tensor([1, 2, 3])
x = x.unsqueeze(1)  # 在第 1 个位置插入一个新维度
print(x)  
print(x.shape)  # torch.Size([3, 1])

# squeeze：去掉长度为 1 的维度
# permute：交换多个维度——以后看 Transformer / ViT 代码非常重要的东西   可以一次重新排列所有维度
x = torch.randn(2, 3, 4)
print(x.shape)
y = x.permute(2, 0, 1)  #原来的第 2 维 → 新的第 0 维  原来的第 0 维 → 新的第 1 维  原来的第 1 维 → 新的第 2 维
print(y.shape)

# transpose：交换两个维度
x = torch.randn(2, 3)
print(x.shape)
y = x.transpose(0, 1)
print(y.shape)

# cat：拼接  
a = torch.tensor([[1, 2]])  # 这是二维 [1,2]
b = torch.tensor([[3, 4]])
c = torch.cat([a, b], dim=0)  # 上下拼  dim=1就是左右拼 [1,4]
print(c)
print(c.shape)  # [2,2]

# stack：增加一个新维度  = 创建一个新维度再堆起来
a = torch.tensor([1, 2])  #这是一维[2]
b = torch.tensor([3, 4])
c = torch.stack([a, b])
print(c)
print(c.shape)  #[2,2]  #上下叠

a = torch.tensor([1, 2, 3])  #[3]
b = torch.tensor([4, 5, 6])

print(a.unsqueeze(0).shape)  # 预测 [1,3]
print(a.unsqueeze(1).shape)  # 预测 [3,1]

print(torch.cat([a, b], dim=0).shape)  # 预测 [6]

print(torch.stack([a, b], dim=0).shape)  # 预测 [2,3]
print(torch.stack([a, b], dim=1).shape)  # 预测 [6,1]  x [3,2]

image = torch.randn(3, 224, 224)

print(image.shape)  # 预测[3,224,224]

image = image.unsqueeze(0)  

print(image.shape)  # 预测[1,3,224,224]