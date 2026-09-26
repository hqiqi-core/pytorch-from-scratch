import torch

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)

y = w * x + b

print("y:", y)
print("w:", w)
print("b:", b)

# nn.Linear 实际上就是 y=xW^T+b
import torch
import torch.nn as nn

linear = nn.Linear(1, 1)  # in_feature,out_feature

print(linear)
print(linear.weight)  # 它们不是普通的 Tensor，而是 torch.nn.Parameter
print(linear.bias)

print(type(linear.weight))
print(type(linear.bias))

print("Parameters:")
for param in linear.parameters():
    print(param)

x = torch.tensor([[3.0]])

y = linear(x)

print("x:", x)
print("y:", y)
print("y.shape:", y.shape)

"""
为什么需要 Parameter ——因为一个神经网络可能有几百万甚至几十亿个参数  我们不可能自己写  所以 PyTorch 提供了 Parameter 机制  
只要一个东西属于模型 并且被注册成 Parameter，PyTorch 就知道 这个东西是模型需要学习的参数  这就引出了 model.parameters() 
model.parameters() = 把模型中所有需要训练的参数拿出来
nn.Linear
    ↓
model.parameters()
    ↓
optimizer
"""

target = torch.tensor([[10.0]])

loss = (y - target) ** 2

print("loss before backward:", loss)

loss.backward()

print("weight grad:", linear.weight.grad)
print("bias grad:", linear.bias.grad)

