# 神经网络不能直接拿 Python list 做矩阵运算  神经网络真正处理的是Tensor ——PyTorch 世界里的多维数组
# Tensor 和 Python list 的区别  看起来很像，但 Tensor 多了很多神经网络需要的能力——GPU、自动求导、矩阵运算、神经网络计算、与 PyTorch 模型直接连接
import torch
x = torch.tensor([1,2,3])
print(x)  # tensor([1, 2, 3])
print(type(x))   # <class 'torch.Tensor'>

# Tensor 的 shape
print(x.shape)   # torch.Size([3])  意思是3个元素

y = torch.tensor([    # 二维tensor
    [1, 2, 3],
    [4, 5, 6]
])

print(y.shape)  # torch.Size([2, 3])  两行三列矩阵

z = torch.randn(2, 3, 4)   # 三维 Tensor
print(z.shape)  # torch.Size([2, 3, 4])  长方体，两个3x4矩阵

"""
为什么要学shape——(B, C, H, W)   (B, N, D)
图像 (B, C, H, W) (8, 3, 224, 224)  意思是  batch_size=8;通道数C=3(也就是RGB);高224;宽224.
Transformer token  (B, N, D)  (8, 196, 768)  意思是  8张图  196个token  每个token 768特征维度
"""

image = torch.randn(4,3,224,224)   
print(image.shape)  # torch.Size([4, 3, 224, 224])  3是通道  224 是高和宽  4是batch？
print(image.dtype)  # torch.float32   里面的元素是浮点数
print(x.dtype) # torch.int64
print(type(image))  # <class 'torch.Tensor'>

"""
常见类型 torch.float32  torch.float16  torch.bffloat16  torch.int64   神经网络参数和特征通常使用浮点数
# type 是“这个东西是什么对象”；dtype 是“这个对象里面的数据是什么类型”。
① type(x)    → x 是什么对象？
② x.shape    → x 的形状是什么？
③ x.dtype    → x 里面的数据是什么类型？
④ x.device   → x 在 CPU 还是 GPU？
"""
x = torch.tensor([1.,2.,3.])
y = torch.tensor([4.,5.,6.])
print(x+y)   # 逐元素相加  [5,7,9]
print(x*y)   # 逐元素相乘  [4,10,18]

# 矩阵乘法
a = torch.randn(2,3)
b = torch.randn(3,4)
c = a@b # c.shap=[2,4]   (2x3)@(3x4)=(2x4)
print(a.shape)
print(b.shape)
print(c.shape)

print(torch.cuda.is_available())
device = torch.device("cuda")
x = torch.randn(3,3).to(device=device)
print(x)
print(x.device)   #哪怕指定卡1啥的，但打印都是cuda:0因为被重映射了 

#CPU Tensor 和 GPU Tensor 不能随便混用
x = torch.randn(3,3)  # 默认cpu
y = torch.randn(3,3).to(device=device)  # GPU
# 这时候 x+y 因为两个tensor设备不同会报错

# 神经网络为什么能够训练？ Autograd
"""
y = wx
假设 w = 2  x = 3  那么y = 6
如果我们希望 y = 10  那么我们需要调整w  问题：w 应该往哪个方向调整？调整多少？这就是梯度发挥作用的地方
"""
# PyTorch 可以自动计算梯度
import torch
w = torch.tensor(2.0, requires_grad=True)   # requires_grad=True——告诉 PyTorch：我要追踪这个变量相关的计算，并且之后可能需要计算它的梯度
x = torch.tensor(3.0)
y = w * x
print(y)
loss = (y - 10) ** 2   # 等价于y = 3w  loss = (3w - 10)²
loss.backward()  # PyTorch 就会自动计算 w.grad   也就是backward() 从 loss 往回计算每个需要梯度的参数应该怎么变化  这就是反向传播的最基础版本
