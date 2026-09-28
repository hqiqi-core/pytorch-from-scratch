# nn.Linear 本身就是一个 PyTorch Module
import torch
import torch.nn as nn

model = nn.Linear(1, 1)

print(type(model))    # <class 'torch.nn.modules.linear.Linear'>  nn.Module 是 PyTorch 模型/网络层的基础类
print(isinstance(model, nn.Module))  # True


# class MyModel(nn.Module): 定义一个 PyTorch 模型
import torch
import torch.nn as nn

"""
PyTorch 需要通过 nn.Module 来管理：
Parameters
子模块
model.parameters()
model.to(device)
model.train()
model.eval()
state_dict()

所以写自定义模型时：

def __init__(self):
    super().__init__()

基本上是标准写法
"""

class MyModel(nn.Module):  # 继承了nn.Module

    def __init__(self):   # 定义模型“有什么”
        super().__init__()  # 调用父类的初始化逻辑
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):  # 数据怎么流过这个模型
        return self.linear2(self.linear1(x))


model = MyModel()

print(model)
print(model.linear)

x = torch.tensor([[3.0]])

y = model(x)

print("x:", x)
print("y:", y)
