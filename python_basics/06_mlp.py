import torch
import torch.nn as nn

"""
典型神经网络
Linear
 ↓
Activation
 ↓
Linear
 ↓
Activation
 ↓
Linear
"""

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        # 两个线性层连续起来，本质上仍然是一个线性变换
        self.linear1 = nn.Linear(1, 10)  # 输入 feature = 1 输出 feature = 10  [Batch, 1]->[Batch,10]
        self.linear2 = nn.Linear(10, 1)  # 输入 feature = 10 输出 feature = 1  [Batch, 10]->[Batch,1]
        # 所以我们需要加入Relu——使模型具有非线性表达能力
        self.relu = nn.ReLU()  # ReLU(x) = max(0, x)
        # PyTorch 还提供 nn.Sequential  把多个层按照顺序串起来  
        self.network = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )



    def forward(self, x):
        #x = self.linear1(x)
        #x = self.relu(x)
        #x = self.linear2(x)
        x = self.network(x)  #!!!但不要认为 Sequential 可以替代 forward
        return x


model = MLP()
print(model)

x = torch.tensor([[3.0]])
y = model(x)

print("x shape:", x.shape)
print("y shape:", y.shape)


