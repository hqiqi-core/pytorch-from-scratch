import torch
import torch.nn as nn

model = nn.Linear(1, 1)

optimizer = torch.optim.SGD(
    model.parameters(),  # 把模型中所有需要训练的参数交给优化器
    lr=0.01           # 更新学习率
)

x = torch.tensor([[3.0]])  
target = torch.tensor([[10.0]])

for step in range(20):

    # 1. forward
    y = model(x)

    # 2. loss
    loss = (y - target) ** 2

    # 3. backward
    loss.backward()

    # 4. update
    optimizer.step()   # 也就是之前写的 with no_grad那两句

    # 5. clear gradients
    optimizer.zero_grad()

    print(
        f"step={step}, "
        f"loss={loss.item():.4f}, "
        f"weight={model.weight.item():.4f}, "
        f"bias={model.bias.item():.4f}"
    )