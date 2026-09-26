import torch
w = torch.tensor(2.0,requires_grad=True)
print("w.grad:",w.grad)
x = torch.tensor(3.0)
y = w*x
loss = (y-10)**2
print("y:",y)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss:",loss)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss.backward()   # 计算∂loss/∂w
print("w.grad:",w.grad)  # w.grad: tensor(-24.)
"""
loss=(y−10)^2 =(wx-10)^2
∂loss/∂w =  ∂loss/∂y * ∂y/∂w =2(y-10)* x  =2(6-10)*3=-24  -24，也就是梯度是负数，这意味着——现在 w=2 太小了，应该把 w 往增大的方向调整
backward() 计算 loss 对各个需要梯度的变量的梯度。这里我们需要的是∂loss/∂w。计算过程中会通过链式法则经过 y，其中确实会计算 ∂y/∂w
训练的时候最基本的更新公式是wnew​=w-η*(∂loss​/∂w)
假设学习率 η=0.1  wnew​=2−0.1(−24)=4.4

梯度的正负，告诉我们应该往哪个方向调整参数  梯度下降本质上就是根据当前位置的斜率，决定参数往哪边走  负数说明参数小了 要变大  正数说明参数大了 要变小
沿着梯度的反方向更新参数

这个：
grad_fn=<MulBackward0>
grad_fn=<PowBackward0>
就是 PyTorch 在背后记录：
w ──×=3──> y ──平方──> loss
所以执行：
loss.backward()时，PyTorch 才能够沿着这条计算关系反向传播

神经网络训练最基本的流程:
输入
 ↓
模型
 ↓
预测 y
 ↓
计算 loss
 ↓
loss.backward()
 ↓
得到参数梯度
 ↓
optimizer.step()   根据梯度真正修改 w
 ↓
参数更新
 ↓
重新预测
 ↓
重新计算 loss
 ↓
……
"""

w1 = torch.tensor(1.0,requires_grad=True)
w2 = torch.tensor(2.0,requires_grad=True)
w3 = torch.tensor(3.0,requires_grad=True)
w4 = torch.tensor(4.0,requires_grad=True)
w5 = torch.tensor(5.0,requires_grad=True)

y1 = w1*x
y2 = w2*x
y3 = w3*x
y4 = w4*x
y5 = w5*x
print("####w=1###")
loss1 = (y1-10)**2
print("y1:",y1)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss1:",loss1)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss1.backward()   # 计算∂loss/∂w
print("w1.grad:",w1.grad)  # w.grad: tensor(-24.)

print("####w=2###")
loss2 = (y2-10)**2
print("y2:",y2)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss2:",loss2)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss2.backward()   # 计算∂loss/∂w
print("w2.grad:",w2.grad)  # w.grad: tensor(-24.)

print("####w3###")
loss3 = (y3-10)**2
print("y3:",y3)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss3:",loss3)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss3.backward()   # 计算∂loss/∂w
print("w3.grad:",w3.grad)  # w.grad: tensor(-24.)

print("####w=4###")
loss4 = (y4-10)**2
print("y4:",y4)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss4:",loss4)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss4.backward()   # 计算∂loss/∂w
print("w4.grad:",w4.grad)  # w.grad: tensor(-24.)

print("####w=5###")
loss5 = (y5-10)**2
print("y5:",y5)   # y: tensor(6., grad_fn=<MulBackward0>)
print("loss5:",loss5)  # loss: tensor(16., grad_fn=<PowBackward0>)
loss5.backward()   # 计算∂loss/∂w
print("w5.grad:",w5.grad)  # w.grad: tensor(-24.)

print("对比：")
print("w1.grad:",w1.grad)
print("w2.grad:",w2.grad)
print("w3.grad:",w3.grad)
print("w4.grad:",w4.grad)
print("w5.grad:",w5.grad)

#  让w 自己训练
w = torch.tensor(1.0,requires_grad=True)
learning_rate = 0.1

for i in range(10):
    y = w*3
    loss = (y - 10) ** 2
    loss.backward()
    print(f"step:{i},w:{w},loss:{loss},grad:{w.grad}")
    # 训练参数更新时通常需要 with torch.no_grad() 不然会把新的计算关系继续放进 autograd 图里
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_() # 清除上一轮梯度   PyTorch 默认情况下，梯度是累加的 
