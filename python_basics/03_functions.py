# 练习1：计算两个数字的和
def add(a,b):
    return a+b   # return 是把函数内部产生的结果交给函数外部

result = add(3,5)
print(result)

def get_info():
    name = "HQQ"
    age = 24
    return name, age   # 实际上返回的是一个 tuple ("HQQ", 24)

name, age = get_info()

print(name)
print(age)

from dataset_manager import load_labels,count_labels,print_statistics
path = "python_basics/data/labels.txt"
labels = load_labels(path)
counts = count_labels(labels)
print_statistics(labels,counts)
