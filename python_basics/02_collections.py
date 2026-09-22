list_a = [0,1,2,3,4,5,6,7,8,9,10]
print("The max num is ",max(list_a))
print("The min num is ",min(list_a))
class_label = [1,1,2,2,1,1,3,3,2,2]
count_1 = 0
count_2 = 0
count_3 = 0
# 这个可以直接用 for label in class_label
for i in range(len(class_label)):
    if class_label[i] == 1:
        count_1 +=1
    if class_label[i] == 2:
        count_2 +=1
    if class_label[i] == 3:
        count_3 +=1
print("类别1出现的次数:", count_1)
print("类别2出现的次数:", count_2)
print("类别3出现的次数:", count_3)

# 去重
unique_label0 = list(set(class_label))  # set()可以直接去重
print(unique_label0)
#自己实现
unique_label = []
for label in class_label:
    if label in unique_label:   #更自然的写法： if label not in unique_label :
        pass
    else :
        unique_label.append(label)
print(unique_label)

# 按分数排序——按照分数从高到低排列
samples = [    #这是一个列表  里面是元组  取元素用[][] 一个就取一行，两个00就是一行的第一个，01就是一行第二个
    ("image_001.jpg", 82),
    ("image_002.jpg", 95),
    ("image_003.jpg", 67),
    ("image_004.jpg", 88),
]

#现成的排序函数  sorted()  这个函数对于元组，默认会从第一个元素开始比较  但我们要看分数也就是第1个元素，因此sorted()有一个参数key 可以告诉sorted 你按照什么东西来排序
print(sorted(samples))  #按照图像名称排序了，而不是分数  
sorted(samples,key= lambda x: x[1])  #后面这个key相当于告诉python 对于每一个元素 x，拿它的第 1 个位置作为排序依据
# 默认排序是从小到大，我们要从大到小就需要用reverse=True参数  
print(sorted(samples,key= lambda x: x[1],reverse=True))

# 模拟医学图像类别统计
samples = [
    ("img001.jpg", "normal"),
    ("img002.jpg", "myopia"),
    ("img003.jpg", "myopia"),
    ("img004.jpg", "high_myopia"),
    ("img005.jpg", "normal"),
    ("img006.jpg", "myopia"),
    ("img007.jpg", "high_myopia"),
]
#任务1 统计次数
count_normal = 0   # 如果不知道具体有几个类别就不能这样写 而是要用字典
count_myopia = 0
count_high_myopia = 0
for x in samples:
    if x[1] == "normal":
        count_normal +=1
    elif x[1]=="myopia":
        count_myopia +=1
    elif x[1]=="high_myopia":
        count_high_myopia +=1
print("normal:",count_normal)
print("myopia:",count_myopia)
print("high_myopia:",count_high_myopia)

#统计每种类别占全部样本的比例
all_num = len(samples)
print(f"normal: {count_normal},({count_normal/all_num})")
print(f"myopia: {count_myopia},({count_myopia/all_num})")
print(f"high_myopia: {count_high_myopia},({count_high_myopia/all_num})")

# 用字典 
counts={}
for x in samples:
    category = x[1]
    counts[category] = counts.get(category, 0) + 1   # dict.get(key, default) 是根据键去取对应的值，如果键不存在，不会报错，而是返回你指定的默认值

print(counts)

total = len(samples)

for category, count in counts.items():  #这个items可以把字典中的key和value成对取出来  category, count是定义的变量名字
    ratio = count / total
    print(f"{category}: {count}, ({ratio:.2%})")  # .2%就表示把小数格式化成百分比 保留2位小数  如果只是.2f则只是保留两位小数不会变成百分比



# =========================
# Set 集合
# =========================

labels = ["normal", "myopia", "myopia", "normal", "high_myopia", "myopia"]

unique_labels = set(labels)  # set自动去掉重复元素，不强调元素顺序

print(unique_labels)

labels = [
    "normal",
    "myopia",
    "myopia",
    "high_myopia",
    "normal",
    "myopia",
    "high_myopia"
]

unique_labels1 = set(labels)
print(unique_labels1)
print(len(unique_labels1))

unique_labels2 = []
for label in labels:
    if label not in unique_labels2:
        unique_labels2.append(label)
print(unique_labels2)
print(len(unique_labels2))


numbers = [1, 2, 3, 4, 5]
squares = []
#  也可以写成 squares = [x * x for x in numbers]   这叫列表推导式  对于 numbers 里面的每一个 x，把 x*x 放进新的列表
for x in numbers:
    squares.append(x * x)   

print(squares)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares1 = []
for x in numbers:
    squares1.append(x*2)
print(squares1)

squares2 = [x *2 for x in numbers]
print(squares2)
# 练习 1：提取偶数
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares3 = [x for x in numbers if x%2==0]
print(squares3)

# 练习 2：提取医学图像文件名
samples = [
    ("img001.jpg", "normal"),
    ("img002.jpg", "myopia"),
    ("img003.jpg", "high_myopia"),
    ("img004.jpg", "myopia"),
]

squares4 = [x[0] for x in samples]
print(squares4)
# 练习 3：只提取 myopia 图像
samples = [
    ("img001.jpg", "normal"),
    ("img002.jpg", "myopia"),
    ("img003.jpg", "high_myopia"),
    ("img004.jpg", "myopia"),
]

squares5 = [x[0] for x in samples if x[1]=="myopia"]
print(squares5)

# String 字符串
filename = "patient_001_fundus.jpg"
# 判断是不是 jpg
print(filename.endswith(filename))   # print(filename.endswith(".jpg"))   endswith() 的意思——是否以括号里传的字符串结尾

# 把 .jpg 改成 .png
filename1=filename.replace(".jpg",".png")
print(filename1)

# 任务 3：把文件名按照 _ 分开
str = []   # 最好不要用str做变量名，因为是一个内置类型
str = filename.split("_")
print(str)

# 任务 4：自己再创建一个字符串
category = "  High_Myopia  "
str1 = category.strip(category.lower())   # strip是去掉首尾所有出现在 chars 中的字符。默认 strip() 去空白
print(str1)
str1 = category.strip().lower() #改正
print(str1)

# 文件读取
labels = []
with open("python_basics/data/labels.txt","r") as f:   #下面多次读取指针到文件最后了，所以没有输出
    # f.readline()  #只读一行
    # 方法1
    # lines = f.readlines()  #读所有行 每行后面有\n 换行符  可以用strip（）去掉
    # 方法2
    # for line in f :    #for循环得到所有行
    #     label = line.strip()   #strip() 去掉行尾换行符和首尾空格
    labels = [line.strip().lower() for line in f if line.strip()]   #labels是个列表
    # # 方法3  读整个文件
    # content = f.read()
    # labels1 = content.splitlines()  

counts = {}
for label in labels :
    counts[label] = counts.get(label,0)+1

total = len(labels)

for label,count in counts.items():
    ratio = count / total
    print(f"{label}:{count},{ratio:.2%}")   # : 代表后面开始写格式说明

  