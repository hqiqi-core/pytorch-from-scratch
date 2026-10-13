"""
使用程序生成的小型图文数据——验证训练机制
B = 12
C = 3
H = W = 64
L = 5
V = vocabulary size
E = 64
D = 32

图像分支：  不要把空间维度直接硬编码为 32*16*16，使用自适应池化能降低输入尺寸耦合
images
[B, 3, 64, 64]
float32
cuda:0
    ↓ Conv2d(3, 16, 3, padding=1)
[B, 16, 64, 64]
    ↓ ReLU + MaxPool2d(2)
[B, 16, 32, 32]
    ↓ Conv2d(16, 32, 3, padding=1)
[B, 32, 32, 32]
    ↓ ReLU + MaxPool2d(2)
[B, 32, 16, 16]
    ↓ AdaptiveAvgPool2d((1,1))
[B, 32, 1, 1]
    ↓ flatten(1)
[B, 32]
    ↓ image_projection: Linear(32, D)
[B, 32]
    ↓ normalize(dim=-1)
[B, 32]

文本分支：
<PAD>, <UNK>, a, red, green, blue, yellow, square, circle, triangle

["a", "red", "square"]
      ↓ token id
[2, 3, 7]
      ↓ padding 到 L=5
[2, 3, 7, 0, 0]
batch后
input_ids
[B, L] = [12, 5]
torch.long
cuda:0
    ↓ Embedding(V, E)
[B, L, E] = [12, 5, 64]
float32
cuda:0
    ↓ mask padding
mask: [B, L, 1]
    ↓ masked mean pooling
[B, E] = [12, 64]
    ↓ text_projection: Linear(E, D)
[B, D] = [12, 32]
    ↓ normalize(dim=-1)
[B, D] = [12, 32]

Masked mean：
mask = input_ids.ne(pad_id).unsqueeze(-1)
summed = (token_embeddings * mask).sum(dim=1)
lengths = mask.sum(dim=1).clamp_min(1)
pooled = summed / lengths
注意：
- input_ids 必须是 torch.long。
- nn.Embedding 不能接收 float token id。
- padding 不应参与文本平均。
- lengths 不能为零。

image_features: [12, 32], float32, cuda:0
text_features:  [12, 32], float32, cuda:0
text_features.T:[32, 12], float32, cuda:0

logits:         [12, 12], float32, cuda:0
labels:         [12],     int64,   cuda:0
loss:           [],       float32, cuda:0

"""

import argparse
from pathlib import Path
import random

import torch
import math
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn.functional as F

COLORS = {
    "red":   (220, 50, 50),
    "green": (50, 180, 80),
    "blue":  (60, 100, 220),
}
SHAPES = ["square", "circle", "triangle"]
BG_COLOR = (240, 240, 240)   # 浅灰


# 随机种子与设备
def set_seed(seed: int) -> None:
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str = "auto") -> torch.device:
    """把 'auto'/'cpu'/'cuda'/'cuda:1' 解析成 torch.device。"""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# 生成配对图片
def create_toy_dataset(root: Path, image_size: int = 64) -> list[dict[str, str]]:
    """
    要求生成一批"图片 + 描述"的配对数据，用于训练/测试像 CLIP 那样的图文模型。所以本质是：用程序批量画出一批简单图片，每张图有确定的颜色和形状，同时记录一句对应的英文描述。
    使用 PIL 创建 RGB 图片。
    每张图有一种颜色和一种形状。
    背景建议用浅灰或白色。  形状颜色（红绿蓝）和背景要有对比度，否则模型学不到形状
    形状放在图片中央，并留出边缘。  如果形状贴边，旋转、裁剪、resize 时容易丢信息。一般四周留 10%~15% 的边距，形状居中。
    颜色和形状列表做笛卡尔积 → 3 × 3 = 9 张图。这就是"固定遍历顺序"的来源：只要颜色列表、形状列表顺序固定，生成结果就固定。 
    返回记录：
    {
        "image_path": "data/simple_clip/red_square.png",
        "caption": "a red square",
    }
    """
    root = Path(root)  # 再次 Path(root) 是为了容错：即使调用方传的是字符串，函数内部也统一成 Path。
    root.mkdir(parents=True, exist_ok=True)  # parents=True：如果 data/simple_clip/ 的父目录 data/ 不存在，也一起创建
    records: list[dict[str, str]] = []

    # 固定遍历顺序：先颜色，后形状
    for color_name, rgb in COLORS.items():
        for shape in SHAPES:
            img = Image.new("RGB", (image_size, image_size), BG_COLOR)   # "RGB" 模式，三通道 尺寸是正方形 (size, size) 
            draw = ImageDraw.Draw(img)  # ImageDraw.Draw 是 PIL 的画笔对象，后续用它画图

            # 留边距，形状居中
            margin = image_size // 8    # 64 像素时，margin=8，形状占据 8~56 的范围，居中且留边  用 // 整除，保证是整数像素（PIL 的坐标要整数）
            x0, y0 = margin, margin  # 形状的"外接框"统一用 [x0, y0, x1, y1]，这样 square / circle 共用一套坐标，代码更简洁
            x1, y1 = image_size - margin, image_size - margin

            if shape == "square":
                draw.rectangle([x0, y0, x1, y1], fill=rgb)   # # 正方形 
            elif shape == "circle":
                draw.ellipse([x0, y0, x1, y1], fill=rgb)   # 圆（外接框内切圆）
            elif shape == "triangle":
                cx = image_size / 2
                draw.polygon(   # 三角形
                    [(cx, y0), (x0, y1), (x1, y1)],  # 传顶点列表，这里用上中点 + 左下 + 右下，画出一个等腰三角形，顶点朝上
                    fill=rgb,
                )

            filename = f"{color_name}_{shape}.png"
            img_path = root / filename   # root / filename 是 Path 的除法运算符，跨平台安全
            img.save(img_path)  

            records.append({
                "image_path": str(img_path),
                "caption": f"a {color_name} {shape}",
            })

    return records
    


# 实现 tokenizer   （分词器） "a red square"  →  tokenizer  →  [2, 3, 4]  →  神经网络 
"""
为什么需要——模型不认识文字，只认识数字。所以要把 "a red square" 变成 [2, 3, 4] 这样的整数序列 
模型喜欢定长输入（比如一批样本都得是长度 5），而句子长短不一。所以要：太长的 → 截断（truncate） 太短的 → 补齐（pad）
这就是 max_length 和 <PAD> 的作用。
token	id	作用
<PAD>	0	占位符，补到定长用，模型会 mask 掉       为什么 PAD 要固定为 0？ 因为 0 常用于做 mask（attention_mask = (ids != 0)），是约定俗成
<UNK>	1	遇到没见过的词时的兜底
普通词	≥2	 真正的词汇

在初始化把传入的captions进行分词，然后赋予从2开始的id，然后
输入一个caption就是调用encode方法，查这个query的每个词（先小写再分词），然后用每个词再建立的字典里面查找每个词对应的id,查得到就用它的 id，查不到就用 UNK 的 id（1），而不是报错或跳过。
然后如果长度大于最大长度就直接截断。如果不到就补齐，最后就是返回这个query的每个词的id序列
vocab = 两个特殊 token（PAD=0, UNK=1） + 一个普通词的字典。
空字符串 "" 也要走完整流程，返回全 PAD 的 [0,0,0,0,0]
""" 

class SimpleTokenizer:
    def __init__(self, captions: list[str], max_length: int = 5):  
        """
        把所有 caption 收集起来，lower + split，得到所有出现过的词。给每个词分配一个 id（从 2 开始，因为 0/1 被特殊 token 占了）。保存 max_length。
        """
        self.max_length = max_length
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.PAD_ID = 0
        self.UNK_ID = 1

        # 1. 收集所有出现过的词
        words = set()  # 用 set 去重
        for cap in captions:
            for tok in cap.lower().split():  # 对 captions 做了 lower + split，和 encode 保持一致
                words.add(tok)

        # 2. 排序保证顺序稳定，从 2 开始分配 id
        sorted_words = sorted(words)   # sorted 保证同一批词得到的 id 每次都一样
        self.word2id = {w: i + 2 for i, w in enumerate(sorted_words)}  # i + 2 因为 0/1 被特殊 token 占了
        self.id2word = {i: w for w, i in self.word2id.items()}  # 反查表，方便调试


    @property
    def vocab_size(self) -> int:  # embedding 层就需要 vocab_size 行  包括那两个特殊词的数量
        # PAD + UNK + 所有普通词
        return 2 + len(self.word2id)

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    def encode(self, text: str) -> torch.Tensor:
        """
        lower + split。
        每个词查表 → id，查不到用 UNK。
        截断到 max_length。
        不够的补 PAD，直到长度等于 max_length。
        转成 torch.long Tensor，形状 [L]。
        """
        # lower + split，注意 "" 会得到 []
        tokens = text.lower().split()  

        # 查表，未登录词 -> UNK
        ids = [self.word2id.get(tok, self.UNK_ID) for tok in tokens]   # dict.get(key, default) 是字典查表带兜底的标准写法。未登录词自动变 UNK。

        # 截断
        ids = ids[: self.max_length]

        # 补齐
        if len(ids) < self.max_length:
            ids = ids + [self.PAD_ID] * (self.max_length - len(ids))

        # 转 tensor，dtype 必须是 long
        return torch.tensor(ids, dtype=torch.long)

# 实现dataset
"""
PyTorch 训练时的标准姿势就是：dataset[i] 拿单条样本，DataLoader 负责拼 batch、打乱、并行加载。所以 Dataset 的职责非常明确：
给我一个 index，我还你一条已经张量化好的样本。
image 和 input_ids 是给模型用的张量

"""
class ImageTextDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, str]],
        tokenizer: SimpleTokenizer,
        image_size: int = 64,
    ):
        """
        PIL RGB → resize 到 [64,64] → Tensor [3,64,64] → float32 → 数值范围 [0,1]
        为什么要PIL RGB——Image.open(path) 读出来的图片可能是灰度、RGBA、CMYK 等各种模式。模型要的是三通道，所以必须 convert("RGB") 统一。这一步不能省：如果某张图是 PNG 带透明通道（RGBA），直接转 Tensor 会变成 4 通道，和模型输入对不上。
        为什么要Tensor [3,64,64]——PIL 图片的内存布局是 [H, W, C]（高在前，通道在后），PyTorch 要的是 [C, H, W]（通道在前）。ToTensor() 会自动帮你完成这个转置。
        为什么要float32 + [0,1]——原始像素是 uint8，范围 0~255。神经网络喜欢小的浮点数，所以统一归一化到 [0,1]。ToTensor() 同时做了两件事：转 dtype 到 float32 + 除以 255。
        必须 Resize 在 ToTensor 前，因为 Resize 接受 PIL Image，ToTensor 之后就是 Tensor 了。
        图片 → dataset[i]["image"] 是 [3, 64, 64]
        文本 → input_ids 是 [L]
        """
        if not samples:
            raise ValueError("samples 不能为空")

        self.samples = samples
        self.tokenizer = tokenizer
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # Resize(64) → 短边缩到 64，长边等比缩放，不保证是正方形。 Resize((64, 64)) → 强制缩成 64×64
            transforms.ToTensor(),  # → [3,H,W] float32, [0,1]
        ])


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:

        sample = self.samples[index]  #  如果 index 越界，Python 自动抛 IndexError，不用手动查

        image_path = sample["image_path"]
        caption = sample["caption"]

        # 图片：PIL RGB → resize → tensor → float32 [0,1]
        img = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(img)   # [3, H, W] float32

        # 文本：caption → id 序列 [L] long
        input_ids = self.tokenizer.encode(caption)

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "caption": caption,
            "image_path": image_path,
        }


# 实现图像编码器
class TinyImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),# Conv2d 的 padding=1 是干嘛的？kernel_size=3 的卷积会让特征图每边缩小 1（64 → 62）。加 padding=1 后，输出尺寸和输入一样，空间尺寸只在 MaxPool 时才减半，逻辑更清晰。
            nn.ReLU(),
            nn.MaxPool2d(2), # 两次 MaxPool(2) 是干嘛的？每次把空间尺寸减半：64 → 32 → 16。这样既降维又增大感受野，是 CNN 的标准套路。
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),# AdaptiveAvgPool2d((1,1)) 是干嘛的？把 [B, 32, H, W] 压成 [B, 32, 1, 1]——每个通道取全局平均 它让网络能接受任意空间尺寸的输入（不管是 64 还是别的）。它把"空间信息"聚合成"通道特征"，是全局池化的标准做法
        )

        self.proj = nn.Linear(32, output_dim)
        # [B,32,1,1] → flatten → [B,32] → Linear(32, output_dim) → [B,output_dim]


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images 应为 4 维 [B,C,H,W]，实际 {images.ndim} 维")
         
        if images.shape[1] != 3:
            raise ValueError(
                f"images 通道应为 3，实际 {images.shape[1]}"
            )
        x = self.features(images)     # [B, 32, 1, 1]
        x = x.flatten(1)              # [B, 32]
        x = self.proj(x)              # [B, output_dim]
        return x                      # 未做 L2 归一化

# 实现文本编码器
"""
input_ids  [B, L]  (long)
    │
    ▼                          padding_idx=pad_id 有什么用?  初始化时：PAD 那一行的向量被设成全 0（并且不会被梯度更新）  语义上：告诉 PyTorch"这个 id 是占位符，没意义"  虽然这个参数不会自动帮你做 mask，但它保证了 PAD 的 embedding 是 0。后面算 mean pooling 时，mask 会把这个 0 排除掉，两者配合才正确
nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)  把一个整数 id 变成一个长度为 embedding_dim 的向量。本质是一张可学习的查找表：vocab_size 行，embedding_dim 列  输入 [B, L] 的整数 → 输出 [B, L, embedding_dim] 的浮点
    │
    ▼
embedded   [B, L, embedding_dim]
    │
    ▼
masked mean pooling（忽略 PAD 位置）   只对真实 token 求平均  mask 怎么构造？ mask = (input_ids != pad_id)   # [B, L] bool，True 表示真实 token
    │
    ▼
pooled     [B, embedding_dim]
    │
    ▼
Linear(embedding_dim, output_dim)
    │
    ▼
out        [B, output_dim]
"""
class TinyTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        self.proj = nn.Linear(embedding_dim, output_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 应为 2 维 [B,L]，实际 {input_ids.ndim} 维"
            )
        if input_ids.dtype != torch.long:
            raise ValueError(
                f"input_ids dtype 应为 torch.long，实际 {input_ids.dtype}"
            )
         # 1. embedding: [B,L] -> [B,L,D]
        embedded = self.embedding(input_ids)

        # 2. masked mean pooling
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()  # [B,L,1]  unsqueeze(-1)：把 [B,L] 变 [B,L,1]，好和 [B,L,D] 广播相乘  bool 不能直接乘 float，转一下。True→1.0, False→0.0
        summed = (embedded * mask).sum(dim=1)                    # [B,D]   PAD 位置乘 0，被清零  沿 L 维求和 → [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)                  # [B,1]  每个样本真实 token 数量 → [B, 1]  防止除零！如果某个样本全是 PAD（比如空字符串）  L 维，被 sum 吃掉  每个结果就是每个样本的token综述
        pooled = summed / counts                                 # [B,D]  每个样本用它自己的 token 数去除自己的特征向量

        # 3. 投影
        return self.proj(pooled)                                 # [B,output_dim]
    


# 实现 SimpleCLIP  用两个编码器分别得到图像向量、文本向量，各自做 L2 归一化，然后算相似度矩阵，返回给训练用。
class SimpleCLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 两个编码器输出都对齐到 embed_dim
        self.image_encoder = TinyImageEncoder(output_dim=embed_dim)
        self.text_encoder = TinyTextEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=64,
            output_dim=embed_dim,
        )
        # 可学习温度：存 log(1/τ)，exp 后为正
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:   # 单独编码图
        features = self.image_encoder(images)          # [B, D]
        return F.normalize(features, p=2, dim=-1)      # [B, D] 单位长度

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:  # 单独编码文
        features = self.text_encoder(input_ids)        # [B, D]
        return F.normalize(features, p=2, dim=-1)      # [B, D] 单位长度

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"images 和 input_ids 的 batch size 不一致："
                f"{images.shape[0]} vs {input_ids.shape[0]}"
            )

        image_features = self.encode_image(images)     # [B, D]
        text_features = self.encode_text(input_ids)    # [B, D]
        # 温度缩放：exp 保证正，clamp 防爆炸
        scale = self.logit_scale.exp().clamp(max=100.0)

        # 相似度矩阵：点积 = 余弦相似度（因为已归一化）
        logits_per_image = scale * (image_features @ text_features.T)  # [B, B]
        
        return logits_per_image, image_features, text_features

"""
logits_per_image 
              text_0  text_1  text_2  text_3
image_0  [  0.9     0.2     0.1    0.3  ]   ← 应该最大的是 text_0（对角线）
image_1  [  0.1     0.8     0.3    0.2  ]
image_2  [  0.2     0.3     0.7    0.1  ]
image_3  [  0.3     0.1     0.2     0.6  ]
对角线上是正样本配对（第 i 张图 + 第 i 条文本）。非对角线是负样本（配错了）
训练目标就一句话：让每行对角线那个值尽可能大，其他值尽可能小。
F.cross_entropy(input, target) 的语义是：对 batch 里的每一个样本，做一次"多分类"的交叉熵，然后取平均。
input 是 [N, C]：N 个样本，每个样本有 C 个类别的分数。
target 是 [N]：N 个样本，每个样本的正确类别下标。
不是"二维矩阵和一维向量做运算"，而是"二维矩阵的每一行配一个标量标签"。
第 1 步：对每一行做 softmax
行 0: softmax([0.9, 0.2, 0.1]) ≈ [0.53, 0.24, 0.23]
行 1: softmax([0.1, 0.8, 0.3]) ≈ [0.24, 0.53, 0.23]
行 2: softmax([0.2, 0.3, 0.7]) ≈ [0.23, 0.26, 0.51]
每行都变成一个概率分布，和为 1。

第 2 步：取"正确类别"对应的概率，取 -log
样本 0 的正确类是 labels[0] = 0 → 概率 0.53 → 损失 -log(0.53) ≈ 0.63
样本 1 的正确类是 labels[1] = 1 → 概率 0.53 → 损失 0.63
样本 2 的正确类是 labels[2] = 2 → 概率 0.51 → 损失 0.67

第 3 步：对样本求平均
loss = (0.63 + 0.63 + 0.67) / 3 ≈ 0.64
这就是 F.cross_entropy(logits, labels) 返回的标量。

拒绝 batch size 1——每行只有一个类别，softmax 后概率恒为 1，loss 恒等于 0，梯度也是 0。
对比学习需要"正样本 vs 负样本"。这个 batch 里，图 0 的唯一配对就是文 0，没有任何负例。没有负例就没有对比，没有对比就没有学习信号

"""

# 实现 CLIP loss
def clip_contrastive_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    if logits_per_image.ndim != 2:
        raise ValueError(
            f"logits_per_image 应为 2 维 [B,B]，实际 {logits_per_image.ndim} 维"
        )
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError(
            f"logits_per_image 应为方阵，实际 {tuple(logits_per_image.shape)}"
        )

    batch_size = logits_per_image.shape[0]

    if batch_size < 2:
        raise ValueError("batch size must be at least 2 for in-batch negatives")

    # 对角线是正样本：图 i ↔ 文 i
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # 图 → 文
    loss_i2t = F.cross_entropy(logits_per_image, labels)

    # 文 → 图（转置后再算一遍）
    loss_t2i = F.cross_entropy(logits_per_image.T, labels)

    # 双向平均
    return (loss_i2t + loss_t2i) / 2


# 训练一个 epoch  
"""
一个 epoch 就是"把整个数据集过一遍"。每过一个 batch，做五件事 前向 → 算 loss → 反向 → 更新参数 → 清零梯度
zero_grad()	把 .grad 里的值全部填 0
zero_grad(set_to_none=True)	把 .grad 设为 None
为什么 set_to_none=True 更好？
省内存：None 不占存储，填 0 要分配同样大的张量。
省时间：填 0 是内存写操作，set_to_none 只是改指针。
语义更准：None 表示"还没有梯度"，0 表示"梯度是 0"，两者语义不同。
model.train() 把模型里所有模块切成训练模式。影响的主要是：
Dropout：训练时启用，推理时关闭。
BatchNorm：训练时用 batch 统计量并更新 running stats，推理时用 running stats。
本任务的模型没有 Dropout 和 BatchNorm，所以 model.train() 实际上没什么可见效果。但必须写
不能漏掉 .to(device)
不能对训练特征调用 .detach()——因为.detach() 会切断梯度流
不能在 forward 里用 torch.no_grad()——torch.no_grad() 会关闭梯度记录。之后 loss.backward() 会报错  no_grad 是给推理/评估用的，训练时绝对不能包在 forward 外面
为什么不能用 loss.item() 替代 Tensor？——loss.item() 把张量变成 Python 数字，脱离了计算图。它能打印、能累加，但不能再 backward  只在累加统计时才 .item() .item() 的正确用途：把 Tensor 转成 Python 浮点数，用于累加、打印、记录。不是用来替代 Tensor
optimizer 里存着动量（Adam 的 m、v）等状态。每轮重建会丢掉这些状态，Adam 退化成几乎只有一阶的 SGD，训练效果大打折扣  在训练循环外建一次，把同一个 optimizer 传进 train_one_epoch
检查 torch.isfinite(loss)——torch.isfinite 检查是不是 NaN 或 Inf。训练中常见的数值爆炸（学习率太大、梯度爆炸）会产生 NaN 
如果不检查，loss.backward() 会把 NaN 传播到所有参数，模型瞬间废掉，而且后续 loss 一直是 NaN，很难定位是哪一步坏的  在 backward 之前检查，能第一时间抛错，定位问题

"""
def train_one_epoch(
    model: SimpleCLIP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)

        logits, image_features, text_features = model(images, input_ids)
        loss = clip_contrastive_loss(logits)

        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is not finite: {loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_batches, 1)


# 验证配对准确率——"训练完了怎么知道学没学会"
"""
给模型看所有图、所有文，问它："每张图最像哪条文？每条文最像哪张图？" 对了几次？这就是配对准确率
配对准确率要在所有样本之间比较
argmax(dim=1)：消掉列这一维，保留行 → 每行输出一个下标 → "每张图选文本"
argmax(dim=0)：消掉行这一维，保留列 → 每列输出一个下标 → "每条文选图片"
"""
@torch.no_grad()
def evaluate_alignment(
    model: SimpleCLIP,
    dataloader: DataLoader,  # batch_size=len(dataset)：一个 batch 装下所有样本
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    # 约定：dataloader 只有一个 batch，且包含全部样本
    batch = next(iter(dataloader))

    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)

    logits, _, _ = model(images, input_ids)   # [N, N]

    N = logits.shape[0]
    labels = torch.arange(N, device=device)

    # 图找文：每行取最大
    i2t_predictions = logits.argmax(dim=1)    # [N]  argmax 返回的是下标
    # 文找图：每列取最大
    t2i_predictions = logits.argmax(dim=0)    # [N]

    i2t_acc = (i2t_predictions == labels).float().mean().item()
    t2i_acc = (t2i_predictions == labels).float().mean().item()

    return {
        "image_to_text_accuracy": i2t_acc,
        "text_to_image_accuracy": t2i_acc,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/simple_clip")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--print-every", type=int, default=25)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    # 数据
    samples = create_toy_dataset(Path(args.data_dir))
    captions = [s["caption"] for s in samples]
    tokenizer = SimpleTokenizer(captions, max_length=5)
    print(f"vocab_size: {tokenizer.vocab_size}, pad_id: {tokenizer.pad_id}")

    dataset = ImageTextDataset(samples, tokenizer, image_size=64)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False
    )

    # 模型 + 优化器
    model = SimpleCLIP(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embed_dim=args.embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 保存第一层卷积权重副本
    parameter_before = (
        model.image_encoder.features[0].weight.detach().clone()
    )

    # ==== 第一个 batch 探针 ====
    model.train()
    probe_batch = next(iter(train_loader))
    probe_images = probe_batch["image"].to(device)
    probe_input_ids = probe_batch["input_ids"].to(device)

    with torch.no_grad():
        logits, img_f, txt_f = model(probe_images, probe_input_ids)
        loss = clip_contrastive_loss(logits)

    B = probe_images.shape[0]
    L = probe_input_ids.shape[1]
    D = args.embed_dim

    print("\n=== 第一个 batch ===")
    print("images shape:        ", probe_images.shape)
    print("images dtype:        ", probe_images.dtype)
    print("images device:       ", probe_images.device)
    print("input_ids shape:     ", probe_input_ids.shape)
    print("input_ids dtype:     ", probe_input_ids.dtype)
    print("input_ids device:    ", probe_input_ids.device)
    print("image_features shape:", img_f.shape)
    print("text_features shape: ", txt_f.shape)
    print("image feature norms: ", img_f.norm(dim=-1))
    print("text feature norms:  ", txt_f.norm(dim=-1))
    print("logits shape:        ", logits.shape)
    print("labels shape:        ", torch.arange(B, device=device).shape)
    print("loss shape:          ", loss.shape)
    print("loss finite:         ", torch.isfinite(loss).item())


    # 断言（从当前 tensor 推导，不硬编码 12）
    assert probe_images.shape == (B, 3, 64, 64)
    assert probe_images.dtype == torch.float32
    assert probe_input_ids.shape == (B, L)
    assert probe_input_ids.dtype == torch.long
    assert img_f.shape == (B, D)
    assert txt_f.shape == (B, D)
    assert logits.shape == (B, B)
    assert loss.shape == ()
    assert torch.isfinite(loss).item()
    assert torch.allclose(
        img_f.norm(dim=-1),
        torch.ones(B, device=device),
        atol=1e-5,
    )
    assert torch.allclose(
        txt_f.norm(dim=-1),
        torch.ones(B, device=device),
        atol=1e-5,
    )

    # ==== 训练 ====
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        if (epoch + 1) % args.print_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate_alignment(model, eval_loader, device)
            print(
                f"epoch {epoch+1:4d} | loss {train_loss:.4f} | "
                f"i2t {metrics['image_to_text_accuracy']:.4f} | "
                f"t2i {metrics['text_to_image_accuracy']:.4f}"
            )

    # ==== 最终验证 ====
    parameter_after = (
        model.image_encoder.features[0].weight.detach().clone()
    )
    parameter_changed = not torch.equal(parameter_before, parameter_after)

    final_metrics = evaluate_alignment(model, eval_loader, device)

    print("\n=== 最终结果 ===")
    print("first conv weight shape:", tuple(parameter_after.shape))
    print("device:                ", parameter_after.device)
    print("dtype:                 ", parameter_after.dtype)
    print("参数是否变化:          ", parameter_changed)
    print("最终 i2t 准确率:       ", final_metrics["image_to_text_accuracy"])
    print("最终 t2i 准确率:       ", final_metrics["text_to_image_accuracy"])


if __name__ == "__main__":
    main()


