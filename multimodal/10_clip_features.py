
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. Load the pretrained CLIP model.
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)  # 加载预训练模型
processor = CLIPProcessor.from_pretrained(model_name)  
model.eval()   #切换到评估模式，改变 Dropout、BatchNorm 等模块的行为

# 2. Load one real image from your gallery.
image_path = Path("data/clip_demo/images/dog.jpg")  

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

with Image.open(image_path) as image:
    image = image.convert("RGB")     # 把图片转换/统一成 RGB 三通道模式

query = "A dog running in a grassy field."  

# 3. Preprocess image and text.
inputs = processor(
    images=image,    # 图片（PIL Image 或 list）
    text=[query],    # 文本，注意是「列表」，即使只有一条
    return_tensors="pt",   # 返回 PyTorch 张量
    padding=True,      # 文本补齐到同一长度 
    truncation=True,   # 超长文本截断
)
# 这两个 key 是 processor 内部的 image_processor 和 tokenizer 自动生成的
"""
processor 内部其实包含一个 image_processor（图像处理器）。当你传 images=image 时，它会自动做一串预处理
# 伪代码：processor 内部大致做了这些
image = image.resize((224, 224))          # 1. 缩放
image = np.array(image) / 255.0           # 2. 归一化到 0~1
image = (image - mean) / std              # 3. 标准化 (ImageNet mean/std)
image = image.transpose(2, 0, 1)          # 4. HWC → CHW
tensor = torch.tensor(image).unsqueeze(0) # 5. 加 batch 维
最后它把这个结果存进字典，键名就叫 pixel_values（这是 HuggingFace 约定的固定名字）

pixel_values 是 image_processor 把 PIL 图预处理成 NCHW 张量 的结果
"""
print("pixel_values:", inputs["pixel_values"].shape)    # torch.Size([1, 3, 224, 224])   形状格式是 [N, C, H, W]，即 PyTorch 惯例的 NCHW

"""
processor 内部还包含一个 tokenizer（分词器）。当你传 text=[query] 时
query = "A dog running in a grassy field."
# tokenizer 做的事情：
# 1. 分词 → ["A", "dog", "running", "in", "a", "grassy", "field", "."]
# 2. 转成 id → [49406, 1929, 3236, 530, 320, 11758, 1691, 269, 49407, ...]
# 3. 加 special token（如 CLIP 的 <|startoftext|> <|endoftext|>）
# 4. padding 到固定长度
结果存进字典，键名固定叫 input_ids
" A dog running in a grassy field. " 经过 CLIP tokenizer 分词后恰好是 10 个 token，包括：
<|startoftext|>（1个）
正文 token
<|endoftext|>（1个）
padding token（如果 padding=True 且设了最大长度）

input_ids 是 tokenizer 把文本转成 token id 序列 的结果
"""
print("input_ids:", inputs["input_ids"].shape)   # torch.Size([1, 10])  10是序列长度，即 token 个数  这个query经过 CLIP tokenizer 分词后恰好是 10 个 token

# 4. Extract pooled features and project into the shared space.  
with torch.inference_mode():  # 推理期间关闭梯度追踪，并减少相关的自动微分开销
    vision_outputs = model.vision_model(   
        pixel_values=inputs["pixel_values"]
    )
    
    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    

    # TODO 1: Extract pooled visual features.  怎样获取视觉编码器的池化特征？	查看 vision_outputs 的属性   编码器的输出最后就是经过一个池化层的输出
    visual_pooled = vision_outputs.pooler_output  

    # TODO 2: Project visual features.  怎样将 768 维视觉特征映射到 512 维？	查看 model.visual_projection 
    image_features =  model.visual_projection(visual_pooled) 

    # TODO 3: Extract and project text features.  怎样获取并投影文本特征？	对照图像分支理解
    text_pooled = text_outputs.pooler_output  
    text_features = model.text_projection(text_pooled)   # 为什么文本是512 还要投影——共同空间的维度相同，不代表它们已经处于同一个语义空间
    # 虽然形状没有变化，但投影层仍然会对特征进行一次可学习的线性变换。它不是单纯地把 512 维变成另一个维度，而是学习如何将文本编码器的表示映射到 CLIP 训练时使用的共同嵌入空间

    # TODO 4: L2-normalize the projected features.  怎样将特征向量归一化？	使用 F.normalize，注意 dim
    image_features = F.normalize(image_features,p=2,dim=-1)
    text_features = F.normalize(text_features,p=2,dim=-1)

print("visual pooled:", visual_pooled.shape)
print("image features:", image_features.shape)
print("text pooled:", text_pooled.shape)
print("text features:", text_features.shape)
print("image feature norm:", image_features.norm(dim=-1))
print("text feature norm:", text_features.norm(dim=-1))


print("visual_projection:", model.visual_projection)
print("text_projection:", model.text_projection)

print("visual projection weight:",
      model.visual_projection.weight.shape)
print("text projection weight:",
      model.text_projection.weight.shape)
