from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


image_dir = Path(    
    "data/clip_demo/images"
)


image_paths = sorted(    # 把生成器里的所有文件路径拿出来，排序后返回一个列表
    image_dir.glob("*.jpg")   # .glob("*.jpg") 会查找该目录下所有以 .jpg 结尾的文件
)   # 得到path list


print("Number of images:", len(image_paths))


images = []


for path in image_paths:  # 得到PIL Image list
    image = Image.open(path)
    image = image.convert("RGB")
    images.append(image)


print("Loaded images:", len(images))

for image, path in zip(images, image_paths):
    print(
        path.name,
        image.size
    )


# 1. Load the pretrained CLIP model.
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)  # 加载预训练模型
processor = CLIPProcessor.from_pretrained(model_name)  
model.eval()   #切换到评估模式，改变 Dropout、BatchNorm 等模块的行为

query = "A beautiful beach."  

# # 3. Preprocess image and text.
# inputs = processor(
#     images=images,    # 图片list
#     text=[query],    # 文本
#     return_tensors="pt",   # 返回 PyTorch 张量
#     padding=True,      # 文本补齐到同一长度 
#     truncation=True,   # 超长文本截断
# )

# print("pixel_values:", inputs["pixel_values"].shape)   #应该是[5,3,224,224]  因为一次性处理了五张图像

# with torch.inference_mode():   # 推理期间关闭梯度追踪，并减少相关的自动微分开销
#     vision_outputs = model.vision_model(   
#         pixel_values=inputs["pixel_values"]
#     )
    
#     text_outputs = model.text_model(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#     )
    
#     visual_pooled = vision_outputs.pooler_output  
#     image_features =  model.visual_projection(visual_pooled) 
#     image_features = F.normalize(image_features,p=2,dim=-1)

#     text_pooled = text_outputs.pooler_output  
#     text_features = model.text_projection(text_pooled)  
#     text_features = F.normalize(text_features,p=2,dim=-1)


# 封装拆成函数  图像和文本分别处理  图像就先处理好 而用户询问实时处理
def encode_images(images, model, processor):
    """
    images:
        list[PIL.Image]

    return:
        image_features [B,512]
    """

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    with torch.inference_mode():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        visual_pooled = vision_outputs.pooler_output

        image_features = model.visual_projection(
            visual_pooled
        )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1
        )

    return image_features



def encode_text(text, model, processor):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.inference_mode():

        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # attention_mask 的作用是告诉模型哪些 token 是"真实内容"，哪些是"填充（padding）"，让模型在计算时忽略掉填充部分。
        )

        text_pooled = text_outputs.pooler_output

        text_features = model.text_projection(
            text_pooled
        )

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1
        )

    return text_features

image_features = encode_images(
    images,
    model,
    processor
)

torch.save(   # 把 image embedding 保存下来
    image_features,
    "data/clip_demo/image_features.pt"
)

features = torch.load(
    "data/clip_demo/image_features.pt"
)
print(features.shape)

text_feature = encode_text(
    query,
    model,
    processor
)

similarity = text_feature @ image_features.T
print(similarity)

values, indices = torch.topk(   # 返回值是一个 namedtuple，包含两个张量  values：前 3 大的数值  indices：这些数值在原张量中的位置索引
    similarity,
    k=3,   # k=3：取前 3 个
    dim=-1   # dim=-1：在最后一个维度上操作（默认也是 -1）
)
print(f"value:{values},indices:{indices}")
print(f"Rank1:{image_paths[indices[0][0]]}")
print(f"Rank2:{image_paths[indices[0][1]]}")
print(f"Rank3:{image_paths[indices[0][2]]}")