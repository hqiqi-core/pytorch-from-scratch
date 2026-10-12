"""
CLIP 加载、图片编码、文本编码
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def load_clip_model(
    model_name: str,
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:
    """
    加载 CLIP processor 和 model。

    Returns:
        model: 已移动到 device，并处于 eval 模式
        processor: CLIP 输入处理器
    """
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model,processor


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """
    将一批 PIL 图片编码为归一化后的 CLIP 特征。

    Returns:
        CPU Tensor，形状为 [B, D]
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")

    inputs = processor(
        images=images,
        return_tensors="pt"   # 返回tensor张量
    ).to(device)

    with torch.inference_mode():
        vision_outputs=model.vision_model(
                pixel_values = inputs["pixel_values"]
        )

        vision_pooled = vision_outputs.pooler_output
        images_features = model.visual_projection(vision_pooled)
        images_features = F.normalize(images_features,p=2,dim=-1)
        # print(f"images_features shape:{images_features.shape}")
        # 第一种 # 检查 NaN
        # has_nan = torch.isnan(images_features).any().item()
        # # 检查 Inf
        # has_inf = torch.isinf(images_features).any().item()

        # print(f"has NaN: {has_nan}")
        # print(f"has Inf: {has_inf}")
        # 第二种
        if not torch.isfinite(images_features).all():
            raise ValueError("image_features contains NaN or Inf")  # 要抛出异常而不是只是打印
        if images_features.ndim != 2:
            raise ValueError(
                f"image_features must be 2D, got {images_features.shape}"
            )

        
    return images_features.cpu()


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """
    将一批文本编码为归一化后的 CLIP 特征。

    Returns:
        CPU Tensor，形状为 [B, D]
    """
    if len(texts) == 0:
        raise ValueError("texts must not be empty")

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty strings")
    
    inputs = processor(
        text=texts,   # 已经是列表
        return_tensors = "pt",
        padding = True,
        truncation=True
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_pooled = text_outputs.pooler_output
        text_features = model.text_projection(text_pooled)
        text_features = F.normalize(text_features,p=2,dim=-1)
        # print(f"texts_features shape:{text_features.shape}")
        if not torch.isfinite(text_features).all():
                    raise ValueError("text_features contains NaN or Inf")  # 要抛出异常而不是只是打印

    return text_features.cpu()
    
    


