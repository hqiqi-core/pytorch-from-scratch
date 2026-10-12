"""
离线提取全部图片特征并建立数据库
"""

import json
from pathlib import Path
from PIL import Image
import torch
from clip_utils import encode_images,load_clip_model

MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_DIR = Path("data/clip_demo/images")
INDEX_DIR = Path("data/clip_demo/index")

FEATURE_PATH = INDEX_DIR / "image_features.pt"
METADATA_PATH = INDEX_DIR / "image_paths.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def find_image_paths(image_dir: Path) -> list[Path]:
    """
    查找图片目录中的图片，并按路径稳定排序。
    规则：
      - 只保留文件（排除目录/符号链接指向的目录等）
      - 后缀转小写后判断是否在支持列表中
      - 结果按路径排序
    """
    # 1. 目录必须存在且是目录，否则抛出明确异常
    if not image_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在或不是目录: {image_dir}")
    
     # 2. 遍历目录，过滤出文件 + 后缀（小写）匹配
    image_paths = sorted(
        p
        for p in image_dir.rglob("*")  # glob("*.jpg") 在 Linux 上大小写敏感，匹配不到 A.JPG  用 iterdir() 遍历 只列当前目录，不递归；如果需要递归子目录，把 iterdir() 换成 rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS  
    )
    # 3. 没有图片时抛出明确异常
    if not image_paths:
        raise FileNotFoundError(
            f"目录中没有找到图片（支持后缀: {sorted(IMAGE_EXTS)}）: {image_dir}"
        )

    return image_paths

def load_images(image_paths: list[Path]) -> list:
    """
    加载图片，并统一转换成 RGB。
    """
    images = []
    for path in image_paths:  # 得到PIL Image list
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            images.append(rgb_image)
    return images


def main():
    """
    选择 device
        ↓
    查找图片
        ↓
    加载图片
        ↓
    加载 CLIP
        ↓
    encode_images()
        ↓
    保存 image_features.pt
        ↓
    保存 image_paths.json
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = find_image_paths(IMAGE_DIR)
    images = load_images(image_paths)
    model,processor = load_clip_model(MODEL_NAME,device)
    image_features = encode_images(model,processor,images,device)
    # 为什么保存特征前要转移到cpu？
    # 将 embedding 转到 CPU 后再保存，可以让保存文件不依赖原来的 GPU 设备。后续即使在没有 CUDA 的机器上也更容易加载，同时避免把仅用于持久化的数据长期占在显存中。加载后如果需要 GPU 计算，再调用 .to(device)。


    print("######################")
    print(f"Device:{device}")
    print(f"Model device:{model.device}")
    print(f"Number of images:{image_features.shape[0]}")
    print(f"Image feature shape:{image_features.shape}")
    print(f"Image feature device:{image_features.device}")
    norms = image_features.norm(p=2, dim=-1)   # 对最后一维求 L2 范数，得到 [N]
    print(f"Feature norms:{norms}")

    # 1. 必须是二维张量 [N, D]
    if image_features.ndim != 2:
        raise ValueError(
            f"image_features 必须是二维张量 [N, D]，当前维度: {image_features.ndim}，"
            f"shape: {tuple(image_features.shape)}"
        )

    # 2. 特征数量必须和路径数量一致
    if image_features.shape[0] != len(image_paths):
        raise ValueError(
            f"特征数量与路径数量不一致: "
            f"image_features.shape[0]={image_features.shape[0]}, "
            f"len(image_paths)={len(image_paths)}"
        )

    # 3. 不能有 NaN / Inf
    if not torch.isfinite(image_features).all():
        bad_mask = ~torch.isfinite(image_features)
        print(f"异常值数量: {bad_mask.sum().item()}")
        print(f"异常值位置(前10个): {bad_mask.nonzero()[:10].tolist()}")
        raise ValueError("image_features 中存在 NaN 或 Inf")

    # 把处理好的特征和对应的元数据（这里是文件路径列表）分别保存到磁盘
    torch.save(image_features, FEATURE_PATH)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [str(path) for path in image_paths],  # 把 Path 对象列表转成字符串列表
            file,
            ensure_ascii=False,
            indent=2,   # 每个层级缩进 2 个空格，输出「美化过的」多行 JSON
        )
    print(f"Saved features: {FEATURE_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")

   
if __name__ == "__main__":
    main()

    


    




