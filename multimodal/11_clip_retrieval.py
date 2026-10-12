"""
加载数据库并执行文本检索
"""
import argparse
import json
from pathlib import Path

import torch

from clip_utils import encode_texts, load_clip_model


def parse_args() -> argparse.Namespace:  # argparse 解析完命令行参数后，返回的一个对象，用来装所有参数值  返回 Namespace
    """
        --query          必填
        --top-k          默认 3
        --feature-path   默认 data/clip_demo/index/image_features.pt
        --metadata-path  默认 data/clip_demo/index/image_paths.json
        --model-name     默认 openai/clip-vit-base-patch32
    """
    parser = argparse.ArgumentParser(   # parser 是用 argparse.ArgumentParser(...) 创建的对象
        description="CLIP 图文检索：用文本 query 检索最相似的图片"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="检索用的文本描述，例如 'A red vehicle'"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回最相似的图片数量（默认: 3）",
    )

    parser.add_argument(
        "--feature-path",
        type=str,
        default="data/clip_demo/index/image_features.pt",
        help="图片特征张量文件路径",
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/clip_demo/index/image_paths.json",
        help="图片路径元数据 JSON 文件路径",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="使用的 CLIP 模型名称",
    )
    """
    parser.parse_args() 返回的 Namespace 相当于
    Namespace(
        query="A red vehicle",
        top_k=3,
        feature_path="data/clip_demo/index/image_features.pt",
        metadata_path="data/clip_demo/index/image_paths.json",
        model_name="openai/clip-vit-base-patch32",
    )

    """
    return parser.parse_args()   # 把这个 Namespace 对象作为函数 parse_args() 的返回值交出去

def load_database(
    feature_path: Path,
    metadata_path: Path,
) -> tuple[torch.Tensor, list[str]]:
    """
    加载离线图片特征和对应路径。
    """
    # 1. 特征文件存在
    if not feature_path.is_file():
        raise FileNotFoundError(f"特征文件不存在: {feature_path}")

    # 2. JSON 文件存在
    if not metadata_path.is_file():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    image_features = torch.load(feature_path,map_location="cpu")
    with metadata_path.open("r", encoding="utf-8") as file:
        image_paths = json.load(file)

    if not isinstance(image_features, torch.Tensor):
        raise TypeError(
            f"image_features 必须是 torch.Tensor，实际类型: {type(image_features)}"
        )
    if image_features.ndim != 2:
        raise ValueError(
            f"image_features 必须是二维张量 [N, D]，"
            f"当前维度 {image_features.ndim}，shape={tuple(image_features.shape)}"
        )
    
    if not isinstance(image_paths, list):
        raise TypeError(
            f"image_paths 必须是 list，实际类型: {type(image_paths)}"
        )
    if not all(isinstance(p, str) for p in image_paths):
        raise TypeError("image_paths 中的元素必须都是 str")
    
    if image_features.shape[0] != len(image_paths):
        raise ValueError(
            f"特征数量与路径数量不一致: "
            f"features={image_features.shape[0]}, paths={len(image_paths)}"
        )
    
    if not torch.isfinite(image_features.float()).all():
        bad_mask = ~torch.isfinite(image_features)
        raise ValueError(
            f"image_features 中存在 NaN/Inf，数量={bad_mask.sum().item()}，"
            f"位置(前10)={bad_mask.nonzero()[:10].tolist()}"
        )

    # 7. 数据库不为空
    if image_features.shape[0] == 0:
        raise ValueError("数据库为空：特征数量为 0")


    return image_features,image_paths


def retrieve_topk(
    query_feature: torch.Tensor,
    image_features: torch.Tensor,
    image_paths: list[str],
    top_k: int,
) -> list[tuple[int, float, str]]:
    """
    返回:
        [(index, score, path), ...]
    """
    if query_feature.ndim != 2:
        raise ValueError(
            f"query_feature must be 2D, got {query_feature.shape}"
        )

    if image_features.ndim != 2:
        raise ValueError(
            f"image_features must be 2D, got {image_features.shape}"
        )

    if query_feature.shape[0] != 1:
        raise ValueError(
            "retrieve_topk currently supports exactly one query"
        )

    if query_feature.shape[1] != image_features.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"query={query_feature.shape[1]}, "
            f"images={image_features.shape[1]}"
        )

    if len(image_paths) == 0:
        raise ValueError("image database must not be empty")

    if image_features.shape[0] != len(image_paths):
        raise ValueError("feature count and metadata count do not match")
    
    
    similarities = query_feature @ image_features.T

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    actual_k = min(top_k, len(image_paths))

    scores, indices = torch.topk(
        similarities.squeeze(0),
        k=actual_k,
    )
    
    results = []

    for index_tensor, score_tensor in zip(indices, scores):
        index = index_tensor.item()
        score = score_tensor.item()
        path = image_paths[index]
        results.append((index, score, path))

    return results

def main():
    args = parse_args()
    if not args.query.strip():
        raise ValueError("query 不能为空")

    if not args.query.strip():
        raise ValueError("query must not be empty")

    if args.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    feature_path = Path(args.feature_path)
    metadata_path = Path(args.metadata_path)

    image_features,image_paths=load_database(feature_path,metadata_path)
    image_features=image_features.to(device)   # 要赋值 而不是只移动
    model,processor = load_clip_model(args.model_name,device)
    query_feature = encode_texts(model,processor,[args.query],device)  # 返回是在cpu
    query_feature=query_feature.to(device)
    results = retrieve_topk(
        query_feature,
        image_features,
        image_paths,
        top_k=args.top_k,
    )

    print(f"Query: {args.query}")
    print(f"Device: {device}")
    print(f"Database size: {image_features.shape[0]}")

    for rank, (index, score, path) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Index: {index}")
        print(f"Score: {score:.6f}")
        print(f"Path: {path}")

if __name__ == "__main__":
    main()




