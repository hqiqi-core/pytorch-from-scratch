"""
recall_at_k_by_ids() 是按 ID 匹配的，不是按位置匹配。所以如果你把文本特征和对应的 caption_image_ids 用同一个排列同步打乱，那么"每条文本的正例图片 ID"没变，只是顺序变了。
数学上，打乱行列只是重排矩阵，每个 query 的候选集合完全一样，所以 R@K 必须逐位相等（在无并列边界的情况下）

"""
from pathlib import Path
from typing import Sequence
import torch
import json
from PIL import Image
import torch.nn.functional as F
import argparse
from clip_utils import load_clip_model

from simple_clip_train import resolve_device
from retrieval_metrics import recall_at_k_by_ids

def load_metadata(data_root: Path):
    image_records = json.loads(
        (data_root / "validation_images.json").read_text(
            encoding="utf-8"
        )
    )

    caption_records = json.loads(
        (data_root / "validation_captions.json").read_text(
            encoding="utf-8"
        )
    )

    image_ids = [
        record["image_id"]
        for record in image_records
    ]

    caption_image_ids = [
        record["image_id"]
        for record in caption_records
    ]
    return image_records, caption_records, image_ids, caption_image_ids


def encode_images(model, processor, images: Sequence[Image.Image], device):
    if len(images) == 0:
        raise ValueError("images must not be empty")

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )
        pooled = vision_outputs.pooler_output          # [B, hidden]
        feats = model.visual_projection(pooled)        # [B, D]

        feats = F.normalize(feats, p=2, dim=-1)

        if not torch.isfinite(feats).all():
            raise ValueError("image_features contains NaN or Inf")
        if feats.ndim != 2:
            raise ValueError(f"image_features must be 2D, got {feats.shape}")

    return feats.cpu()


def encode_texts(model, processor, texts: Sequence[str], device):
    if len(texts) == 0:
        raise ValueError("texts must not be empty")

    inputs = processor(
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled = text_outputs.pooler_output            # [B, hidden]
        feats = model.text_projection(pooled)          # [B, D]

        feats = F.normalize(feats, p=2, dim=-1)

        if not torch.isfinite(feats).all():
            raise ValueError("text_features contains NaN or Inf")
        if feats.ndim != 2:
            raise ValueError(f"text_features must be 2D, got {feats.shape}")

    return feats.cpu()


def extract_features(
    model, processor, image_records, caption_records,
    data_root, device, batch_size,
):
    # ---- 图片：切 batch，逐 batch 编码，拼顺序 ----
    image_features_list = []
    for start in range(0, len(image_records), batch_size):
        batch = image_records[start:start + batch_size]

        images = []
        for r in batch:
            with Image.open(data_root / r["image_path"]) as im:
                images.append(im.convert("RGB"))

        image_features_list.append(
            encode_images(model, processor, images, device)
        )

    image_features = torch.cat(image_features_list, dim=0)

    # ---- 文本：切 batch，逐 batch 编码，拼顺序 ----
    texts = [r["text"] for r in caption_records]
    text_features_list = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        text_features_list.append(
            encode_texts(model, processor, batch_texts, device)
        )

    text_features = torch.cat(text_features_list, dim=0)

    return image_features, text_features



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/flickr8k")
    )
    parser.add_argument(
        "--model-name", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/flickr8k/clip_baseline.json"),
    )
    parser.add_argument(
        "--check-order-invariance",
        action="store_true",
    )
    parser.add_argument(
        "--compare-first-images",
        type=int,
        default=None,
    )

    return parser.parse_args()


def find_t2i_r1_failure(similarities, caption_records, caption_image_ids, image_ids):
    """返回第一个 t2i R@1 失败的 (index, record, correct_id, top1_id, topk)"""
    topk = similarities.topk(3, dim=1)

    top1_indices = topk.indices[:, 0].tolist()
    top3_indices = topk.indices.tolist()
    top3_scores = topk.values.tolist()

    for i, record in enumerate(caption_records):
        correct_id = caption_image_ids[i]
        top1_id = image_ids[top1_indices[i]]
        if top1_id != correct_id:
            return {
                "index": i,
                "caption": record["text"],
                "correct_id": correct_id,
                "top1_id": top1_id,
                "top3": [
                    (image_ids[j], top3_scores[i][r])
                    for r, j in enumerate(top3_indices[i])
                ],
            }
    return None

def evaluate_both_directions(scores, image_query_ids, caption_query_ids, ks):
    """返回 (i2t, t2i) 两个 {R@k: value} 字典。

    scores: [num_image_queries, num_caption_queries]
    image_query_ids:    score 行对应的图片 ID
    caption_query_ids:  score 列对应的文本所属图片 ID
    """
    i2t, t2i = {}, {}
    for k in ks:
        i2t[f"R@{k}"] = recall_at_k_by_ids(
            scores, image_query_ids, caption_query_ids, k
        )
        t2i[f"R@{k}"] = recall_at_k_by_ids(
            scores.T, caption_query_ids, image_query_ids, k
        )
    return i2t, t2i

def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, processor = load_clip_model(args.model_name, device)

    image_records, caption_records, image_ids, caption_image_ids = load_metadata(
        args.data_dir
    )

    image_features, text_features = extract_features(
        model, processor, image_records, caption_records,
        args.data_dir, device, args.batch_size,
    )

    # ---- 打印一次特征信息 ----
    print("Image feature shape:", tuple(image_features.shape))
    print("Image feature device:", image_features.device)
    print("Image feature dtype:", image_features.dtype)
    print(
        "Image norm min/max:",
        image_features.norm(dim=-1).min().item(),
        image_features.norm(dim=-1).max().item(),
    )
    print("Text feature shape:", tuple(text_features.shape))
    print("Text feature device:", text_features.device)
    print("Text feature dtype:", text_features.dtype)
    print(
        "Text norm min/max:",
        text_features.norm(dim=-1).min().item(),
        text_features.norm(dim=-1).max().item(),
    )

    # ---- 相似度与双向指标 ----
    similarities = image_features @ text_features.T   # [num_images, num_captions]

    
    # ---- 顺序不变性检查（先只做 i2t）----
    if args.check_order_invariance:
        # 造一个打乱顺序
        generator = torch.Generator().manual_seed(42)  # 固定随机种子
        permutation = torch.randperm(   # torch.randperm(100)：生成 0..99 的一个随机排列   结果 permutation 是个索引表：它说"新第 0 个位置放旧第 57 个，新第 1 个放旧第 3 个……"
            len(caption_image_ids),    # 它不是打乱"值"，而是打乱"下标"
            generator=generator,
        )

        shuffled_text_features = text_features[permutation]
        shuffled_caption_image_ids = [
            caption_image_ids[index]
            for index in permutation.tolist()
        ]

        shuffled_similarities = (
            image_features @ shuffled_text_features.T
        )

        print("Order invariance check:")
        for k in args.ks:
            # ---- i2t ----
            i2t_original = recall_at_k_by_ids(
                similarities, image_ids, caption_image_ids, k
            )
            i2t_shuffled = recall_at_k_by_ids(
                shuffled_similarities, image_ids,
                shuffled_caption_image_ids, k,
            )
            # ---- t2i ----
            t2i_original = recall_at_k_by_ids(
                similarities.T, caption_image_ids, image_ids, k
            )
            t2i_shuffled = recall_at_k_by_ids(
                shuffled_similarities.T, shuffled_caption_image_ids,
                image_ids, k,
            )

            print(
                f"R@{k} i2t original={i2t_original:.6f} "
                f"shuffled={i2t_shuffled:.6f} | "
                f"t2i original={t2i_original:.6f} "
                f"shuffled={t2i_shuffled:.6f}"
            )
            assert abs(i2t_original - i2t_shuffled) < 1e-8, (
                f"i2t order invariance failed at R@{k}"
            )
            assert abs(t2i_original - t2i_shuffled) < 1e-8, (
                f"t2i order invariance failed at R@{k}"
            )

    i2t,t2i = evaluate_both_directions(similarities, image_ids, caption_image_ids, args.ks)

    print("i2t:", i2t)
    print("t2i:", t2i)
    comparison = None
    if args.compare_first_images is not None:
        n = args.compare_first_images
        if n <= 0 or n > len(image_ids):
            raise ValueError(
                f"--compare-first-images must be in (0, {len(image_ids)}], got {n}"
            )

        # 前 n 张图，以及它们对应的所有文本（按 ID 找，不假设是前 n*5 条）
        small_image_ids = image_ids[:n]
        small_image_id_set = set(small_image_ids)
        small_caption_indices = [   # 文本列表里的位置下标（整数）
            index
            for index, image_id in enumerate(caption_image_ids)
            if image_id in small_image_id_set
        ]
        """
        caption_image_ids 是什么——是"每条文本的正例图片 ID 列表"——文本 j 的正确答案就是 caption_image_ids[j] 这张图
        第 j 个元素 = "第 j 条文本属于哪张图"
        caption_image_ids =
        [ "validation_000000",  # 第 0 条文本属于图 000000
        "validation_000000",  # 第 1 条文本也属于图 000000（同一张图的第2条描述）
        ...
        "validation_000000",  # 第 4 条
        "validation_000001",  # 第 5 条属于图 000001
        ...
        ]

        """
        small_caption_image_ids = [  # small_caption_image_ids 不是"要检索的图片列表"，而是"这 100 条文本各自的正例是哪张图" 
            caption_image_ids[index]
            for index in small_caption_indices
        ]

        # A：小库
        small_scores = similarities[:n, small_caption_indices]   # [n, n*5]
        small_i2t, small_t2i = evaluate_both_directions(
            small_scores, small_image_ids, small_caption_image_ids, args.ks
        )

        # B：固定 query，扩大候选库
        i2t_expanded_scores = similarities[:n, :]                # [n, all_captions]
        i2t_exp_i2t = {}
        for k in args.ks:
            i2t_exp_i2t[f"R@{k}"] = recall_at_k_by_ids(
                i2t_expanded_scores, small_image_ids, caption_image_ids, k
            )
        # t2i：固定文本 query，候选放到全部图片。单独算，不用函数里的 .T
        t2i_expanded_scores = similarities[:, small_caption_indices].T  # [n*5, all_images]
        t2i_exp_t2i = {}
        for k in args.ks:
            t2i_exp_t2i[f"R@{k}"] = recall_at_k_by_ids(
                t2i_expanded_scores, small_caption_image_ids, image_ids, k
            )

        # C：完整子集（就是刚才已经算过的 i2t/t2i）

        comparison = {
            "compare_first_images": n,
            "small": {
                "num_image_queries": len(small_image_ids),
                "num_caption_queries": len(small_caption_image_ids),
                "i2t": small_i2t,
                "t2i": small_t2i,
            },
            "expanded_candidates_fixed_queries": {
                "i2t_num_image_queries": len(small_image_ids),
                "i2t_num_caption_candidates": len(caption_image_ids),
                "i2t": i2t_exp_i2t,
                "t2i_num_caption_queries": len(small_caption_image_ids),
                "t2i_num_image_candidates": len(image_ids),
                "t2i": t2i_exp_t2i,
            },
            "full": {
                "num_image_queries": len(image_ids),
                "num_caption_queries": len(caption_image_ids),
                "i2t": i2t,
                "t2i": t2i,
            },
        }

        print("\nComparison (A=small, B=expanded candidates, C=full):")
        for name, block in [
            ("A small", comparison["small"]),
            ("B expanded", comparison["expanded_candidates_fixed_queries"]),
            ("C full", comparison["full"]),
        ]:
            print(f"{name} i2t={block['i2t']}")
            print(f"{name} t2i={block['t2i']}")

    # ---- 保存 JSON ----
    dataset_revision = "unknown"
    source_path = args.data_dir / "source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        dataset_revision = source.get("revision", "unknown")

    result = {
        "model_name": args.model_name,
        "dataset_revision": dataset_revision,
        "split": "validation",
        "selection": "first_n_without_shuffle",
        "num_images": len(image_records),
        "num_captions": len(caption_records),
        "metric_definition": "at_least_one_annotated_positive_in_top_k",
        "i2t": i2t,
        "t2i": t2i
    }
    if comparison is not None:
        result["comparison"] = comparison

    

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved to {args.output_path}")

    # ---- 失败案例 ----
    failure = find_t2i_r1_failure(
        similarities.T, caption_records, caption_image_ids, image_ids
    )
    if failure is None:
        print("该子集未发现 t2i R@1 失败。")
    else:
        print("\n--- t2i R@1 失败案例 ---")
        print("caption:", failure["caption"])
        print("correct image_id:", failure["correct_id"])
        print("top-1 image_id:", failure["top1_id"])
        print("top-3:")
        for img_id, score in failure["top3"]:
            print(f"  {img_id}  {score:.4f}")

    # ---- 扩大候选库导致 i2t R@1 失败的案例（仅 --compare-first-images 时）----
    if args.compare_first_images is not None:
        # small_scores / i2t_expanded_scores / n 在此块内已定义
        small_top1 = small_scores.argmax(dim=1)
        expanded_top1 = i2t_expanded_scores.argmax(dim=1)

        for i in range(n):
            small_col = small_top1[i].item()
            small_hit = small_caption_image_ids[small_col] == small_image_ids[i]
            expanded_col = expanded_top1[i].item()
            expanded_hit = caption_image_ids[expanded_col] == small_image_ids[i]

            if small_hit and not expanded_hit:
                print("\n--- i2t R@1 案例：小库对、扩大后错 ---")
                print("query image_id:", small_image_ids[i])
                print("小库 Top-3:")
                small_top3 = small_scores[i].topk(3)
                for score, col in zip(
                    small_top3.values.tolist(),
                    small_top3.indices.tolist(),
                ):
                    print(f"  {small_caption_image_ids[col]}  {score:.4f}")
                print("扩大库 Top-3:")
                exp_top3 = i2t_expanded_scores[i].topk(3)
                for score, col in zip(
                    exp_top3.values.tolist(),
                    exp_top3.indices.tolist(),
                ):
                    print(f"  {caption_image_ids[col]}  {score:.4f}")
                break

if __name__ == "__main__":
    main()