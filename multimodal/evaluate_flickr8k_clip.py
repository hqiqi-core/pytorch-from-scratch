
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

    i2t, t2i = {}, {}
    for k in args.ks:
        i2t[f"R@{k}"] = recall_at_k_by_ids(
            similarities, image_ids, caption_image_ids, k
        )
        t2i[f"R@{k}"] = recall_at_k_by_ids(
            similarities.T, caption_image_ids, image_ids, k
        )

    print("i2t:", i2t)
    print("t2i:", t2i)

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
        "t2i": t2i,
    }

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


if __name__ == "__main__":
    main()