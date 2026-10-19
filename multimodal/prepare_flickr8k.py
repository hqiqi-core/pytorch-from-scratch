"""Prepare a small Flickr8k retrieval subset."""

from __future__ import annotations  # 让类型注解（比如 list[dict]）在旧版本 Python 上也能正常使用，不会在运行时被求值。这是一个兼容性保险，防止环境 Python 版本偏低时报错。

import argparse  # 用来解析命令行参数。脚本要支持 --output-dir、--limit、--revision，这些都需要 argparse
import json   # 把图片记录、文本记录、来源信息写成 JSON 文件
from collections import Counter    # 关键检查里要数“每张图片有几条描述”。Counter 可以直接统计，不用手写循环
from pathlib import Path   # 所有路径操作用 Path，比字符串拼接更安全、更清晰。比如 output_dir / "images" / "validation"

from datasets import load_dataset  # 从 Hugging Face 读取 jxie/flickr8k
from huggingface_hub import HfApi  # 用来查数据集的真实 commit。因为 main 会变，我们要把它解析成一个固定的 revision，保证可复现


DATASET_ID = "jxie/flickr8k"  # 数据集名字，只在写来源信息和加载时用一次。提成常量，改的时候只改一处
# SOURCE_SPLIT = "validation"  # 今天只读 validation。提成常量，后面路径、ID 前缀、source.json 都用它
CAPTIONS_PER_IMAGE = 5  # Flickr8k 每张图固定 5 条描述。关键检查里会用它断言每张图恰好 5 条

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small Flickr8k validation subset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/flickr8k"),
        help="Directory where images and metadata will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of images to take from the validation split.",
    )
    parser.add_argument(   # 数据集版本，默认 main。我们不会直接用 main 加载，而是先把它解析成具体 commit
        "--revision",
        type=str,
        default="main",
        help="Dataset revision to resolve before loading.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to take images from.",
    )
    return parser.parse_args()  # 解析命令行并返回

def resolve_revision(dataset_id: str, revision: str) -> str:
    """Resolve a mutable revision name to a concrete commit hash.
    传的 --revision main 是一个会变的指针。今天 main 指向 A，明天可能指向 B 
        如果直接用 main 加载数据，将来重建这个子集时可能拿到不同的图片和描述，指标就无法比较
        HfApi().dataset_info() 查一次，把 main 解析成具体 commit hash（info.sha） 
        后续 load_dataset 用这个 commit，而不是 main。这样固定版本、固定数据
    """
    info = HfApi().dataset_info(dataset_id, revision=revision)
    return info.sha

def check_records(
    image_records: list[dict],
    caption_records: list[dict],
) -> None:
    """Validate the image-caption ID relations in the metadata."""
    # 取出所有图片 ID，并转成集合
    image_ids = [record["image_id"] for record in image_records]
    image_id_set = set(image_ids)

    assert len(image_ids) == len(image_id_set), "Duplicate image_id found." # 如果列表长度和集合长度不一致，说明有重复 ID。图片 ID 必须唯一

    # 统计每条描述属于哪张图片。counts 形如 {"validation_000000": 5, "validation_000001": 5, ...}
    counts = Counter(
        record["image_id"] for record in caption_records
    )

    # 所有描述对应的图片 ID 集合，必须和图片记录集合完全一致。这保证 每条描述都能找到对应图片   每张图片都至少有一条描述
    assert set(counts) == image_id_set, (
        "Caption records do not match image records."
    )
    # 每张图片恰好 5 条描述
    assert all(
        count == CAPTIONS_PER_IMAGE for count in counts.values()
    ), "Some images do not have exactly 5 captions."


def write_json(path: Path, payload) -> None:  # 输入是路径和要写的内容。payload 不加类型注解，因为它既可能是 list（图片记录、文本记录），也可能是 dict（source.json）。这是有意保留灵活性
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)  # indent=2：格式化输出，方便人工查看 JSON


def main() -> None:
    """
        从 Hugging Face 的 jxie/flickr8k 读取 validation 划分。
        取前 20 张图片，每张 5 条描述，共 100 条描述。
        保存图片到 data/flickr8k/images/validation/。
        保存 validation_images.json、validation_captions.json、source.json。
        在脚本内检查图文 ID 关系是否完整。
    """
    args = parse_args()

    if args.limit <= 0:  # 要求“limit > 0”。提前拦掉非法值，比后面取 0 张图片再报错更清楚
        raise ValueError(f"--limit must be positive, got {args.limit}")
    source_split = args.split
    output_dir: Path = args.output_dir
    image_dir = output_dir / "images" / source_split
    image_dir.mkdir(parents=True, exist_ok=True)

    revision = resolve_revision(DATASET_ID, args.revision)
    print(f"Dataset: {DATASET_ID}")
    print(f"Revision: {revision}")
    print(f"Split: {source_split}")

    dataset = load_dataset(
        DATASET_ID,
        split=source_split,
        revision=revision,
        streaming=True,  # streaming=True：流式读取，适合只取少量样本
    )

    image_records: list[dict] = []
    caption_records: list[dict] = []

 
    # take(args.limit)：只取前 N 个，不打乱  enumerate：拿到顺序索引 source_index
    for source_index, sample in enumerate(dataset.take(args.limit)):
        image_id = f"{source_split}_{source_index:06d}"  # 本地 ID，形如 validation_000000。用本地 ID，不是原始 Flickr ID。来源靠 source.json 里的版本、划分和 source_index 追踪
        relative_path = f"images/{source_split}/{image_id}.png"  
        absolute_path = output_dir / relative_path  # metadata 里保存相对路径，实际写盘用 output_dir / relative_path。以后读取就是 data_root / record["image_path"]

        image = sample["image"].convert("RGB")  # 统一通道，避免灰度或带 alpha 的图片出问题 
        image.save(absolute_path)  

        image_records.append(   # 每张图只加一条图片记录。注意：不要为每条 caption 重复保存一张图片
            {
                "image_id": image_id,
                "image_path": relative_path,
                "source_index": source_index,
                "split": source_split,
            }
        )

        for caption_index in range(CAPTIONS_PER_IMAGE):
            text = sample[f"caption_{caption_index}"]
            if not isinstance(text, str) or not text.strip():  # 要求“每张图片的 5 条 caption 均为非空字符串”。这里直接检查。注意不是 except: continue 静默跳过，而是明确报错，保留原始信息。
                raise ValueError(
                    f"Empty caption for image_id={image_id}, "
                    f"caption_index={caption_index}"
                )
            caption_records.append(  
                {
                    "caption_id": f"{image_id}_caption_{caption_index}",
                    "image_id": image_id,
                    "text": text.strip(),
                    "split": source_split,
                }
            )

    if len(image_records) != args.limit:
        raise RuntimeError(
            f"Expected {args.limit} images, got {len(image_records)}"
        )

    check_records(image_records, caption_records)
    print("Image-caption mapping check passed.")

    write_json(output_dir / f"{source_split}_images.json", image_records)
    write_json(output_dir / f"{source_split}_captions.json", caption_records)

    source_info = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "source_split": source_split,
        "selection": "first_n_without_shuffle",
        "num_images": len(image_records),
        "num_captions": len(caption_records),
    }
    write_json(output_dir / "source.json", source_info)

    print(f"Number of images: {len(image_records)}")
    print(f"Number of captions: {len(caption_records)}")
    print(
        "Captions per image: "
        f"{len(caption_records) // len(image_records)}"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
