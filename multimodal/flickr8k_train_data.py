"""Flickr8k training dataset: one image, one randomly chosen caption."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from prepare_flickr8k import write_json  # 可选：如果不想导入，自己写 json.dump
import json


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class Flickr8kTrainDataset(Dataset):
    """以图片为单位。每张图每次随机返回 1 条描述。"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

        image_records = load_json(self.data_root / "train_images.json")
        caption_records = load_json(self.data_root / "train_captions.json")

        # --- 三个必要检查 ---
        assert image_records, "No image records found."
        image_ids = [r["image_id"] for r in image_records]  # r["image_id"] 就是在每个 dict 里取 "image_id" 这个键 → 得到 "train_000000"、"train_000001"……最后 image_ids 是长度为 100 的字符串列表
        assert len(image_ids) == len(set(image_ids)), "Duplicate image_id."

        captions_by_image_id = defaultdict(list)  # defaultdict 在你第一次访问不存在的键时自动创建空 list  一次遍历把 500 条按 image_id 归组，__getitem__ 时 O(1) 拿到该图的 5 条
        """
        record里面是这样：
        {
            "caption_id": "train_000000_caption_2",
            "image_id": "train_000000",
            "text": "a black dog ...",
            "split": "train"
        }
        """
        for record in caption_records:    # record["image_id"] → "train_000000"
            captions_by_image_id[record["image_id"]].append(record)   #  captions_by_image_id["train_000000"].append(record) → 把这条整条描述 dict 放进以 "train_000000" 为键的列表里

        """
        captions_by_image_id = {
            "train_000000": [
                {"caption_id": "train_000000_caption_0", "image_id": "train_000000", "text": "...", "split": "train"},
                {"caption_id": "train_000000_caption_1", "image_id": "train_000000", "text": "...", "split": "train"},
                {"caption_id": "train_000000_caption_2", ...},
                {"caption_id": "train_000000_caption_3", ...},
                {"caption_id": "train_000000_caption_4", ...},
            ],
            "train_000001": [ ... 5 条 ... ],
            // ... 共 100 个键，每键 5 条
        }
        """
        for image_id in image_ids:
            assert captions_by_image_id[image_id], (
                f"No captions for image_id={image_id}"
            )

        # 保留图片记录顺序
        self.image_records = image_records
        self.captions_by_image_id = dict(captions_by_image_id)

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, index: int) -> dict:
        # record 是"图片记录" dict：
        # {"image_id": "train_000042", "image_path": "images/train/train_000042.png", "source_index": 42, "split": "train"}
        record = self.image_records[index]  

        image_id = record["image_id"]  # "train_000042"

        caption_record = random.choice(self.captions_by_image_id[image_id])  # 从五个caption中随便选一个caption作为label

        with Image.open(self.data_root / record["image_path"]) as image: 
            rgb_image = image.convert("RGB")

        return {
            "image": rgb_image,
            "text": caption_record["text"],
            "image_id": image_id,
            "caption_id": caption_record["caption_id"],
        }


def collate_image_text(batch: list[dict]) -> dict:
    return {
        "images":     [item["image"]     for item in batch],  # 8 个 PIL
        "texts":      [item["text"]      for item in batch],  # 8 个 str
        "image_ids":  [item["image_id"]  for item in batch],  # 8 个 str
        "caption_ids":[item["caption_id"] for item in batch], # 8 个 str
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Flickr8kTrainDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_image_text,  # 告诉 DataLoader"凑够一个 batch 后，怎么把 list 里的样本拼成一批"  自定义拼 batch 的函数
    )
    """
    默认的 collate_fn（torch.utils.data.default_collate）会尝试：
    对 dict 的每个键，把 8 个样本的值收集起来
    如果是 Tensor → torch.stack 成 [8, ...]
    如果是数字 → 变成 Tensor
    如果是字符串 / 任意对象 → 通常报错或行为不可控
    问题就在这里：
    image 是 PIL.Image，默认 collate 不知道该拼成什么 Tensor（也没法 stack 不同尺寸的图）
    text 是 str，默认 collate 也不会变 Tensor
    所以直接 DataLoader(dataset, batch_size=8) 会报错（或得到不想要的结果）。
    {
        "images":      [<PIL>, <PIL>, ..., <PIL>],   # 长度 8
        "texts":       ["...", "...", ..., "..."],   # 长度 8
        "image_ids":   ["train_000042", ...],        # 长度 8
        "caption_ids": ["train_000042_caption_3", ...],  # 长度 8
    }
    """

    print(f"Dataset images: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")

    # 从 metadata 建 caption_id -> image_id 映射，用于检查配对
    caption_records = load_json(args.data_dir / "train_captions.json")
    caption_to_image = {r["caption_id"]: r["image_id"] for r in caption_records}

    first_batch = True
    seen_image_ids: list[str] = []

    for batch in loader:
        ids = batch["image_ids"]

        # 1) batch 内图片 ID 不重复
        assert len(ids) == len(set(ids)), "Duplicate image_id within batch."

        # 2) 选中的描述确实属于对应图片
        for image_id, caption_id in zip(ids, batch["caption_ids"]):
            assert caption_to_image[caption_id] == image_id, (
                f"caption {caption_id} does not belong to {image_id}"
            )

        if first_batch:  # 只在第一个 batch 打印两条样例
            print(f"Unique image IDs: {len(set(ids))}")  
            for i in range(min(2, len(ids))):  # 只遍历前 2 个样本
                print("---")
                print(f"image_id: {batch['image_ids'][i]}")
                print(f"caption_id: {batch['caption_ids'][i]}")
                print(f"text: {batch['texts'][i]}")
                print(f"image size: {batch['images'][i].size}")
            first_batch = False

        seen_image_ids.extend(ids)  # 把本 batch 的 ID 追加进总表

    # 3) 一个 epoch 覆盖全部图片，无重复无遗漏
    assert len(seen_image_ids) == len(dataset)
    assert len(set(seen_image_ids)) == len(dataset)  # 去重后仍是 100。若某图出现两次（重复）而另一图没出现，去重后 < 100，失败  set用来去重

    print("Pairing and epoch coverage check passed.")


if __name__ == "__main__":
    main()