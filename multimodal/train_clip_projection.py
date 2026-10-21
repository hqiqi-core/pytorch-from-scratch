from clip_train_step import configure_projection_training
from flickr8k_train_data import Flickr8kTrainDataset, collate_image_text
from clip_utils import load_clip_model
from evaluate_flickr8k_clip import evaluate_both_directions, extract_features, load_metadata
from simple_clip_train import clip_contrastive_loss, resolve_device, set_seed
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import json


def train_one_epoch(model, processor, dataloader, optimizer, device) -> float:
    # 验证函数会调用 model.eval()，恢复训练时要重新设置模块模式
    """
    shape和梯度流
    PIL list B → pixel_values [B,3,224,224]
    str  list B → input_ids [B,L] + attention_mask [B,L]
    冻结 encoder → vision pooled [B,768] / text pooled [B,512]
    可训练 projection → 两侧 embedding [B,512]
    logits [B,B] → loss []
    梯度：loss → 归一化/相似度 → 两侧 projection → 只更新这两层
    """
    model.train()
    model.vision_model.eval()
    model.text_model.eval()

    total_loss = 0.0
    total_samples = 0
    steps = 0

    for batch in dataloader:
        inputs = processor(
            images=batch["images"],
            text=batch["texts"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.text_config.max_position_embeddings,
        ).to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        loss = clip_contrastive_loss(outputs.logits_per_image)

        if not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["images"])  # 按样本数加权平均。最后一批只有 4 个样本，如果直接对 batch loss 取算术平均，会让它权重偏大。加权后每个样本权重相等。
        total_samples += len(batch["images"])  
        steps += 1

    print(f"steps={steps}")  # 可选：确认每轮 13
    return total_loss / total_samples

def evaluate_validation(model, processor, data_root, device, batch_size, ks) -> dict:
    model.eval()

    image_records, caption_records, image_ids, caption_image_ids = load_metadata(data_root)

    image_features, text_features = extract_features(
        model, processor, image_records, caption_records, data_root, device, batch_size
    )

    similarities = image_features @ text_features.T    # [100, 500]
    i2t, t2i = evaluate_both_directions(
        similarities, image_ids, caption_image_ids, ks
    )
    return {"i2t": i2t, "t2i": t2i}


def load_source_json(data_root: Path) -> dict | None:
    path = data_root / "source.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(model, args, train_losses, before_metrics, after_metrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 权重
    torch.save(
        {
            "format": "clip_projection_experiment_v1",
            "base_model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "completed_epochs": args.epochs,
        },
        output_dir / "final_model.pt",
    )

    # 2) 报告   是这个模型对应的 配置对象（CLIPConfig 实例）里面存着模型结构、文本最大长度、logit_scale 初始值等 它同时可能带一个字段 _commit_hash，记录"这份权重是从哪个 commit 拉的"
    model_revision = getattr(model.config, "_commit_hash", None)  

    results = {
        "model_name": args.model_name,
        "model_revision": model_revision,          # 读不到就是 None -> JSON 里是 null
        "train_source": load_source_json(args.train_dir),
        "val_source": load_source_json(args.val_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_scope": "projection_only",
        "train_losses": train_losses,
        "before": before_metrics,
        "after": after_metrics,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved weights and results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short CLIP projection training with before/after validation."
    )
    parser.add_argument(
        "--train-dir", type=Path, default=Path("data/flickr8k_train100")
    )
    parser.add_argument(
        "--val-dir", type=Path, default=Path("data/flickr8k_val100")
    )
    parser.add_argument(
        "--model-name", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/clip_projection_day22"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    model, processor = load_clip_model(args.model_name, device)
    model = model.float()

    configure_projection_training(model)  # 冻结模型 只打开投影层训练

    # train Dataset / DataLoader
    train_dataset = Flickr8kTrainDataset(args.train_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_image_text,
    )

    # optimizer 只传可训练参数，训练前创建一次
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=0.0
    )

    print("=== Before ===")
    before_metrics = evaluate_validation(
        model, processor, args.val_dir, device, args.batch_size, ks=(1, 3, 5)
    )
    print(before_metrics)

    train_losses = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, processor, train_loader, optimizer, device)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.6f}")

    print("=== After ===")
    after_metrics = evaluate_validation(
        model, processor, args.val_dir, device, args.batch_size, ks=(1, 3, 5)
    )
    print(after_metrics)

    # 保存
    save_results(model, args, train_losses, before_metrics, after_metrics, Path(args.output_dir))


if __name__ == "__main__":
    main()