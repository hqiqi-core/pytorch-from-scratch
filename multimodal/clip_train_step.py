"""
One CLIP training step: update only the two projection layers.
让真实图文进 CLIP，算出 loss、回传梯度、更新指定参数，只走一个 step
已经从零写过简化 CLIP。今天是在预训练 CLIP 上做轻量微调
图片 → Vision Encoder(冻结) → visual_projection(可训) → normalize → ┐
                                                                      ├→ 图文 logits     logit_scale(冻结)
文本 → Text Encoder(冻结)   → text_projection(可训)   → normalize → ┘
对 openai/clip-vit-base-patch32：
图像 pooled feature [B,768] → visual_projection [512,768] → 图像 embedding [B,512]
文本 pooled feature [B,512] → text_projection [512,512] → 文本 embedding [B,512]

parameter.requires_grad_(False)	不为该参数算梯度
model.eval() / .train()	切换模块的行为（dropout、BN 等）
eval() 不会关闭梯度。所以冻结编码器设为 eval() 模式，完全不妨碍投影层训练。两者是正交的，别混。

今天直接调用 CLIP 模型的 forward，用返回的 logits_per_image
outputs = model(pixel_values=..., input_ids=..., attention_mask=..., return_dict=True)
logits = outputs.logits_per_image
这个输出已经包含投影、归一化和 temperature 缩放。不要再手动 normalize，也不要再乘一次 scale。

"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flickr8k_train_data import (
    Flickr8kTrainDataset,
    collate_image_text,
)
from clip_utils import load_clip_model
from simple_clip_train import (
    clip_contrastive_loss,
    resolve_device,
    set_seed,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/flickr8k_train100"))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def configure_projection_training(model) -> None:
    """冻结全部，再打开两侧投影层；编码器设 eval，模型整体设 train。"""
    # 1) 全部冻结
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # 2) 打开两侧投影层
    for parameter in model.visual_projection.parameters():
        parameter.requires_grad_(True)
    for parameter in model.text_projection.parameters():
        parameter.requires_grad_(True)

    # 3) 模块模式
    model.train()                    # 顶层 train
    model.vision_model.eval()        # 冻结编码器 eval（不影响梯度开关）
    model.text_model.eval()
    """
    为什么 model.train() 之后还要单独 vision_model.eval()？
    因为顶层 train() 会递归把所有子模块设为 train；
    我们想保留冻结编码器的 eval 行为（关 dropout 等），所以再单独设回去。
    投影层是 Linear，train/eval 行为一致，无所谓。
    """


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # --- 加载模型，FP32 ---
    model, processor = load_clip_model(args.model_name, device)
    model = model.float()

    configure_projection_training(model)  # 设置让编码器冻结 

    # --- 打印可训练参数 ---
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    for name, parameter in trainable_parameters:
        print(name, tuple(parameter.shape))
    print(
        "Trainable parameters:",
        sum(parameter.numel() for _, parameter in trainable_parameters),
    )

    # --- DataLoader，取一个 batch ---
    dataset = Flickr8kTrainDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_image_text,
    )
    batch = next(iter(loader))
    assert len(batch["images"]) >= 2, "Need at least 2 samples for contrastive loss."

    # optimizer 要在 trainable_parameters 建好之后、forward 之前创建
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_parameters],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    inputs = processor(
        images=batch["images"],
        text=batch["texts"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
    )
    inputs = inputs.to(device)

    print("pixel_values:", tuple(inputs["pixel_values"].shape),
          inputs["pixel_values"].dtype, inputs["pixel_values"].device)   # L 由本 batch 文本分词结果决定，每批可能不同
    print("input_ids:", tuple(inputs["input_ids"].shape),
          inputs["input_ids"].dtype, inputs["input_ids"].device)
    print("attention_mask:", tuple(inputs["attention_mask"].shape),
          inputs["attention_mask"].dtype, inputs["attention_mask"].device)
    
    visual_before = model.visual_projection.weight.detach().clone()
    text_before = model.text_projection.weight.detach().clone()

    frozen_parameter = next(model.vision_model.parameters())
    frozen_before = frozen_parameter.detach().clone()


    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
    )

    logits = outputs.logits_per_image
    loss = clip_contrastive_loss(logits)

    print("image_embeds:", tuple(outputs.image_embeds.shape))  # (8, 512)
    print("text_embeds:", tuple(outputs.text_embeds.shape))  # (8, 512)
    print("logits:", tuple(logits.shape))   # (8, 8)
    print("loss:", loss.item(), "requires_grad:", loss.requires_grad)

    assert torch.isfinite(loss), "Loss is not finite."
    assert logits.shape[0] == logits.shape[1] == len(batch["images"]), (
        "logits should be [B, B]"
    )

    loss.backward()
    for name, parameter in trainable_parameters:  # 注意：有梯度 ≠ 梯度非零。打印范数就是为了看实际值——如果打印出 0.0，说明该参数这次没被更新，需要想想为什么（但一般不期望是 0）
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        print(f"{name} grad norm:", parameter.grad.norm().item())

    # 冻结编码器不应有梯度
    assert all(
        parameter.grad is None
        for parameter in model.vision_model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.text_model.parameters()
    )
    optimizer.step()

    visual_delta = (
        model.visual_projection.weight.detach() - visual_before
    ).abs().max().item()
    text_delta = (
        model.text_projection.weight.detach() - text_before
    ).abs().max().item()
    frozen_unchanged = torch.equal(
        frozen_before,
        frozen_parameter.detach(),
    )

    print("visual projection max delta:", visual_delta)   # 0.00010000169277191162
    """
    为什么两个 delta 恰好都是 0.00010000...？
    AdamW 第一步更新量约等于 lr × m̂/(√v̂+ε)，在第一次 step 时 m̂/√v̂ 近似为 ±1 量级，所以单步参数变化的最大绝对值约等于学习率 1e-4。两个层都出现 1e-4 是正常的，说明优化器在正常小步更新。
    """
    print("text projection max delta:", text_delta)   # 0.00010000169277191162
    print("frozen parameter unchanged:", frozen_unchanged)

    assert visual_delta > 0, "visual_projection was not updated."
    assert text_delta > 0, "text_projection was not updated."
    assert frozen_unchanged, "A frozen encoder parameter changed!"

    with torch.no_grad():
        outputs_after = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        loss_after = clip_contrastive_loss(outputs_after.logits_per_image)

    print("Loss before update:", loss.item())
    print("Loss after update:", loss_after.item())


if __name__ == "__main__":
    main()



