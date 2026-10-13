# multimodal/test_12_simple_clip.py
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from simple_clip_train import (  # 按你的实际模块名改
    create_toy_dataset,
    SimpleTokenizer,
    ImageTextDataset,
    SimpleCLIP,
    clip_contrastive_loss,
    train_one_epoch,
    evaluate_alignment,
    set_seed,
    resolve_device,
)

def test_tokenizer():
    captions = ["a red square", "a green circle", "a blue triangle"]
    tok = SimpleTokenizer(captions, max_length=5)

    # PAD 固定为 0
    assert tok.pad_id == 0
    assert tok.PAD_ID == 0
    assert tok.UNK_ID == 1

    # 正常编码
    ids = tok.encode("a red square")
    assert ids.shape == (5,)
    assert ids.dtype == torch.long
    assert ids.tolist() == [tok.word2id["a"], tok.word2id["red"],
                            tok.word2id["square"], 0, 0]

    # 未知词 -> UNK
    ids_unk = tok.encode("a pink square")
    assert ids_unk[1].item() == 1   # pink -> UNK

    # 长文本截断
    ids_long = tok.encode("a red square green circle triangle blue")
    assert ids_long.shape == (5,)
    assert (ids_long == 0).sum().item() == 0   # 被截满，没有 PAD

    # 短文本补齐
    ids_short = tok.encode("a")
    assert ids_short.shape == (5,)
    assert ids_short[1:].tolist() == [0, 0, 0, 0]

    # 空字符串 -> 全 PAD
    ids_empty = tok.encode("")
    assert ids_empty.tolist() == [0, 0, 0, 0, 0]

    print("test_tokenizer passed")

def test_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        samples = create_toy_dataset(Path(tmp), image_size=64)
        captions = [s["caption"] for s in samples]
        tokenizer = SimpleTokenizer(captions, max_length=5)
        dataset = ImageTextDataset(samples, tokenizer, image_size=64)

        assert len(dataset) == len(samples)

        item = dataset[0]

        # 图片
        assert item["image"].shape == (3, 64, 64)
        assert item["image"].dtype == torch.float32
        assert item["image"].min().item() >= 0.0
        assert item["image"].max().item() <= 1.0

        # 文本
        assert item["input_ids"].shape == (5,)
        assert item["input_ids"].dtype == torch.long

        # 原始信息
        assert isinstance(item["caption"], str)
        assert isinstance(item["image_path"], str)

    print("test_dataset passed")

def test_model_forward():
    set_seed(42)
    vocab_size = 9
    pad_id = 0
    B = 4

    model = SimpleCLIP(vocab_size=vocab_size, pad_id=pad_id, embed_dim=32)

    images = torch.randn(B, 3, 64, 64)
    # 关键：每行至少一个非 PAD token，避免随机全 padding
    input_ids = torch.randint(1, vocab_size, (B, 5), dtype=torch.long)

    image_features = model.encode_image(images)
    text_features = model.encode_text(input_ids)
    logits, img_f2, txt_f2 = model(images, input_ids)

    assert image_features.shape == (B, 32)
    assert text_features.shape == (B, 32)
    assert logits.shape == (B, B)
    assert img_f2.shape == (B, 32)
    assert txt_f2.shape == (B, 32)

    # 归一化：模长接近 1
    assert torch.allclose(
        image_features.norm(dim=-1),
        torch.ones(B),
        atol=1e-5,
    )
    assert torch.allclose(
        text_features.norm(dim=-1),
        torch.ones(B),
        atol=1e-5,
    )

    # 全部 finite
    assert torch.isfinite(image_features).all()
    assert torch.isfinite(text_features).all()
    assert torch.isfinite(logits).all()

    print("test_model_forward passed")

def test_loss():
    # 正常
    logits = torch.randn(4, 4)
    loss = clip_contrastive_loss(logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # B=1 应报错
    try:
        clip_contrastive_loss(torch.randn(1, 1))
        raise AssertionError("B=1 应抛 ValueError")
    except ValueError:
        pass

    # 非方阵应报错
    try:
        clip_contrastive_loss(torch.randn(2, 3))
        raise AssertionError("非方阵应抛 ValueError")
    except ValueError:
        pass

    # 1 维应报错
    try:
        clip_contrastive_loss(torch.randn(4))
        raise AssertionError("1 维应抛 ValueError")
    except ValueError:
        pass

    print("test_loss passed")

def test_gradients():
    set_seed(42)
    model = SimpleCLIP(vocab_size=9, pad_id=0, embed_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = torch.randn(4, 3, 64, 64)
    input_ids = torch.randint(1, 9, (4, 5), dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = model(images, input_ids)
    loss = clip_contrastive_loss(logits)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all(), f"{name} 梯度非有限"

    print("test_gradients passed")


def test_parameter_update():
    set_seed(42)
    model = SimpleCLIP(vocab_size=9, pad_id=0, embed_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    images = torch.randn(4, 3, 64, 64)
    input_ids = torch.randint(1, 9, (4, 5), dtype=torch.long)

    # 拿第一层卷积权重做对比
    target = model.image_encoder.features[0].weight

    # ❌ 错误写法：before = target
    #    这只是引用，step 后 before 也变了

    # ✅ 正确写法：clone 一份副本
    before = target.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = model(images, input_ids)
    loss = clip_contrastive_loss(logits)
    loss.backward()
    optimizer.step()

    after = target.detach().clone()

    assert not torch.equal(before, after), "参数没有更新"

    print("test_parameter_update passed")

def test_overfit():
    set_seed(42)
    device = resolve_device("auto")

    with tempfile.TemporaryDirectory() as tmp:
        samples = create_toy_dataset(Path(tmp), image_size=64)
        captions = [s["caption"] for s in samples]
        tokenizer = SimpleTokenizer(captions, max_length=5)
        dataset = ImageTextDataset(samples, tokenizer, image_size=64)

        train_loader = DataLoader(dataset, batch_size=len(dataset),
                                  shuffle=True)
        eval_loader = DataLoader(dataset, batch_size=len(dataset),
                                 shuffle=False)

        model = SimpleCLIP(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            embed_dim=32,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        initial_loss = None
        for epoch in range(300):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            if epoch == 0:
                initial_loss = train_loss

        metrics = evaluate_alignment(model, eval_loader, device)

    assert initial_loss is not None
    assert train_loss < initial_loss, \
        f"loss 没下降: {initial_loss:.4f} -> {train_loss:.4f}"
    assert metrics["image_to_text_accuracy"] >= 0.9, \
        f"i2t 太低: {metrics['image_to_text_accuracy']:.4f}"
    assert metrics["text_to_image_accuracy"] >= 0.9, \
        f"t2i 太低: {metrics['text_to_image_accuracy']:.4f}"

    print(f"test_overfit passed: loss {initial_loss:.4f} -> {train_loss:.4f}, "
          f"i2t={metrics['image_to_text_accuracy']:.2f}, "
          f"t2i={metrics['text_to_image_accuracy']:.2f}")

if __name__ == "__main__":
    test_tokenizer()
    test_dataset()
    test_model_forward()
    test_loss()
    test_gradients()
    test_parameter_update()
    test_overfit()
    print("\nAll tests passed.")