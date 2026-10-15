"""
测试一：保存再加载，词表和输出一致。
无需训练，随机初始化模型就能测试序列化是否正确：
创建模型和 tokenizer
→ 固定输入
→ eval 模式下记录输出
→ 保存
→ 加载成一个新的模型
→ 用相同输入计算输出
→ 比较

"""

import tempfile
from pathlib import Path

import torch

from simple_clip_checkpoint import load_checkpoint, load_training_checkpoint, save_checkpoint
from simple_clip_train import SimpleCLIP, SimpleTokenizer, clip_contrastive_loss


def test_save_load_roundtrip() -> None:
    # 1. 固定种子，保证可复现
    torch.manual_seed(42)

    # 2. 创建 tokenizer 和模型（不训练）
    captions = [
        "a red square",
        "a green circle",
        "a blue triangle",
    ]
    tokenizer = SimpleTokenizer(captions, max_length=5)
    model = SimpleCLIP(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embed_dim=32,
    )

    # 3. 都放到 CPU、都设 eval，输出比较更直接
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 4. 固定输入
    input_ids = tokenizer.encode("a red square").unsqueeze(0)  # [1, L]
    images = torch.randn(1, 3, 64, 64)

    # 5. 记录原始输出
    with torch.no_grad():
        original_logits, _, _ = model(images, input_ids)

    # 6. 保存到临时目录
    """
    它创建一个临时目录，with 块结束时自动递归删除。这样测试不会在 data/simple_clip/checkpoints/ 里留下垃圾文件，也不会覆盖你正式训练出来的 checkpoint。任务里说「避免污染正式数据」，就是这个意思。
    """
    with tempfile.TemporaryDirectory() as tmp:   
        ckpt_path = Path(tmp) / "model.pt"
        save_checkpoint(ckpt_path, model, tokenizer)

        # 7. 加载成新模型
        loaded_model, loaded_tokenizer = load_checkpoint(ckpt_path, device)

    # 8. 用相同输入计算加载后的输出
    with torch.no_grad():
        loaded_logits, _, _ = loaded_model(images, input_ids)

    # 9. 三项关键检查
    assert loaded_tokenizer.word2id == tokenizer.word2id

    assert torch.equal(
        loaded_tokenizer.encode("a red square"),
        tokenizer.encode("a red square"),
    )

    """
    为什么比较 logits 而不是 image_features？
    logits 经过 logit_scale 缩放和矩阵乘，涵盖了图像分支、文本分支、温度参数三条路径。只要 state_dict() 里任何一个参数没恢复对，logits 就会不同。用一个量覆盖全部参数，比逐个比 state_dict() 更省事、也更能抓到问题。
    """
    assert torch.allclose(
        original_logits,
        loaded_logits,
        atol=1e-6,
        rtol=1e-5,
    )



"""
测试二：文件不存在时明确报错。
"""
def test_load_missing_checkpoint_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        missing_path = Path(tmp) / "missing.pt"

        try:
            load_checkpoint(missing_path, torch.device("cpu"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(
                "missing checkpoint should raise FileNotFoundError"
            )


# 恢复后的下一次更新与连续训练一致
"""
路径 A（连续训练）：
    模型A + Adam
    → 更新一次
    → 保存 checkpoint
    → A 再更新一次
    → 记为参数 a

路径 B（恢复训练）：
    从 checkpoint 加载成 模型B + 恢复的 Adam
    → 更新一次
    → 记为参数 b

断言：a 和 b 应逐参数 allclose
"""
def test_resume_matches_continuous_training() -> None:
    """
    恢复后的下一次更新，应该与连续训练的下一步更新一致。
    这直接验证 Adam 的 m / v / step 历史被正确恢复。
    """
    torch.manual_seed(42)

    device = torch.device("cpu")

    # 1. 固定 tokenizer 和模型配置
    captions = ["a red square", "a green circle", "a blue triangle"]
    tokenizer = SimpleTokenizer(captions, max_length=5)

    def build_model() -> SimpleCLIP:
        return SimpleCLIP(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            embed_dim=32,
        ).to(device)

    # 2. 固定输入 batch（两条路径必须完全一样）
    input_ids = torch.stack([
        tokenizer.encode("a red square"),
        tokenizer.encode("a green circle"),
        tokenizer.encode("a blue triangle"),
    ]).to(device)                              # [3, L]
    images = torch.randn(3, 3, 64, 64, device=device)

    def one_step(
        model: SimpleCLIP,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """一次完整的 zero_grad → forward → loss → backward → step"""
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(images, input_ids)
        loss = clip_contrastive_loss(logits)
        loss.backward()
        optimizer.step()

    # ===== 路径 A：连续训练两次 =====
    model_a = build_model()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    one_step(model_a, optimizer_a)               # 第一次更新

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "train.pt"

        # 保存训练 checkpoint（带 optimizer 状态、completed_epochs=1）
        save_checkpoint(
            ckpt_path,
            model_a,
            tokenizer,
            optimizer=optimizer_a,
            completed_epochs=1,
        )

        # A 继续更新一次
        one_step(model_a, optimizer_a)

        # ===== 路径 B：从 checkpoint 恢复，再更新一次 =====
        model_b, loaded_tokenizer_b, optimizer_b, start_epoch = (
            load_training_checkpoint(ckpt_path, device)
        )

        # 3. 检查恢复的轮数
        assert start_epoch == 1

        # tokenizer 也应一致
        assert loaded_tokenizer_b.word2id == tokenizer.word2id

        one_step(model_b, optimizer_b)

        # 4. 逐参数比较：恢复后的第一次更新 == 连续训练的第二次更新
        for parameter_a, parameter_b in zip(
            model_a.parameters(),
            model_b.parameters(),
        ):
            assert torch.allclose(
                parameter_a,
                parameter_b,
                atol=1e-6,
                rtol=1e-5,
            ), (
                f"parameter mismatch after resume: "
                f"shape={tuple(parameter_a.shape)}"
            )

def test_inference_checkpoint_cannot_resume_training() -> None:
    """
    不带 optimizer 状态的 checkpoint：
    - load_training_checkpoint 必须抛 ValueError
    - load_checkpoint 仍应正常加载
    """
    torch.manual_seed(0)

    device = torch.device("cpu")

    captions = ["a red square", "a green circle", "a blue triangle"]
    tokenizer = SimpleTokenizer(captions, max_length=5)
    model = SimpleCLIP(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embed_dim=32,
    ).to(device)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "inference.pt"

        # 只用三参数调用 → 不保存 optimizer_state_dict
        save_checkpoint(ckpt_path, model, tokenizer)

        # 1. 完整训练恢复必须报错
        try:
            load_training_checkpoint(ckpt_path, device)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "load_training_checkpoint should raise ValueError "
                "when optimizer state is missing"
            )

        # 2. 推理加载仍应成功
        loaded_model, loaded_tokenizer = load_checkpoint(
            ckpt_path, device
        )
        assert loaded_tokenizer.word2id == tokenizer.word2id

if __name__ == "__main__":
    test_save_load_roundtrip()
    test_load_missing_checkpoint_raises()
    test_resume_matches_continuous_training()
    test_inference_checkpoint_cannot_resume_training()
    print("All checkpoint tests passed.")