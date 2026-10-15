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

from simple_clip_checkpoint import load_checkpoint, save_checkpoint
from simple_clip_train import SimpleCLIP, SimpleTokenizer


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

if __name__ == "__main__":
    test_save_load_roundtrip()
    test_load_missing_checkpoint_raises()
    print("All checkpoint tests passed.")