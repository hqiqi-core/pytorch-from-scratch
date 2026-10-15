"""
训练模型 → 保存 checkpoint → 结束进程
                ↓
新进程 → 恢复词表和模型 → 全库评估
保存的到底是什么？ 模型由两部分组成：
模型结构：Python 类定义
模型参数：训练得到的 Tensor
如第一层卷积权重
shape:  [16,3,3,3]
dtype:  float32
device: cuda:0
model.state_dict() 保存各层参数，以及模型中注册的持久 buffer。它不保存 forward() 的 Python 实现，也不保存 tokenizer。
所以加载模型时需要： 
相同的模型结构
+ 正确的模型配置
+ 保存的参数
+ 相同的文本词表
只保存字典、数字、字符串和 Tensor，避免直接保存整个模型对象： # 今天不用这种方式 torch.save(model, path)
checkpoint = {
    "format_version": 1,
    "model_config": {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "embed_dim": model.embed_dim,
    },
    "tokenizer": {
        "word2id": dict(tokenizer.word2id),
        "max_length": tokenizer.max_length,
    },
    "model_state_dict": model.state_dict(),
}
"""
from pathlib import Path

import torch

from simple_clip_train import SimpleCLIP, SimpleTokenizer


def save_checkpoint(
    path: Path,
    model: SimpleCLIP,
    tokenizer: SimpleTokenizer,
) -> None:
    """
    创建保存目录。
    按上述格式组织字典。
    使用 torch.save() 写入。
    不重新训练，不改变模型参数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "embed_dim": model.embed_dim,
        },
        "tokenizer": {
            "word2id": dict(tokenizer.word2id),
            "max_length": tokenizer.max_length,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)



def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[SimpleCLIP, SimpleTokenizer]:
    """
    检查文件存在
    → 加载 checkpoint 到 CPU
    → 检查格式版本
    → 恢复 tokenizer
    → 按配置创建模型
    → 加载模型参数
    → 移到目标 device
    → model.eval()
    → 返回模型与 tokenizer
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 文件不存在: {path}")
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError(
            f"unsupported checkpoint format_version: "
            f"{checkpoint['format_version']}"
        )
    tokenizer_state = checkpoint["tokenizer"]  # 读取字典是这样读取

    tokenizer = SimpleTokenizer(
        captions=[],
        max_length=tokenizer_state["max_length"],
    )

    tokenizer.word2id = dict(tokenizer_state["word2id"])
    tokenizer.id2word = {
        token_id: word
        for word, token_id in tokenizer.word2id.items()
    }
    if tokenizer.vocab_size != checkpoint["model_config"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary size does not match model config")
    
    model = SimpleCLIP(**checkpoint["model_config"])
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer

