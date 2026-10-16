# Flickr8k Retrieval Subset Protocol

## 1. 数据来源

- 数据集：`jxie/flickr8k`
- 来源：Hugging Face 社区整理版本
- 实际 revision：`56f58c967835f7c508d684f36bd7897cca9d7634`
- 原始划分：train / validation / test
- 今天使用：validation 的前 20 张图片，共 100 条描述
- 选择方式：`first_n_without_shuffle`，固定版本、固定划分、固定顺序

## 2. 本地 ID 与路径

- 本地图片 ID：`validation_000000`、`validation_000001`、……
- 图片相对路径：`images/validation/validation_000000.png`
- `source_index`：该图片在原始 validation 流中的顺序索引
- metadata 中的 `image_path` 相对于 `--output-dir`

## 3. 正例关系

- 图找文：20 个 query，每个 query 有 5 条标注正例
- 文找图：100 个 query，每个 query 有 1 张标注正例
- 相似度矩阵 shape：`[20, 100]`
- 正例关系由 `image_id` 决定，不能用 `arange(20)` 表达全部关系

## 4. 评估口径

下一天采用 R@K，定义为：

> 对每个 query，只要 Top-K 中出现至少一个标注正例，就记为命中，再对 query 求平均。

这也可以明确称为 Hit@K。它不同于“找回了全部相关项中的多少比例”。

例如某张图的 5 条正确描述中，Top-3 只找回 1 条：

- 至少命中一个的 R@3：1
- 相关项召回比例：1/5

## 5. 子集用途与限制

- 该子集仅用于调试流程，不能与完整验证集结果直接比较。
- 数据集的标注正例不是所有可能语义相关项的穷举；不同图片之间也可能存在语义相似的描述。
- 保留镜像提供的 train / validation / test 划分。一张图片和它的 5 条描述必须在同一个划分。
- 今天只读取 validation，不建立新的随机划分。
- 保留来源划分不等于已经完成重复图片审计，后续正式实验再检查这一点。


---

## 实际运行结果（Day 18）

### 运行命令

```bash
CUDA_VISIBLE_DEVICES=1 python multimodal/evaluate_flickr8k_clip.py \
  --data-dir data/flickr8k \
  --model-name openai/clip-vit-base-patch32 \
  --batch-size 8 \
  --device cuda \
  --ks 1 3 5 \
  --output-path data/flickr8k/clip_baseline.json
```

### 环境与数据

- model_name: `openai/clip-vit-base-patch32`
- dataset_id: `jxie/flickr8k`
- dataset_revision: `56f58c967835f7c508d684f36bd7897cca9d7634`
- split: validation
- selection: first_n_without_shuffle
- num_images: 20
- num_captions: 100
- metric_definition: at_least_one_annotated_positive_in_top_k

### 特征检查

- Image feature shape: (20, 512), device: cpu, dtype: torch.float32
- Image norm min/max: 0.9999999 / 1.0000001
- Text feature shape: (100, 512), device: cpu, dtype: torch.float32
- Text norm min/max: 0.9999998 / 1.0000001

范数接近 1，说明 L2 归一化生效。

### 指标

| 方向 | R@1 | R@3 | R@5 |
|---|---|---|---|
| i2t | 0.90 | 1.00 | 1.00 |
| t2i | 0.83 | 1.00 | 1.00 |

来源：`data/flickr8k/clip_baseline.json`（该文件位于被 git 忽略的 `data/` 下，不提交）

### 案例分析：t2i R@1 失败

- caption: `the boy laying face down on a skateboard is being pushed along the ground by another boy .`
- 标注 image_id: `validation_000000`
- Top-3 image_id 与相似度:
  - `validation_000008`  0.3140
  - `validation_000000`  0.3125
  - `validation_000015`  0.2623

观察正确图片 `validation_000000` 与 top-1 图片 `validation_000008` 后：

- 混淆类型：<主体 / 动作 / 场景 / 细节，选一个或组合>
- top-1 是否也存在合理语义相关性：<是 / 否，说明理由>
- 补充：top-1 与正确图片的相似度差仅 0.0015，属于极接近的边界情况；R@3=1.0 说明正确答案仍在 top-3 内。

### 备注

- 本次运行仅记录模型名称，未固定模型 revision。同一模型名可能对应不同权重版本，正式对比实验前需固定 revision。
- 数据与生成报告 `data/flickr8k/clip_baseline.json` 留在被忽略的 `data/` 中，仓库只保留协议文档。


