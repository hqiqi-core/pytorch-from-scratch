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