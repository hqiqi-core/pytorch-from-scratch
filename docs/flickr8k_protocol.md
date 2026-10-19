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


## Day 19：顺序不变性与候选库规模实验

### 实验设置

- 模型：`openai/clip-vit-base-patch32`
- 数据版本：`56f58c967835f7c508d684f36bd7897cca9d7634`
- 数据范围：validation前100张图片及其500条描述，未打乱选择顺序。
- 图片和文本分别编码一次，归一化后计算余弦相似度。
- 指标定义：每个query的Top-K中至少出现一个标注正例即为命中，再对query求平均。

三组实验：

- A：前20张图片及其100条描述，在小库内评估。
- B：保持A的query不变，将文本候选扩大到500条、图片候选扩大到100张。
- C：使用全部100张图片、500条描述作为双向query和候选。

### 实际结果

| 设置 | 方向 | Query数 | 候选数 | R@1 | R@3 | R@5 |
|---|---|---:|---:|---:|---:|---:|
| A：小库 | i2t | 20 | 100 | 0.900 | 1.000 | 1.000 |
| A：小库 | t2i | 100 | 20 | 0.830 | 1.000 | 1.000 |
| B：固定query、扩大候选 | i2t | 20 | 500 | 0.850 | 0.950 | 0.950 |
| B：固定query、扩大候选 | t2i | 100 | 100 | 0.710 | 0.850 | 0.930 |
| C：完整子集 | i2t | 100 | 500 | 0.870 | 0.980 | 0.980 |
| C：完整子集 | t2i | 500 | 100 | 0.752 | 0.922 | 0.958 |

### 顺序不变性

使用seed=42生成文本排列，同步重排文本特征与其所属image_id。

完整100张子集上的双向R@1、R@3、R@5在重排前后均一致，支持当前评估依据ID关系而非固定位置匹配。

### 候选库扩大的影响

比较A与B，query集合固定：

- i2t R@1从0.90降至0.85，下降5个百分点，即20个query中新增1个首位错误。
- t2i R@1从0.83降至0.71，下降12个百分点，即100个query中新增12个首位错误。

这说明本次实验中，新增候选提高了检索难度。A与C同时改变了query集合和候选库，不能用两者差值单独衡量候选库扩大的影响。

### 排名变化案例

图片query：`validation_000018`。

小库Top-3文本均属于正确图片，分数分别为：

- `validation_000018`：0.2773
- `validation_000018`：0.2662
- `validation_000018`：0.2550

扩大候选库后，Top-3为：

- `validation_000053`：0.2916
- `validation_000044`：0.2908
- `validation_000044`：0.2893

正确图片所属文本的最高分仍为0.2773，但新增的非正例文本得分更高，使该query从R@1命中变为未命中。

这里重复的image_id代表同一图片的不同描述。当前日志未包含具体caption_id和文本，尚不能据此判断语义混淆的具体原因。

### 限制

以上结果来自固定validation前100张图片，是小规模检索实验，不代表完整验证集性能。模型权重revision尚未固定，后续正式对比需补充。

## Day 20：训练采样协议

- 训练数据：`jxie/flickr8k` train 划分前 100 张图片、500 条描述，
  revision `56f58c967835f7c508d684f36bd7897cca9d7634`，
  选择方式 `first_n_without_shuffle`，存放于 `data/flickr8k_train100`。
- Dataset 以图片为单位（长度 100），每次 `__getitem__` 从该图的 5 条描述中随机选 1 条。
- DataLoader 使用 `shuffle=True` 无放回打乱，一个 epoch 内每条图片记录只出现一次，
  因此同一 batch 内不会出现重复图片 ID。
- batch size 为 8：12 个完整 batch + 最后 1 个 batch 为 4。
- 训练标签使用对角线 `arange(B)`：batch 内第 i 张图与第 i 条所选描述配对。
- validation 维持完整多描述评估，正例关系由 `image_id` 决定，不能用 `arange` 表达。
- 该采样只能避免"同一图片的其他描述进入当前 batch 当负例"，
  不能消除不同图片之间语义相似造成的假负例。
- validation 用的 `data/flickr8k_val100` 不参与参数更新。