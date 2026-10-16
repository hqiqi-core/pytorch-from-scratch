"""
理解 Recall@K
当前有9对图文，每张图片只有一条正确文本，每条文本也只有一张正确图片。
假设某张图片检索文本，排名是：
第1名：错误文本
第2名：错误文本
第3名：正确文本
那么这个 query：
R@1：未命中
R@3：命中
R@5：命中
在今天“一条 query 只有一个正确答案”的设定下：
R@K=正确答案位于前K名的query数量/query总数量
例如9条 query 中，6条的正确答案位于前3：
R@3=6/9=0.6667
注意这个定义的适用范围：有多个正确答案时，还需要明确指标采用“至少命中一个”还是“召回全部相关项中的比例”。今天只实现单正例版本。
图找文：similarities.topk(k, dim=1).indices
文找图：similarities.T.topk(k, dim=1).indices
统一转成“每行是一个 query、每列是一个候选”，后面就可以共用同一个指标函数。
scores:       [Q,C]   Q：query数量  C：候选数量
topk_indices: [Q,K]
targets:      [Q]  targets[i]：第 i 个 query 的正确候选下标。
targets[:,None]:
              [Q,1]
比较结果:      [Q,K]
每行是否命中:  [Q]
最终Recall:    标量


"""

import torch


def recall_at_k(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """
    scores:  [Q,C]，浮点数，值越大越相关
    targets: [Q]，torch.long，与scores在同一设备
    k:       正整数
    需要检查的边界仅包括：
    - scores 是二维浮点 Tensor，且 query、候选数量均大于0。
    - scores 不包含 NaN/Inf。
    - targets 为 [Q] 的 long Tensor，与 scores 在同一设备。
    - target下标在 [0,C)。
    - k > 0。
    非法输入抛出清楚的 ValueError 即可。
    """
    # ---- 边界检查 ----
    if not isinstance(scores, torch.Tensor):
        raise ValueError("scores 必须是 torch.Tensor")
    if scores.dim() != 2:
        raise ValueError("scores 必须是二维 Tensor [Q, C]")
    if not torch.is_floating_point(scores):
        raise ValueError("scores 必须是浮点类型")
    Q, C = scores.shape
    if Q <= 0 or C <= 0:
        raise ValueError("scores 的 query 和候选数量必须大于 0")
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise ValueError("scores 不能包含 NaN 或 Inf")

    if not isinstance(targets, torch.Tensor):
        raise ValueError("targets 必须是 torch.Tensor")
    if targets.dim() != 1:
        raise ValueError("targets 必须是一维 Tensor [Q]")
    if targets.dtype != torch.long:
        raise ValueError("targets 必须是 torch.long 类型")
    if targets.shape[0] != Q:
        raise ValueError("targets 的长度必须等于 scores 的 query 数量")
    if targets.device != scores.device:
        raise ValueError("targets 必须和 scores 在同一个设备上")
    if (targets < 0).any() or (targets >= C).any():
        raise ValueError("targets 的下标必须在 [0, C) 范围内")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数")
    actual_k = min(k, C)
    topk_indices = scores.topk(  # 要看这k个候选有没有包含targets[i]   [Q,K]
        actual_k,
        dim=1,
    ).indices     # .indices 只取下标
    targets = targets.unsqueeze(1) # 从[Q]变成[Q,1]列向量，每行一个结果 就可以直接做广播比较

    matches = topk_indices == targets # [Q,K] 比较每个位置是否等于正确答案  每个位置 matches[i][j] 表示：第 i 个 query 的第 j 名候选，是不是正确答案？
    hits = matches.any(dim=1)  # [Q]  每个 query 是否命中   matches 是 [Q, K]，每一行表示这个 query 的前 K 名里，每个位置是不是正确答案    hits 是布尔 tensor [Q]
    # .any(dim=1) 表示：这一行里只要有一个 True，这个 query 就算命中  dim 不是“按哪个维度保留”，而是“沿着哪个维度进行压缩/归约”。
    """
    dim=0 表示沿着行方向（上下）压缩，结果少掉第 0 维。
    dim=1 表示沿着列方向（左右）压缩，结果少掉第 1 维。
    换句话说：
    dim 指定的是“要被消掉的那个维度”，不是“要保留的维度”。
            K=0    K=1    K=2
    Q=0   False  False   True
    Q=1    True  False  False
    第 0 维是 Q（行，每个 query）
    第 1 维是 K（列，前 K 名）
    对每个 query（每一行），看这一行的 K 个位置里有没有 True。
    也就是：把 K 这一维消掉，保留 Q 这一维。
    Q=0: any(False, False, True) = True
    Q=1: any(True,  False, False) = True
    """

    return hits.float().mean().item()   # .float() 把 True/False 转成 1.0/0.0  .mean() 求平均，就是命中比例  .item() 把标量 tensor 转成 Python 的 float


# 根据ID实现多正例指标
def recall_at_k_by_ids(
    scores: torch.Tensor,
    query_image_ids: list[str],
    candidate_image_ids: list[str],   # 文本虽然有独立的 caption_id，这里传入的仍然是它所属图片的 image_id  候选ID允许重复，因为一张图片有多条描述
    k: int,
) -> float:
    if not isinstance(scores, torch.Tensor):
        raise ValueError("scores 必须是 torch.Tensor")
    if scores.dim() != 2:
        raise ValueError("scores 必须是二维 Tensor [Q, C]")
    if scores.numel() == 0:
        raise ValueError("scores 不能为空")
    if not torch.is_floating_point(scores):
        raise ValueError("scores 必须是浮点类型")
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise ValueError("scores 不能包含 NaN 或 Inf")
    if k <= 0:
        raise ValueError("k 必须大于 0")

    if len(query_image_ids) != scores.shape[0]:
        raise ValueError(
            f"query_image_ids 长度 {len(query_image_ids)} 与 scores.shape[0]={Q} 不一致"
        )
    if len(candidate_image_ids) != scores.shape[1]:
        raise ValueError(
            f"candidate_image_ids 长度 {len(candidate_image_ids)} 与 scores.shape[1]={C} 不一致"
        )

    candidate_id_set = set(candidate_image_ids)

    if any(
        image_id not in candidate_id_set
        for image_id in query_image_ids
    ):
        raise ValueError("some queries have no positive candidate")
    
    actual_k = min(k, scores.shape[1])

    topk_indices = scores.topk(
        actual_k,
        dim=1,
    ).indices.cpu().tolist()

    hits = []

    for query_index, candidate_indices in enumerate(topk_indices):
        query_id = query_image_ids[query_index]  

        hit = any(
            candidate_image_ids[index] == query_id
            for index in candidate_indices
        )

        hits.append(hit)
    

    return sum(hits) / len(hits)