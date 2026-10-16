import torch

from retrieval_metrics import recall_at_k, recall_at_k_by_ids
"""
query	正确答案	排名	R@1	    R@2	    R@3
0	    候选0	    第1名	命中	命中	命中
1	    候选1	    第2名	未命中	命中	命中
2	    候选2	    第3名	未命中	未命中	命中
"""
def test_recall_known_ranks():
    scores = torch.tensor([
        [0.9, 0.6, 0.3],
        [0.8, 0.5, 0.2],
        [0.7, 0.4, 0.1],
    ])

    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    assert abs(recall_at_k(scores, targets, 1) - 1 / 3) < 1e-6
    assert abs(recall_at_k(scores, targets, 2) - 2 / 3) < 1e-6
    assert recall_at_k(scores, targets, 3) == 1.0
    assert recall_at_k(scores, targets, 100) == 1.0

    transposed_recall = recall_at_k(scores.T, targets, 1)
    assert abs(transposed_recall - 1 / 3) < 1e-6


def test_recall_invalid_inputs():
    scores = torch.tensor([
        [0.9, 0.6, 0.3],
        [0.8, 0.5, 0.2],
        [0.7, 0.4, 0.1],
    ])
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    # k=0
    try:
        recall_at_k(scores, targets, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("k=0 should raise ValueError")

    # target 下标越界
    bad_targets = torch.tensor([0, 1, 3], dtype=torch.long)
    try:
        recall_at_k(scores, bad_targets, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range target should raise ValueError")


def test_recall_by_ids_multiple_positives():
    scores = torch.tensor([
        [0.8, 0.9, 0.7, 0.1],
        [0.6, 0.5, 0.4, 0.3],
    ])

    image_ids = ["a", "b"]
    caption_image_ids = ["a", "b", "a", "b"]

    # 图找文
    assert recall_at_k_by_ids(
        scores, image_ids, caption_image_ids, 1
    ) == 0.0

    assert recall_at_k_by_ids(
        scores, image_ids, caption_image_ids, 2
    ) == 1.0

    # 文找图
    assert recall_at_k_by_ids(
        scores.T, caption_image_ids, image_ids, 1
    ) == 0.75

    assert recall_at_k_by_ids(
        scores.T, caption_image_ids, image_ids, 2
    ) == 1.0


if __name__ == "__main__":
    test_recall_known_ranks()
    test_recall_invalid_inputs()
    test_recall_by_ids_multiple_positives()
    print("All retrieval metric tests passed.")