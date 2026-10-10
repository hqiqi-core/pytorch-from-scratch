"""
开始理解图像和文本如何映射到同一个向量空间，让计算机能够根据文字找到相关图片或文档。
1. 理解 CLIP 双编码器——图像编码器和文本编码器分别提取特征，再把它们映射到同一个语义空间。
2. 理解对比学习和 InfoNCE——让匹配的图文向量更接近，让不匹配的图文向量相对更远。
3. 写出一个最小可运行的图文对比学习模型——自己实现向量归一化、相似度矩阵、对比损失和反向传播。
4. 连接到你的主项目——弄清楚训练时的对比学习，与部署时使用 FAISS 做向量检索有什么区别。
一、为什么需要 CLIP？
假设你有一个医学图像或文档知识库，里面有三张图片： 
图片 A：眼底图像
对应描述：A color fundus photograph showing retinal vessels.
图片 B：OCT 图像
对应描述：An OCT scan showing retinal layers.
图片 C：研究图表
对应描述：A bar chart summarizing clinical study results.

现在用户输入：Find an OCT image showing retinal layers.
传统图像分类模型可能只能告诉你图片属于什么类别；CLIP 这种图文双编码器，则能把用户的文字和候选图片变成可比较的向量，按语义相似度检索图片。
关键思想是：不直接比较图片像素和文字字符，而是比较它们编码后得到的向量。
先建立一个重要区分：
    CLIP：一种学习图像与文本共同表示的模型和训练方法。
    图文检索：使用共同表示空间，根据文本向量查找最相关图片。
    FAISS：用于高效搜索大量向量的检索工具，本身不是图文编码模型。

二、CLIP 的模型结构
CLIP 全称是 Contrastive Language–Image Pre-training
图像输入->Image Encoder(通常是ViT或者Resnet)->Image Projection + L2 Normalize(得到图像向量)
文本输入->Text Encoder(Transformer)->Text Projection + L2 Normalize(得到文本向量)
计算图文相似度->得到 [B, B] 相似度矩阵，并计算对比损失  【图像编码器和文本编码器的内部结构可以不同，但最终必须把输出投影到同一个维度的语义空间，才能直接计算相似度】

假设一个 batch 中有 4 张图片及其对应的 4 条描述
数据                    形状                 含义     
图像输入           [4, 3, 224, 224]      4 张 RGB 图片
文本 Token IDs        [4, 32]         每条文本最多 32 个 token
图像向量              [4, 512]        每张图片一个 512 维向量
文本向量              [4, 512]        每条文本一个 512 维向量

为什么需要 L2 归一化？——归一化后，向量的 L2 范数为 1。两个归一化向量的点积就等于余弦相似度  这使我们更关注两个向量的方向是否相似，而不是单纯比较向量长度
对应pytorch
image_embeds = torch.nn.functional.normalize(
    image_embeds, p=2, dim=-1
)
text_embeds = torch.nn.functional.normalize(
    text_embeds, p=2, dim=-1
)  dim=-1 表示对最后一个维度归一化   p=2 表示使用 L2 范数（也叫欧几里得范数）来做归一化

三、理解对比学习的核心
假设一个 batch 里有 3 组配对数据：
图片 0 ↔ 文本 0
图片 1 ↔ 文本 1
图片 2 ↔ 文本 2
把图片向量堆成矩阵 I，把文本向量堆成矩阵 T，通过矩阵乘法计算所有图片与所有文本之间的相似度
I∈RB×D,T∈RB×D  S=IT⊤​/τ  I*T的转置除以τ   其中τ：温度参数（temperature） B 是 batch size，D 是共同嵌入维度
归一化后，原始点积通常在 [−1,1] 之间。除以温度参数会改变 logits 的尺度 
较小的温度会放大相似度之间的差距，使 Softmax 分布更尖锐；较大的温度则会使分布更平缓。真正的 CLIP 实现常使用可学习的 logit scale。
尖锐 = 概率集中在少数几个类别上（更关注最难负样本，但可能不稳定、梯度饱和）   平缓 = 概率分散在很多类别上，接近均匀分布 （更均匀考虑所有负样本，但区分信号弱）
CLIP 的对比损失希望：
    正样本对，也就是匹配的图文，相似度尽量高；
    负样本对，也就是不匹配的图文，相似度尽量低。
温度控制 Softmax 的“尖锐程度”：
    温度小 → 分布尖锐 → 模型更关注那些很难区分的负样本；
    温度大 → 分布平缓 → 所有负样本都被较均匀地考虑，学习信号更平滑。
但温度不能太小，也不能太大：
    太小：Softmax 接近 one-hot，梯度容易消失，训练不稳定；
    太大：正负样本区分不明显，对比学习效果弱。
训练目标是让每行的正确文本得分更高，也让每列的正确图片得分更高

用交叉熵实现对比学习
对于第 i 张图片，它的正确文本索引是 i。所以标签为y=[0,1,2,…,B−1]
labels = torch.arange(B, device=logits.device)
loss = torch.nn.functional.cross_entropy(logits, labels)  
为什么可以直接用交叉熵？因为 cross_entropy 会将一行 logits 转化成类别概率，并惩罚正确类别的概率过低。这里的“类别”不是固定的语义类别，而是当前 batch 中哪条文本与这张图片匹配
仅做图像到文本的方向还不够。CLIP 通常使用双向对比损失 LI→T​=CE(S,y) LT→I​=CE(S⊤,y)  L=（LI→T​+LT→I​​）/2  第一项学习“给定图片找文本”，第二项学习“给定文本找图片”

请特别注意：如果 batch size 是 1，矩阵就是 [1,1]，只有一个候选文本，没有其他负样本。Softmax 只能得到概率 1，交叉熵也会是 0。这样的 batch 无法提供有效的对比信号。
这也是为什么对比学习需要多个配对样本，并且训练时要关注 batch 的构成

四、训练与检索不是一回事
训练阶段——学习一个好的向量空间
输入成批的图文配对，构造 [B, B] 相似度矩阵，使用双向对比损失更新编码器参数。
检索阶段——用查询找到知识库中的候选证据
提前计算并保存知识库向量。用户输入查询后编码一次，使用向量搜索取出 Top-K 候选。

例如，你的知识库有 100 万条文档或图片。
训练时，一个 batch 可能只有 64 组图文，因此相似度矩阵是 [64,64]。但检索时，不会为了每个查询重新训练，也不必把所有知识库内容和查询重新组成训练 batch。
假设用户查询编码成：q∈RD  知识库向量矩阵为：E∈RN×D  其中 N 是知识库条数。检索时，可以计算：s=qE⊤∈RN  
再选出得分最高的 K 条记录。这就是最基本的向量检索逻辑。实际规模很大时，通常使用 FAISS 等工具加速搜索。


CLIP 相似度高，不代表模型一定能回答问题。 它适合召回视觉或文本证据，但涉及复杂文档理解、表格推理、跨页证据组合时，仍可能需要重排序器和 VLM


第 1 题：为什么 CLIP 要使用两个编码器，而不是直接拼接图像和文本？
CLIP 使用两个编码器，是因为它希望分别处理两种不同模态的输入：
图像编码器负责把图像转换为视觉特征。
文本编码器负责把文本转换为语言特征。
投影层把两种特征映射到共同的嵌入空间。
这样，图库中的每张图片和用户查询都可以独立编码。图库向量提前计算好后，用户每次查询只需编码文本，不需要把查询与所有图片重新拼接、逐对运行一个模型。
你原回答里“直接拼接就没办法学习对应关系”并不完全准确。直接拼接也可以设计成学习图文对应关系的模型，例如融合式跨模态模型；只是这种架构通常需要把图文一起输入模型，计算成本更高，不适合直接对百万级图库进行高效的独立向量检索。
你应该记住的区别是：
双编码器（CLIP 类）            融合式模型（Cross-Encoder 类）
图像和文本分别编码               图像与文本在模型内部交互
可以预先计算图库向量          通常需要对每个候选图文对进行计算
适合大规模初步召回               适合对少量候选精细重排序

第 5 题：为什么 batch 内负样本和 FAISS Top-K 不是同一个概念？
你的回答：基本正确，还需要补上训练与检索的目的差异。
训练时的 batch 内负样本： 其他图文对作为当前样本的对比对象，帮助模型学习如何区分匹配与不匹配的样本。
检索时的 Top-K： 从真实知识库中找出与查询最相似的 K 个候选，用于后续证据筛选和推理。
两者还有一个重要差异：训练中的负样本不一定是真正的错误答案。假如一张眼底图片配有两条都正确的描述，按照简单的对角线标签设计，另一条描述仍可能被当作负样本。这就是对比学习中的**假负样本（false negative）**问题。
而检索时的 Top-K 也不等于“已确认的正确答案”。它们只是按当前向量相似度排名的候选，还可能需要 Reranker 或 VLM 进一步判断。

路线：
真实图片 + 真实文本
        ↓
两个编码器分别提取特征
        ↓
投影到共同维度 + L2 归一化
        ↓
计算 [B, B] 图文相似度
        ↓
双向对比损失更新编码器
        ↓
训练完成后，离线建立知识库向量
        ↓
用户查询编码 → 向量检索 Top-K
        ↓
Reranker 精排 → VLM 证据推理

"""

import torch
import torch.nn.functional as F


def contrastive_loss(image_embeds, text_embeds, temperature=0.07):  # 对比学习损失
    """
    image_embeds: [B, D]
    text_embeds:  [B, D]
    返回：
        loss: 标量
        logits_per_image: [B, B]   图像对文本的相似度
        logits_per_text:  [B, B]   文本对图像的相似度
    """
    # 1. 检查输入  图像和文本经过对应编码器后必须维度一致 
    assert image_embeds.ndim == 2
    assert text_embeds.ndim == 2
    assert image_embeds.shape == text_embeds.shape
    assert temperature > 0

    B, D = image_embeds.shape  # B是batchsize D是向量维度

    # 2. L2 归一化，每个样本得到单位向量
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)   # 选择L2范数  dim=-1对最后一维特征进行归一化
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    print(f"after L2 norm:image:{image_embeds},text:{text_embeds}")

    # 3. 所有图像与所有文本的相似度
    # [B, D] @ [D, B] -> [B, B]
    logits_per_image = image_embeds @ text_embeds.T  
    logits_per_image = logits_per_image / temperature   # 温度缩放

    # 4. 反方向：文本找图像
    logits_per_text = logits_per_image.T

    # 5. 正确匹配的索引是对角线 0, 1, ..., B-1    生成一个 [0, 1, 2, ..., B-1] 的张量，表示每一行图像（或文本）对应的正确匹配索引
    labels = torch.arange(B, device=image_embeds.device)    # 构造正确匹配的索引 也就是告诉交叉熵损失函数：每一行里，哪一个位置才是正样本 

    # 6. 双向交叉熵
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2

    return loss, logits_per_image, logits_per_text


def retrieve_topk(query, database, k=5):
    """
    query:    [D]
    database: [N, D]
    返回：Top-K 索引和对应相似度
    """
    query = F.normalize(query, dim=-1)
    database = F.normalize(database, dim=-1)

    scores = database @ query
    k = min(k, database.shape[0])
    return torch.topk(scores, k=k)


def main():
    torch.manual_seed(42)  # 随机种子

    B = 8
    D = 16

    # 用随机向量模拟两个编码器的输出
    image_embeds = torch.randn(B, D, requires_grad=True)
    text_embeds = torch.randn(B, D, requires_grad=True)

    loss, logits_i2t, logits_t2i = contrastive_loss(
        image_embeds,
        text_embeds,
        temperature=0.07,
    )

    print("Image embeddings:", image_embeds.shape)
    print("Text embeddings:", text_embeds.shape)
    print("Image-to-text logits:", logits_i2t.shape)
    print("Text-to-image logits:", logits_t2i.shape)
    print("Loss:", loss.item())

    # 7. 反向传播，检查梯度
    loss.backward()

    print("Image grad exists:", image_embeds.grad is not None)
    print("Text grad exists:", text_embeds.grad is not None)
    print("Image grad finite:", torch.isfinite(image_embeds.grad).all().item())
    print("Text grad finite:", torch.isfinite(text_embeds.grad).all().item())

    assert logits_i2t.shape == (B, B)
    assert logits_t2i.shape == (B, B)
    assert image_embeds.grad is not None
    assert text_embeds.grad is not None
    assert torch.isfinite(image_embeds.grad).all()
    assert torch.isfinite(text_embeds.grad).all()

    print("Day 9 smoke test passed.")

    # 构造一个有明确正确答案的小型检索库
    torch.manual_seed(42)

    database = F.normalize(torch.randn(10, 16), dim=-1)

    # 用数据库中的第 3 条向量作为查询
    query = database[3].clone()

    values, indices = retrieve_topk(query, database, k=3)

    print("Top-K indices:", indices)
    print("Top-K scores:", values)


if __name__ == "__main__":
    main()
