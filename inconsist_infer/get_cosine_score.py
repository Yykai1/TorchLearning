# -*- coding:utf-8 -*-
# @Time    : 2022/5/12 19:38
# @Author  : Yinkai Yang
# @FileName: get_cosine_score.py
# @Software: PyCharm
# @Description: this is a program related to
# 模型用时创建
# sent_index, K, mc_K 从外部传入
# 打印分数结果
from similarities import Similarity
import get_axiom_set as get_a

model1 = Similarity(model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# sent_index, _, K, mc_K = get_a.get_axioms_dictionary('AUTOMCS/')
# print('data loaded')


#  method_1
def agg_all(Ki, alpha, sent_bert_scores, sent_index):
    res = 0
    for beta in Ki:
        res += sent_bert_scores[sent_index[beta]][sent_index[alpha]]
    return res / len(Ki)


def mc_all_io(alpha, mc_K, sent_bert_scores, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += agg_all(Ki, alpha, sent_bert_scores, sent_index)
        else:
            res -= agg_all(Ki, alpha, sent_bert_scores, sent_index)
    return res


def score_all_io(Ki, mc_K, sent_bert_scores, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_all_io(alpha, mc_K, sent_bert_scores, sent_index)
        res += mc
    return res


# method_2
def mc_all(alpha, mc_K, sent_bert_scores, sent_index):
    return sum([(agg_all(Ki, alpha, sent_bert_scores, sent_index) + 1) for Ki in mc_K])


def score_all(Ki, mc_K, sent_bert_scores, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_all(alpha, mc_K, sent_bert_scores, sent_index)
        res += mc
    return res


#  method_3
def equal_head(alpha, beta):
    return alpha.split()[0] == beta.split()[0]


def equal_tail(alpha, beta):
    return alpha.split()[-1] == beta.split()[-1]


def H(Ki, alpha):
    h = []
    for beta in Ki:
        if equal_head(beta, alpha):
            h.append(beta)
    return h


def T(Ki, alpha):
    t = []
    for beta in Ki:
        if equal_tail(beta, alpha):
            t.append(beta)
    return t


def agg_local_H(Ki, alpha, sent_bert_scores, sent_index):
    h = H(Ki, alpha)
    if len(h) == 0:
        return 0
    else:
        return sum([sent_bert_scores[sent_index[alpha]][sent_index[beta]] for beta in h]) / len(h)


def agg_local_T(Ki, alpha, sent_bert_scores, sent_index):
    t = T(Ki, alpha)
    if len(t) == 0:
        return 0
    else:
        return sum([sent_bert_scores[sent_index[alpha]][sent_index[beta]] for beta in t]) / len(t)


def mc_local(alpha, Ki, mc_K, sent_bert_scores, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += ((agg_local_H(Ki, alpha, sent_bert_scores, sent_index) + agg_local_T(Ki, alpha, sent_bert_scores,
                                                                                        sent_index)) / 2 + 1)
    return res


def score_local(Ki, mc_K, sent_bert_scores, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_local(alpha, Ki, mc_K, sent_bert_scores, sent_index)
        res += mc
    return res


#  method_4
def Mk(k, Ki, alpha, sent_bert_scores, sent_index):
    sim_list = [sent_bert_scores[sent_index[alpha]][sent_index[beta]] for beta in Ki]
    sim_list.sort(reverse=True)
    mk = sim_list[:k]
    return mk


def agg_mink(k, Ki, alpha, sent_bert_scores, sent_index):
    mk = Mk(k, Ki, alpha, sent_bert_scores, sent_index)
    if len(mk) >= k:
        return sum([s for s in mk]) / k
    elif k > len(mk) > 0:
        return sum([s for s in mk]) / len(mk)
    else:
        return 0


def mc_mink(k, mc_K, alpha, sent_bert_scores, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += (agg_mink(k, Ki, alpha, sent_bert_scores, sent_index) + 1)
    return res


def score_mink(k, Ki, mc_K, sent_bert_scores, sent_index):
    res = 0
    t = 0
    for alpha in Ki:
        mc = mc_mink(k, mc_K, alpha, sent_bert_scores, sent_index)
        res += mc
    return res


def get_cosine_similarity(sent_index, K, mc_K):
    # 计算大矩阵的相似度
    sent_bert_scores = model1.similarity(K, K)
    print("similarity compute completed! ")
    print("Sentence-BERT ########################################")

    cnt = 1
    for Ki in mc_K:
        print("Method 1 K{0}  score: {1}".format(cnt, score_all_io(Ki, mc_K, sent_bert_scores, sent_index)))
        cnt += 1

    cnt = 1
    for Ki in mc_K:
        print("Method 2 K{0}  score: {1}".format(cnt, score_all(Ki, mc_K, sent_bert_scores, sent_index)))
        cnt += 1

    cnt = 1
    for Ki in mc_K:
        print("Method 3 K{0}  score: {1}".format(cnt, score_local(Ki, mc_K, sent_bert_scores, sent_index)))
        cnt += 1

    for k in range(2, 5):
        print("{0} ----------------".format(k))
        cnt = 1
        for Ki in mc_K:
            print("Method 4 K{0}  score: {1}".format(cnt, score_mink(k, Ki, mc_K, sent_bert_scores, sent_index)))
            cnt += 1

# get_cosine_similarity(sent_index, K, mc_K)
