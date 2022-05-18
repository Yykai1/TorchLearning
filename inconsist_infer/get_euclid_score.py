# -*- coding:utf-8 -*-
# @Time    : 2022/5/13 18:50
# @Author  : Yinkai Yang
# @FileName: get_euclid_score.py
# @Software: PyCharm
# @Description: this is a program related to
from text2vec import SentenceModel, EncoderType
import time
import get_axiom_set as get_a

model = SentenceModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", encoder_type=EncoderType.MEAN)
# sent_index, _, K, mc_K = get_a.get_axioms_dictionary('AUTOMCS/')


def similar(v1, v2):
    return 1 / (1 + (sum((v1 - v2) ** 2)) ** 0.5)


def calculate_time(K, embeds, mat):
    start = time.time()
    for i in range(len(K)):
        for j in range(i, len(K)):
            mat[i][j] = similar(embeds[i], embeds[j])

    for i in range(len(K)):
        for j in range(i):
            mat[i][j] = mat[j][i]

    end = time.time()
    print(end - start)
    print("similarity compute completed! ")


# Method_1
def agg_all(Ki, alpha, mat, sent_index):
    res = 0
    for beta in Ki:
        res += mat[sent_index[beta]][sent_index[alpha]]
    return res / len(Ki)


def mc_all_io(alpha, mc_K, mat, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += agg_all(Ki, alpha, mat, sent_index)
        else:
            res -= agg_all(Ki, alpha, mat, sent_index)
    return res


def score_all_io(Ki, mc_K, mat, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_all_io(alpha, mc_K, mat, sent_index)
        res += mc
    return res


# Method_2
def mc_all(alpha, mc_K, mat, sent_index):
    return sum([(agg_all(Ki, alpha, mat, sent_index) + 1) for Ki in mc_K])


def score_all(Ki, mc_K, mat, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_all(alpha, mc_K, mat, sent_index)
        res += mc
    return res


# Method_3

def equalHead(alpha, beta):
    if len(alpha) == 0 and len(beta) == 0:
        return True
    if len(alpha) == 0 or len(beta) == 0:
        return True
    return alpha.split()[0] == beta.split()[0]


def equalTail(alpha, beta):
    if len(alpha) == 0 and len(beta) == 0:
        return True
    if len(alpha) == 0 or len(beta) == 0:
        return True
    return alpha.split()[-1] == beta.split()[-1]


def H(Ki, alpha):
    h = []
    for beta in Ki:
        if equalHead(beta, alpha):
            h.append(beta)
    return h


def T(Ki, alpha):
    t = []
    for beta in Ki:
        if equalTail(beta, alpha):
            t.append(beta)
    return t


def agg_local_H(Ki, alpha, mat, sent_index):
    h = H(Ki, alpha)
    if len(h) == 0:
        return 0
    else:
        return sum([mat[sent_index[alpha]][sent_index[beta]] for beta in h]) / len(h)


def agg_local_T(Ki, alpha, mat, sent_index):
    t = T(Ki, alpha)
    if len(t) == 0:
        return 0
    else:
        return sum([mat[sent_index[alpha]][sent_index[beta]] for beta in t]) / len(t)


def mc_local(alpha, mc_K, mat, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += ((agg_local_H(Ki, alpha, mat, sent_index) + agg_local_T(Ki, alpha, mat, sent_index)) / 2 + 1)
    return res


def score_local(Ki, mc_K, mat, sent_index):
    res = 0
    for alpha in Ki:
        mc = mc_local(alpha, mc_K, mat, sent_index)
        res += mc
    return res


# Method_4

def Mk(k, Ki, alpha, mat, sent_index):
    sim_list = [mat[sent_index[alpha]][sent_index[beta]] for beta in Ki]
    sim_list.sort(reverse=True)
    mk = sim_list[:k]
    return mk


def agg_mink(k, Ki, alpha, mat, sent_index):
    mk = Mk(k, Ki, alpha, mat, sent_index)
    if len(mk) >= k:
        return sum([s for s in mk]) / k
    elif len(mk) < k and len(mk) > 0:
        return sum([s for s in mk]) / len(mk)
    else:
        return 0


def mc_mink(k, alpha, mc_K, mat, sent_index):
    res = 0
    for Ki in mc_K:
        if alpha in Ki:
            res += (agg_mink(k, Ki, alpha, mat, sent_index) + 1)
    return res


def score_mink(k, Ki, mc_K, mat, sent_index):
    res = 0
    t = 0
    for alpha in Ki:
        mc = mc_mink(k, alpha, mc_K, mat, sent_index)
        res += mc
    return res


def get_euclid_similarity(sent_index, K, mc_K):
    mat = [([0] * len(K)) for i in range(len(K))]
    embeds = []
    for sentence in K:
        embeds.append(model.encode(sentence, show_progress_bar=True))
    calculate_time(K, embeds, mat)

    print("Sentence-BERT Euclid########################################")

    start = time.time()
    cnt = 1
    for Ki in mc_K:
        print("Method 1 K{0}  score: {1}".format(cnt, score_all_io(Ki, mc_K, mat, sent_index)))
        cnt += 1
    end = time.time()
    print(end - start)

    start = time.time()
    cnt = 1
    for Ki in mc_K:
        print("Method 2 K{0}  score: {1}".format(cnt, score_all(Ki, mc_K, mat, sent_index)))
        cnt += 1
    end = time.time()
    print(end - start)

    start = time.time()
    cnt = 1
    for Ki in mc_K:
        print("Method 3 K{0}  score: {1}".format(cnt, score_local(Ki, mc_K, mat, sent_index)))
        cnt += 1
    end = time.time()
    print(end - start)

    for k in range(2, 5):
        print("{0} ----------------".format(k))
        cnt = 1
        start = time.time()
        for Ki in mc_K:
            print("Method 4 K{0}  score: {1}".format(cnt, score_mink(k, Ki, mc_K, mat, sent_index)))
            cnt += 1
        end = time.time()
        print(end - start)

# get_euclid_similarity(sent_index,K,mc_K)
