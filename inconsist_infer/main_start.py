# -*- coding:utf-8 -*-
# @Time    : 2022/5/12 19:39
# @Author  : Yinkai Yang
# @FileName: main_start.py
# @Software: PyCharm
# @Description: this is a program related to
import get_axiom_set as get_a
import get_cosine_score as get_c
import get_euclid_score as get_u

path = 'AUTOMCS/'
sent_index, _, K, mc_K = get_a.get_axioms_dictionary(path)
get_u.get_euclid_similarity(sent_index, K, mc_K)
get_c.get_cosine_similarity(sent_index, K, mc_K)
