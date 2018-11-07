#!/usr/bin/env python3
# coding=utf-8

import math
import numpy as np


def evaluate_model(sess, model, dataset, topk):
    hits, ndcgs = [], []
    for u, i in dataset.testPosSet:
        hit, ndcg = _evaluate_one_case(u, i, dataset.testPair2NegList,
                                       sess, model, topk)
        hits.append(hit)
        ndcgs.append(ndcg)
        # break
    return np.asarray(hits).mean(), np.asarray(ndcgs).mean()


def _evaluate_one_case(u, i, key2candidates, sess, model, topk):
    key = (u, i)
    assert (key in key2candidates)
    items = key2candidates[key]
    users = np.full(len(items), key[0], dtype=np.int32)
    predictions = sess.run(model.output, {
        model.user_indices: users,
        model.item_indices: items})

    # 预测出 items 的排名，选出最大的topk个
    topk = min(topk, len(items))
    sorted_idx = np.argsort(predictions)[::-1]
    selected_items = items[sorted_idx[0:topk]]
    # print(sorted_idx)
    # print(i,items[sorted_idx[0]])
    ndcg = _get_ndcg(selected_items, i)
    hit = _get_hit_ratio(selected_items, i)
    return hit, ndcg


def _get_hit_ratio(items, iid):
    return 1.0 if iid in items else 0.0


def _get_ndcg(items, iid):
    for i in range(len(items)):
        if items[i] == iid:
            return math.log(2) / math.log(i + 2)
    return 0.
