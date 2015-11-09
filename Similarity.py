#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from scipy.spatial.distance import cosine


def calc_item_score(target_user_index, user_rating_matrix):
    target_user_ratings = user_rating_matrix[target_user_index]
    item_similarity = numpy.zeros(len(target_user_ratings))
    for compare_user_index in range(len(user_rating_matrix)):
        compare_user_ratings = user_rating_matrix[compare_user_index]
        # print("compare_user_ratings ", compare_user_ratings)
        if compare_user_index == target_user_index:
            # 同一ユーザー
            continue
        # ユーザーの類似度をコサイン距離類似度から求める
        user_similarity = 1.0 - cosine(
            target_user_ratings, compare_user_ratings)
        # ユーザーの類似度×アイテム評価
        item_similarity += user_similarity * compare_user_ratings
    return item_similarity


def get_rep_forecast(target_user_index, user_rating_matrix):
    item_similarity = calc_item_score(target_user_index, user_rating_matrix)
    target_user_ratings = user_rating_matrix[target_user_index]
    for item_index in range(len(target_user_ratings)):
        if 0 != target_user_ratings[item_index]:
            target_user_ratings[item_index] = item_similarity[item_index]
    return(target_user_ratings)


def calc_user_score(target_user_index, user_rating_matrix):
    target_user_ratings = user_rating_matrix[target_user_index]
    user_similarity = numpy.zeros(len(user_rating_matrix))
    for compare_user_index in range(len(user_rating_matrix)):
        compare_user_ratings = user_rating_matrix[compare_user_index]
        # print("compare_user_ratings ", compare_user_ratings)
        if compare_user_index == target_user_index:
            # 同一ユーザー
            continue
        # ユーザーの類似度をコサイン距離類似度から求める
        user_similarity[compare_user_index] = 1.0 - cosine(
            target_user_ratings, compare_user_ratings)
    return user_similarity


def most_simil_index(target_user_index, user_rating_matrix):
    user_similarity = calc_user_score(target_user_index, user_rating_matrix)
    # もっとも類似度の高いユーザーのインデックスを返す
    return(numpy.argmax(user_similarity))


# ユーザーを次元、評価値を各次元の値
R = numpy.array([
    [5, 3, 0, 0],
    [4, 0, 4, 1],
    [1, 1, 0, 5],
    [0, 0, 4, 4],
    [0, 1, 5, 4],
    ])

R_ = R.T


if __name__ == "__main__":
    predict_ratings = calc_item_score(0, R)
    print("predict_ratings ", predict_ratings)

    # ユーザーの類似度配列取得
    user_similarity = calc_user_score(0, R)
    print("user_similarity", user_similarity)

    simil_user_index = most_simil_index(0, R)
    print("simil_user_index ", simil_user_index)

