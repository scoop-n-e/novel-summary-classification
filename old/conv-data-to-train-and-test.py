#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
from sklearn.model_selection import train_test_split


# 改行を削除
# タイトル、あらすじに含まれる改行を削除するために使う
def remove_linefeed(s):
    return "".join(s.splitlines())


# ラベルを0, 1, 2, 3, 4に変換
def conv_label(genre):
    genres = [201, 303, 401, 402, 404]
    return genres.index(genre)


# データの重複チェック
def duplicate_check(data):
    data_num = len(data)
    print(data_num)

    s = set()
    for dic in data:
        s.add(dic["ncode"])

    ncode_num = len(s)
    print(ncode_num)

    # 何故か重複あり
    #assert data_num == ncode_num


def conv_format(data):
    labels = []
    sentences = []

    for dic in data:
        # 改行を削除
        title = remove_linefeed(dic["title"])
        summary = remove_linefeed(dic["story"])

        # タイトルとあらすじを結合
        sentence = f"{title} {summary}"

        # ラベルを0, 1, 2, 3, 4に変換
        label = conv_label(dic["genre"])
        
        labels.append(label)
        sentences.append(sentence)
    
    return {"label" : labels, "sentence":sentences}


def main():
    data = []

    # 書き出したjsonファイルから読み込み
    with open("./data.json", "r") as f:
        data = json.load(f)

    # 重複するデータがないか確認
    duplicate_check(data)

    # 学習用のフォーマットに変換
    dataset = conv_format(data)
    
    #print(dataset)
    from datasets import Dataset
    ds = Dataset.from_dict(dataset)
    print(ds)
    ds = ds.train_test_split(test_size=0.2, shuffle=True)
    print(ds)
    print(ds["train"])
    print(ds["test"])
    """
    train_data, other_data = train_test_split(data_set, train_size=0.7, shuffle=True)
    eval_data, test_data = train_test_split(data_set, train_size=0.7, shuffle=True)

    # データ数確認
    print("train_size:", len(train_data))
    print("eval_size:", len(eval_data))
    print("test_size:", len(test_data))

    # 学習用データ書き出し
    with open("./train-data.json", "w") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open("./eval-data.json", "w") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)
    with open("./test-data.json", "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    """

if __name__ == "__main__":
    main()
