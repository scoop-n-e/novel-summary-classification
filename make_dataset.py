import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from transformers import DataCollatorWithPadding

from sklearn.model_selection import train_test_split

import json
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_metric

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
        title = remove_linefeed(dic
        
        ["title"])
        summary = remove_linefeed(dic["story"])

        # タイトルとあらすじを結合
        sentence = f"{title} {summary}"

        # ラベルを0, 1, 2, 3, 4に変換
        label = conv_label(dic["genre"])
        
        labels.append(label)
        sentences.append(sentence)
    
    return {"input_ids":sentences, "labels":labels}

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
    ds = ds.train_test_split(test_size=0.3, shuffle=True)

    train_ds = ds["train"]
    eval_and_test = ds["test"].train_test_split(test_size=0.3, shuffle=True)
    eval_ds, test_ds = eval_and_test["train"], eval_and_test["test"]

    train_ds.to_json("./train.json")
    eval_ds.to_json("./eval.json")
    test_ds.to_json("./test.json")


    



if __name__ == "__main__":
    main()
