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
    #bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese", model_max_length=128)
    model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=5)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["input_ids"], truncation=True)



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
    ds = ds.map(preprocess_function, batched=True)
    ds = ds.train_test_split(test_size=0.2, shuffle=True)
    print(ds)
    print(ds["train"])
    print(ds["test"])

    del data
    del dataset

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()    
    trainer.evaluate()    


if __name__ == "__main__":
    main()
