import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from transformers import DataCollatorWithPadding

from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from datasets import load_metric
import pandas as pd


def main():
    test_ds = Dataset.from_json("test.json")

    model = AutoModelForSequenceClassification.from_pretrained("./outputs/checkpoint-10500/", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("./outputs/checkpoint-10500", model_max_length=128)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["input_ids"], truncation=True)

    test_ds = test_ds.map(preprocess_function, batched=True)

    trainer = Trainer(
        model=model,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    outputs = trainer.predict(test_ds)
    print(outputs)
    

    num = len(outputs.label_ids)
    cnt = 0
    for pred, label in zip(outputs.predictions.argmax(axis=-1), outputs.label_ids):
        if pred == label:
            cnt += 1

    print(cnt / num)

if __name__ == "__main__":
    main()
