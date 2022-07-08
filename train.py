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
    train_ds = Dataset.from_json("train.json")
    eval_ds = Dataset.from_json("eval.json")
    test_ds = Dataset.from_json("test.json")

    model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese", model_max_length=128)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["input_ids"], truncation=True)

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    train_ds = train_ds.map(preprocess_function, batched=True)
    eval_ds = eval_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./outputs",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()    
    trainer.evaluate()
   
    #trainer.save_pretrained("./best")
    #AttributeError: 'Trainer' object has no attribute 'save_pretrained'
    outputs = trainer.predict(test_ds)
    print(outputs)
    preds_df = pd.DataFrame({
        'pred': outputs.predictions.argmax(axis=-1),
        'label': outputs.label_ids
    })
    preds_df.to_csv('./predict.csv')

if __name__ == "__main__":
    main()
