from transformers import TextClassificationPipeline


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
    model = AutoModelForSequenceClassification.from_pretrained("./outputs/checkpoint-10500/", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("./outputs/checkpoint-10500", model_max_length=128)
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    res = pipe("""ロスリックと呼ばれる王国には、世界の根源である、はじまりの火を継いだ『薪の王』たちの亡骸と彼らの故郷が流れ着く。ある時、はじまりの火が陰り、鐘が響き渡る。火が陰ったことで生死の境があいまいになり、死んでいた薪の王たちが蘇る。""")
    print(res)

if __name__ == "__main__":
    main()
