import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from transformers import DataCollatorWithPadding

from sklearn.model_selection import train_test_split

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, path, transform1 = None, transform2 = None, train = True):
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train

        self.labelset = torchvision.datasets.CIFAR10(root = path, train = self.train, download = True)
        self.dataset = torchvision.datasets.CIFAR10(root = path, train = self.train, download = True)

        self.datanum = len(dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_label = self.labelset[idx][0]
        out_data = self.dataset[idx][0]

        if self.transform1:
            out_label = self.transform1(out_label)

        if self.transform2:
            out_data = self.transform2(out_data)

        return out_data, out_label



#bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=5)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""
dataset = []
with open("./dataset.json", "r") as f:
    dataset = json.load(f)[:10]
dataset = list(map(lambda data:{"label":data[1], "text": data[0]}, dataset))


from datasets import load_dataset

dataset = load_dataset('json', data_files={
    'train': 'train-data.json',
    'eval': 'eval-data.json',
    'test' : 'test-data.json',
}).map(lambda example: {'label': example[1], "sentence":example[0]})
"""


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

dataset = list(map(preprocess_function, dataset))
print(dataset)

train_data, other_data = train_test_split(dataset, train_size=0.7, shuffle=True)
eval_data, test_data = train_test_split(dataset, train_size=0.7, shuffle=True)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()