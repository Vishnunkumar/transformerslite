# TransformersProcessors
Data pre-processing for transformer models using simple python wrapper

## Implementation

```python

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from TransformersProcessors.processor import ClassificationProcessor

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=m)

max_seq_length = k
classification_processor = ClassificationProcessor(tokenizer, k)
tokenized_datasets = dataset.map(classification_processor.process, batched=True)

batch_size = 32
args = TrainingArguments(
    "sample",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    cmodel,
    args,
    train_dataset=encoded_datasets,
    tokenizer=tokenizer
)

trainer.train()
```
