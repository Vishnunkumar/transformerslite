# transformerslite
Train simple lite transformer models in few lines of code

## Implementation

- **Sequence Classification** [bert-base-uncased](https://huggingface.co/transformers/pretrained_models.html)

```python
from transformerslite import pipeline
from datasets import load_dataset

# mandatory to provide valid and train files for now
data = load_dataset('csv', data_files={
    "train": "hg.csv",
    "valid": "hg2.csv"
})


training_pipeline = pipeline.SeqClassifier(data, 
                                           epochs=4, 
                                           max_input_length=32, 
                                           batch_size=1,
                                           learning_rate=0.0001, 
                                           num_class=2)
trainer, tokenizer = training_pipeline.model()
trainer.train()
```

- **Sequence to Sequence Modeling** [t5-small](https://huggingface.co/transformers/pretrained_models.html)

```python
from transformerslite import pipeline
from datasets import load_dataset

# mandatory to provide valid and train files for now
data = load_dataset('csv', data_files={
    "train": "hg.csv",
    "valid": "hg2.csv"
})


training_pipeline = pipeline.T5Seq2Seq(data,
                                       max_input_length=32,
                                       max_target_length=32, 
                                       prefix='seq: ',
                                       epochs=4, 
                                       batch_size=1,
                                       learning_rate=0.0001)

trainer, tokenizer = training_pipeline.model()
trainer.train()
```


