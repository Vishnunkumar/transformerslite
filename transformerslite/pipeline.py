from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

class ClassificationProcessor:
  """
  Processor for Text Classification
  """
  
  def __init__(self, max_input_length):
    
    self.max_input_length = max_input_length
  
  def process(self, examples):

    self.examples = examples

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model_inputs = tokenizer(examples['sentence'], max_length=self.max_input_length, truncation=True, padding=True)

    return model_inputs
  
  def tokenizerfunction(self):

    return AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


class T5Seq2SeqProcessor:
  """
  Processor for T5 models
  """
  
  def __init__(self, max_input_length, max_target_length, prefix):
    
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.prefix = prefix
    
  def process(self, examples):

    self.examples = examples

    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    inputs = [self.prefix + doc for doc in self.examples["source"]]
    model_inputs = tokenizer(inputs, max_length=self.max_input_length, truncation=True, padding=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(self.examples["target"], max_length=self.max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs 
  
  def tokenizer_function(self):
    
    return AutoTokenizer.from_pretrained("t5-small", use_fast=True)
  

class LanguageModelProcessor:
  """
  Processor for T5 models
  """

  def __init__(self, max_input_length):
    
    self.max_input_length = max_input_length
  
  def process(self, examples):

    self.examples = examples
    tokenizer = AutoTokenizer.from_pretrained("gpt-2", use_fast=True)
    model_inputs = tokenizer(examples['sentence'], max_length=self.max_input_length, truncation=True, padding=True)

    return model_inputs

  def tokenizer_function(self):

    return AutoTokenizer.from_pretrained("gpt-2", use_fast=True)


class SeqClassifier:

    def __init__(self, dataset, epochs, max_input_length, batch_size, learning_rate, num_class):

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_class = num_class
        self.max_input_length = max_input_length
        
    def fit(self):

        preprocessor = ClassificationProcessor(max_input_length = self.max_input_length)
        encoded_datasets = self.dataset.map(preprocessor.process, batched=True)
        columns_to_return = ['input_ids', 'labels', 'attention_mask']
        encoded_datasets.set_format(type='torch', columns=columns_to_return)

        tokenizer = preprocessor.tokenized_function()
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
        num_labels=self.num_class)

        args = TrainingArguments(
            "seq-classification",
            learning_rate=self.learning_rate,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            fp16=False,
            push_to_hub=False,
        )

        trainer = Trainer(
                model,
                args,
                train_dataset=encoded_datasets['train'],
                eval_dataset=encoded_datasets['valid'],
                tokenizer=tokenizer
        )
        
        return trainer.train()


class T5Seq2Seq:

    def __init__(self, dataset, max_input_length, max_target_length, prefix,
    epochs, batch_size, learning_rate):

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length 
        self.prefix = prefix

    def fit(self):

        preprocessor = T5Seq2SeqProcessor(max_input_length = self.max_input_length, 
        max_target_length=self.max_target_length, prefix=self.prefix)

        encoded_datasets = self.dataset.map(preprocessor.process, batched=True)
        columns_to_return = ['input_ids', 'labels', 'attention_mask']
        encoded_datasets.set_format(type='torch', columns=columns_to_return)
        
        tokenizer = preprocessor.tokenized_function()
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        args = Seq2SeqTrainingArguments(
            "seq-seq-t5",
            learning_rate=self.learning_rate,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        trainer = Seq2SeqTrainer(
                model,
                args,
                train_dataset=encoded_datasets['train'],
                eval_dataset=encoded_datasets['valid'],
                data_collator=data_collator,
                tokenizer=tokenizer
        )
        
        return trainer.train()
