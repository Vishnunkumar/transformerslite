from transformers import AutoTokenizer

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
