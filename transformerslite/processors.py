from transformers import AutoTokenizer

class ClassificationProcessor:
  """
  Processor for Text Classification
  """
  
  def __init__(self, model_name, max_input_length):
    
    self.model_name = model_name
    self.max_input_length = max_input_length
    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    return tokenizer
  
  
  def process(self, examples, self.tokenizer):

    self.examples = examples
    model_inputs = self.tokenizer(examples['sentence'], max_length=self.max_input_length, truncation=True, padding=True)

    return model_inputs

  
class T5Seq2SeqProcessor:
  """
  Processor for T5 models
  """
  
  def __init__(self, max_input_length, max_target_length, prefix):
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.prefix = prefix
    
    return tokenizer

  def process(self, examples, tokenizer):

    self.examples = examples
    self.tokenizer = tokenizer
    
    inputs = [self.prefix + doc for doc in self.examples["source"]]
    model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True, padding=True)

    # Setup the tokenizer for targets
    with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(self.examples["target"], max_length=self.max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs 
  
  
class LanguageModelProcessor:
  """
  Processor for T5 models
  """

  def __init__(self, model_name, max_input_length):
    
    self.model_name = model_name
    self.max_input_length = max_input_length
    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    return tokenizer
  
  def process(self, examples, self.tokenizer):

    self.examples = examples
    model_inputs = self.tokenizer(examples['sentence'], max_length=self.max_input_length, truncation=True, padding=True)

    return model_inputs
