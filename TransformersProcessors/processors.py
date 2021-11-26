from transformers import AutoTokenizer

class ClassificationProcessor:

  def __init__(self, tokenizer, max_input_length):
    
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length

  def process(self, examples):

    self.examples = examples
    model_inputs = self.tokenizer(examples['sentence'], max_length=self.max_input_length, truncation=True, padding=True)

    return model_inputs

  
class Seq2SeqProcessor:

  def __init__(self, tokenizer, max_input_length, max_target_length, prefix):
    
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.prefix = prefix

  def process(self, examples):

    self.examples = examples
    inputs = [self.prefix + doc for doc in self.examples["source"]]
    model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True, padding=True)

    # Setup the tokenizer for targets
    with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(self.examples["target"], max_length=self.max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs 
  
  
