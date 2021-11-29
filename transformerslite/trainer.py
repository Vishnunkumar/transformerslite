from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from processor import *

class SeqClassifier:

    def __init__(self, dataset, epochs, batch_size, learning_rate):

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def fit(self, max_input_length):
        
        self.max_input_length = max_input_length
        preprocessor = Class
        
        
        
